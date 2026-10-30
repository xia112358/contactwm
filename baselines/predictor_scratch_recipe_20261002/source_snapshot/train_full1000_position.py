#!/usr/bin/env python3
"""Train the predictor on complete, self-observed demonstration actions."""
import argparse
import atexit
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
import hashlib
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import tyro
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
from generator.full1000_position_predictor import Full1000PositionPredictor
from generator.conditioned_pose_predictor import ConditionedPosePrediction, conditioned_pose_objective
from generator.contact_location_predictor import contact_points_to_observed_heightmap_map
from generator.planned_contact_predictor import (audit_intended_surfaces, contact_plan_objective,
    plan_points_in_pose_frame, relative_contact_layout_loss)
from contact_solver.newton_witness_loss import query_local_distances
from contact_solver.contact_layout import relative_plan_objective, observed_endpoint_targets, persistent_role_consistent, representative_pairs, relative_layout_statistics
from generator.next_interaction_heightmap_v2 import _quaternion_multiply_wxyz, render_root_yaw_box_heightmaps
from contact_solver.interaction_acceptance import contact_acceptance, DEFAULT_ACCEPTANCE
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.robot_assets import encode_robot_asset_json, decode_robot_asset_json
from somaforge_core.newton_contact_data import require_current_newton_manifest
from generator.full1000_training_variants import (event_consistent_roles,
    teacher_mask_for_batch, training_course)
from generator.rollout_termination import TerminationLimits, physical_reset_masks, demonstration_loss
from generator.parallel_rollout_pool import rollout_transition
from generator.training_data import prepare, take
from generator.predictor_initialization import initialize_predictor, initialize_training_joint_bias, state_fingerprint
from generator.scene_workers import Workers
import somaforge_core.newton_scene_router as router


@dataclass
class Config:
    query_checkpoint: Path = ROOT / 'runtime/current/holosoma/logs/WholeBodyTracking/20260727_085520-g1_29dof_wbt_single_climb00_completionema_horizon50_ncon160_from4k_to10k-locomotion/model_06000.pt'
    query_source_manifest: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/versions/native_support_edit207_20261001/scenes/0e21fa0c89cb673d/manifest.json'
    query_source_model: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/versions/native_support_edit207_20261001/scenes/0e21fa0c89cb673d/model.json'
    manifest: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_demo_observed207_geometry_v2_20261002/manifest.json'
    data_cache: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_demo_observed207_geometry_v2_20261002/data.pt'
    q_cache: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_demo_observed207_geometry_v2_20261002/q_cache_64.npz'
    output: Path = ROOT / 'tmp/predictor_scratch_recipe_new_run'
    initialization: Literal['scratch'] = 'scratch'
    scratch_joint_mean_bias: bool = True
    reference_warmup_steps: int = 100
    mode: str = 'full'
    overfit_motion_id: int | None = None
    steps: int = 1100
    schedule_steps: int = 1100
    audit_checkpoint: Path | None = None  # Read-only gradient audit; never training initialization.
    query_worlds: int | None = 16
    gpu_pipeline: bool = True
    execution_only: bool = False
    event_roles: bool = True
    event_endpoint_gate: bool = False
    pending_plan_repair: bool = False
    anchored_plan_execution: bool = False
    relative_plan_supervision: bool = False
    profile_components: bool = False
    defer_statistics: bool = True
    reuse_fk: bool = True
    performance_audit: bool = False
    compile_modules: bool = False
    compile_fk: bool = False
    width: int = 192
    layers: int = 3
    location_width: int = 32
    batch_size: int = 128
    learning_rate: float = 0.0003
    evaluation_every: int = 25
    seed: int = 20260917
    heightmap_supersample: int = 1
    missing_pair_approach_weight: float = 0.1
    relative_layout_weight: float = 1.0
    rollout_steps: int = 3
    rollout_batch_fraction: float = 0.0
    fixed_teacher_probability: float | None = None
    constant_learning_rate: bool = False
    consistent_event_roles: bool = True
    collision_aggregation: str = 'max'
    body_geometry: bool = False
    part_geometry: bool = False
    region_plan: bool = True
    unified_contact: bool = True
    recover_failed_states: bool = False
    rollout_steps_final: int | None = None
    rollout_curriculum_switch: int = 10
    recoverable_penetration_m: float = 0.03
    parallel_rollouts: bool = True
    parallel_pool_size: int | None = 128
    parallel_updates_per_step: int | None = 10
    parallel_episode_limit: int = 64
    parallel_retry_limit: int = 8
    parallel_allow_post_demo: bool = False
    training_prediction_budget: int | None = None
    execution_plan_gradients: bool = False
    execution_observation_gradients: bool = True
    pilot_samples: int | None = None
    pilot_visible_only: bool = False
    comparison_batch_order: bool = False
    constraint_training: bool = False
    constraint_validation: bool = False
    constraint_max_trials: int = 6
    constraint_root_cost: float = 100.0
    require_forward_progress: bool = False
    progress_window: int = 3
    progress_minimum_m: float = 0.03615079075098038
    static_gradient_audit: bool = False
    static_audit_samples: int = 32
    static_audit_pool: Path | None = None


def main():
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    official, remaining = parser.parse_known_args()
    cfg = tyro.cli(Config, args=remaining)
    if cfg.reference_warmup_steps < 0:
        raise ValueError('Reference warmup steps must be nonnegative')
    if cfg.execution_only or not cfg.execution_observation_gradients:
        raise ValueError('Scratch training requires shared observation gradients and a trainable planner')
    if cfg.audit_checkpoint is not None and not cfg.static_gradient_audit:
        raise ValueError('An audit checkpoint cannot initialize or resume training')
    if cfg.reference_warmup_steps and cfg.constraint_training:
        raise ValueError('Reference warmup is unsupported with constrained optimizer training')
    if cfg.recover_failed_states or cfg.pending_plan_repair:
        raise ValueError('Legacy recovery/plan-repair flags are unsupported; physically continuable raw-state feedback is now automatic.')
    if cfg.anchored_plan_execution and not (cfg.gpu_pipeline and cfg.parallel_rollouts):
        raise ValueError('Anchored endpoint execution requires parallel GPU training')
    if official.visualizer is not None or official.enable_cameras:
        raise ValueError('predictor training is headless')
    if cfg.query_worlds is None:
        cfg.query_worlds = 32 if cfg.gpu_pipeline else 1
    if cfg.query_worlds < 1:
        raise ValueError('query worlds must be positive')
    if cfg.performance_audit and not cfg.gpu_pipeline:
        raise ValueError('performance contact audit requires GPU witness pipeline')
    if cfg.mode not in ('full', 'overfit') or cfg.steps < 1:
        raise ValueError('invalid training mode/steps')
    if cfg.fixed_teacher_probability is not None and not 0 <= cfg.fixed_teacher_probability <= 1:
        raise ValueError('fixed teacher probability must be in [0,1]')
    if cfg.collision_aggregation not in ('max', 'body_mean_plus_max'):
        raise ValueError('unknown collision aggregation')
    if cfg.collision_aggregation != 'max' and not cfg.gpu_pipeline:
        raise ValueError('multi-body short test requires GPU witness pipeline')
    if cfg.recover_failed_states and not cfg.consistent_event_roles:
        raise ValueError('recovery requires event-consistent labels')
    if (cfg.region_plan or cfg.unified_contact) and not cfg.gpu_pipeline:
        raise ValueError('Regional objectives require fresh GPU Newton witness tensors')
    if cfg.parallel_rollouts and (not cfg.gpu_pipeline or cfg.execution_only):
        raise ValueError('Parallel rollouts require GPU pipeline; execution-only adaptation uses reference batches')
    if cfg.training_prediction_budget is not None and cfg.training_prediction_budget < 1:
        raise ValueError('Training prediction budget must be positive')
    if cfg.static_gradient_audit and (not cfg.gpu_pipeline or not cfg.region_plan or not cfg.unified_contact
                                    or cfg.static_audit_samples < 1 or cfg.audit_checkpoint is None):
        raise ValueError('Static contact audit requires a position checkpoint, GPU regional objective and positive sample count')
    if cfg.comparison_batch_order:
        torch.backends.mha.set_fastpath_enabled(False)
    if cfg.pilot_visible_only and cfg.pilot_samples is None:
        raise ValueError('pilot-visible-only requires an explicit pilot sample limit')
    if cfg.constraint_training:
        if not cfg.gpu_pipeline or cfg.parallel_rollouts or cfg.rollout_batch_fraction or cfg.unified_contact:
            raise ValueError('Constrained minibatch pilot requires GPU queries, reference batches, rollout-batch-fraction 0, and no unified-contact')
        torch.backends.mha.set_fastpath_enabled(False)
    cfg.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    device = torch.device('cpu' if official.cpu else official.device)
    manifest = json.loads(cfg.manifest.read_text())
    if cfg.pending_plan_repair and not (cfg.parallel_rollouts and cfg.gpu_pipeline and cfg.event_roles and cfg.region_plan):
        raise ValueError('Pending-plan repair requires parallel GPU regional event-role model')
    if cfg.pending_plan_repair and cfg.relative_plan_supervision:
        raise ValueError('Pending-plan trial requires body-relative cell supervision: --no-relative-plan-supervision')
    if cfg.event_endpoint_gate and not (cfg.event_roles and cfg.gpu_pipeline and cfg.parallel_rollouts):
        raise ValueError('Event endpoint trial requires explicit roles and parallel GPU training')
    require_current_newton_manifest(manifest, context='full1000 position training')
    model = Full1000PositionPredictor(cfg.width, cfg.layers, cfg.location_width,
                                     body_geometry=cfg.body_geometry, part_geometry=cfg.part_geometry,
                                     region_plan=cfg.region_plan, unified_contact=cfg.unified_contact,
                                     execution_plan_gradients=cfg.execution_plan_gradients,
                                     execution_observation_gradients=cfg.execution_observation_gradients,
                                     event_roles=cfg.event_roles).to(device)
    initial, loaded, fresh, warm_metadata = initialize_predictor(model,
        initialization=cfg.initialization, seed=cfg.seed)
    if cfg.static_gradient_audit:
        checkpoint = torch.load(cfg.audit_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get('schema') != 'full1000_position_predictor_v1':
            raise ValueError('Gradient audit requires a complete predictor checkpoint')
        decode_robot_asset_json(checkpoint['robot_asset_json'], context='gradient audit')
        model.load_state_dict(checkpoint['model'], strict=True)
        warm_metadata.update(initialization='read_only_audit', source=str(cfg.audit_checkpoint.resolve()),
            source_step=checkpoint['step'], predictor_checkpoints_loaded=[str(cfg.audit_checkpoint.resolve())])
    if cfg.execution_only and cfg.rollout_batch_fraction:
        raise ValueError('execution-only adaptation requires rollout-batch-fraction 0')
    payload = torch.load(cfg.data_cache, map_location='cpu', weights_only=False)
    assert payload['robot_asset_json'] == encode_robot_asset_json()
    inputs, target, scene, split, samples, _ = prepare(model, cfg, device)
    inputs = {key: value for key, value in inputs.items()
              if key in ('current_q', 'current_contact', 'current_anchor', 'heightmap')}
    train = split['overfit'] if cfg.mode == 'overfit' else split['train']
    validation = train if cfg.mode == 'overfit' else split['validation']
    if cfg.pilot_samples is not None:
        if cfg.pilot_samples < 1: raise ValueError('pilot-samples must be positive')
        train = train[torch.linspace(0, len(train)-1, min(len(train), cfg.pilot_samples), device=train.device).long()]
        validation = validation[torch.linspace(0, len(validation)-1, min(len(validation), cfg.pilot_samples), device=validation.device).long()]
    if cfg.scratch_joint_mean_bias and not cfg.static_gradient_audit:
        warm_metadata['joint_bias_initialization'] = initialize_training_joint_bias(model, target['q'], train)
    warm_metadata['joint_bias_initialization'] = warm_metadata.get('joint_bias_initialization')
    if cfg.parallel_rollouts:
        if cfg.parallel_pool_size is None: cfg.parallel_pool_size=cfg.batch_size
        if cfg.parallel_updates_per_step is None: cfg.parallel_updates_per_step=(len(train)+cfg.batch_size-1)//cfg.batch_size
        if cfg.parallel_pool_size < cfg.batch_size or cfg.parallel_updates_per_step < 1:
            raise ValueError('Parallel pool must fill a batch, with at least one update per reporting interval')
    bridges = {tuple(row) for row in manifest.get('evaluation_only_bridges', [])}
    if any((samples[i]['motion_id'], samples[i]['current_frame'], samples[i]['target_frame']) in bridges for i in train.tolist()):
        raise ValueError('evaluation-only standing bridge cannot enter training')
    data = payload['data']
    role = torch.zeros_like(target['planned_surface'])
    role[torch.as_tensor(data['end_contact'], device=device).bool()] = 3
    role[torch.as_tensor(data['persistent_contact'], device=device).bool()] = 2
    role[torch.as_tensor(data['interaction_touchdown'], device=device).bool()] = 1
    target['role'] = role
    assert torch.equal(role != 0, target['planned_contact'])
    ids = torch.unique(torch.cat((train, validation))).tolist()
    heights = sorted({round(float(samples[i]['height']), 8) for i in ids})
    workers = Workers(cfg.output / 'newton_workers', heights, str(device),
        checkpoint=cfg.query_checkpoint, source_manifest=cfg.query_source_manifest, source_model=cfg.query_source_model, query_worlds=cfg.query_worlds,
                      tensor_transport=cfg.gpu_pipeline).__enter__()
    atexit.register(workers.__exit__, None, None, None)
    router.configured_router = lambda: workers.router
    tensor_scene_ids = torch.full((len(samples),), -1, dtype=torch.long, device=device)
    for i in ids:
        height = min(heights, key=lambda h: abs(h - float(samples[i]['height'])))
        target['newton_model_fingerprint'][i] = torch.tensor(list(bytes.fromhex(workers.entries[height]['fp'])), dtype=torch.uint8, device=device)
        if cfg.gpu_pipeline:
            tensor_scene_ids[i] = workers.entries[height]['scene_id']
    basis, origin = target['newton_world_basis'], target['newton_world_origin']
    frame_quat = torch.as_tensor(Rotation.from_matrix(basis.cpu().numpy()).as_quat()[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)

    def to_local(q, indices):
        position = torch.einsum('bij,bj->bi', basis[indices].transpose(1, 2), q[:, :3] - origin[indices])
        inverse = frame_quat[indices].clone(); inverse[:, 1:] *= -1
        return torch.cat((position, _quaternion_multiply_wxyz(inverse, q[:, 3:7]), q[:, 7:]), -1)

    def points_world(points, indices):
        return origin[indices, None] + torch.einsum('bij,bpj->bpi', basis[indices], points)

    def points_local(points, indices):
        return torch.einsum('bij,bpj->bpi', basis[indices].transpose(1, 2), points - origin[indices, None])

    static_query_snapshot = None

    def query(q, indices, qlocal=None):
        nonlocal static_query_snapshot
        if qlocal is None:
            qlocal = to_local(q, indices)
        if cfg.gpu_pipeline:
            from contact_solver.device_contact_objective import DeviceWitnessRows
            # A frozen audit compares derivatives at the exact same physical
            # pose. Reuse one real query only when q and scene rows are bitwise
            # identical; avoid solver candidate-order jitter between variants.
            reuse = (cfg.static_gradient_audit and static_query_snapshot is not None
                and torch.equal(indices, static_query_snapshot[0])
                and torch.equal(q.detach(), static_query_snapshot[1]))
            if reuse:
                observed = static_query_snapshot[2]
            else:
                observed = workers.tensor_router(q.detach(), tensor_scene_ids[indices])
                if cfg.static_gradient_audit:
                    static_query_snapshot = (indices.detach().clone(), q.detach().clone(), observed)
            rows = DeviceWitnessRows(model.fk, qlocal, observed,
                world_frame=(origin[indices], basis[indices]), collision_aggregation=cfg.collision_aggregation)
            return rows, observed
        return query_local_distances(model.fk, qlocal, take(scene, indices),
            world_frame=(origin[indices], basis[indices]), fingerprints=target['newton_model_fingerprint'][indices])

    # Keep exact native q and one fixed physical scene per motion, including
    # historical intermediate standing intervals. No generated q or reset.
    native_q = torch.empty_like(inputs['current_q'])
    native_target_q = torch.empty_like(native_q)
    global_directions = torch.empty((len(samples),2),device=device)
    first_by_motion = {}
    for motion, entry in enumerate(manifest['motion_files']):
        rows = [i for i, sample in enumerate(samples) if sample['motion_id'] == motion]
        if not rows: continue
        first_by_motion[motion] = min(rows, key=lambda i: samples[i]['current_frame'])
        with np.load(entry['motion_file'], allow_pickle=False) as archive:
            decode_robot_asset_json(archive['robot_asset_json'], context='full1000 canonical motion')
            native_q[rows] = torch.as_tensor(archive['joint_pos'][[samples[i]['current_frame'] for i in rows]], device=device)
            native_target_q[rows] = torch.as_tensor(archive['joint_pos'][[samples[i]['target_frame'] for i in rows]], device=device)
            global_directions[rows] = torch.as_tensor(archive['joint_pos'][-1,:2]-archive['joint_pos'][0,:2],device=device)
    fixed_scene = torch.tensor([first_by_motion[s['motion_id']] for s in samples], device=device)
    center_w = origin + torch.einsum('bij,bj->bi', basis, scene['box_center'])
    rotation_w = basis @ scene['box_rotation']
    ground_w = origin[:, 2] + scene['ground_height']
    region_label_metadata = None
    if cfg.region_plan:
        region_labels = torch.zeros(len(samples), 6, 4, dtype=torch.bool, device=device)
        audited_pairs = []
        def cpu_tree(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu()
            if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
            if isinstance(value, (list, tuple)): return [cpu_tree(v) for v in value]
            return value
        with torch.no_grad():
            for indices in torch.tensor(ids, device=device).split(cfg.batch_size):
                rows, observed = query(native_target_q[indices], indices)
                region_labels[indices] = model.region_geometry.actual_mask(model.fk, rows, target['planned_surface'][indices])
                audited_pairs.append({'indices': indices.cpu(), 'observed': cpu_tree(observed)})
        target['region_label'] = region_labels
        target['region_label_known'] = region_labels.any(-1) & target['planned_contact']
        # Initialize the new region decision from training labels only, so
        # zero logits do not impose an arbitrary all-regions-flat plan.
        bits = (region_labels[train].long()*(2**torch.arange(4, device=device))).sum(-1)
        prior_counts = torch.zeros(6, 15, device=device)
        with torch.no_grad():
            for part in range(6):
                known = target['region_label_known'][train, part]
                prior_counts[part] = torch.bincount(bits[known, part]-1, minlength=15)
            if not initial.get('config', {}).get('region_plan', False):
                model.region_prior.copy_((prior_counts+1).log())
        region_label_metadata = {'schema': model.region_geometry.schema,
            'source': 'fresh actual Newton active+allocated primary-upward-face target witnesses',
            'unknown_required_parts': int((target['planned_contact'][ids] & ~target['region_label_known'][ids]).sum()),
            'known_required_parts': int(target['region_label_known'][ids].sum()),
            'initial_region_prior_training_counts': prior_counts.cpu().tolist(),
            'unknown_policy': 'reported and excluded from region classification supervision; never geometric fallback',
            'sampling': 'fresh static query at native target q; no integration or temporal mixing'}
        torch.save({'metadata': region_label_metadata, 'robot_asset_json': encode_robot_asset_json(),
            'labels': region_labels.cpu(), 'known': target['region_label_known'].cpu(),
            'native_target_q': native_target_q.cpu(), 'newton_batches': audited_pairs}, cfg.output/'region_labels.pt')
        print(json.dumps({'region_labels': region_label_metadata}), flush=True)

    def render(q, indices):
        fixed = fixed_scene[indices]
        if cfg.gpu_pipeline:
            from generator.next_interaction_heightmap_v2 import render_root_yaw_box_heightmaps_device
            return render_root_yaw_box_heightmaps_device(center_w[fixed], rotation_w[fixed],
                scene['box_half_extents'][fixed], ground_w[fixed], q.detach(), supersample=cfg.heightmap_supersample)
        result = render_root_yaw_box_heightmaps(center_w[fixed].cpu().numpy(), rotation_w[fixed].cpu().numpy(),
            scene['box_half_extents'][fixed].cpu().numpy(), ground_w[fixed].cpu().numpy(),
            q.detach().cpu().numpy(), supersample=cfg.heightmap_supersample)
        return torch.as_tensor(result, device=device)

    inputs['current_anchor'] = points_world(inputs['current_anchor'], torch.arange(len(samples), device=device))
    inputs['current_q'] = native_q
    inputs['heightmap'] = render(native_q, torch.arange(len(samples), device=device))
    anchors_w = points_world(target['anchor'], torch.arange(len(samples), device=device))
    timings = {}

    @contextmanager
    def timed(name):
        if cfg.profile_components and device.type == 'cuda': torch.cuda.synchronize(device)
        begun = time.perf_counter()
        yield
        if cfg.profile_components and device.type == 'cuda': torch.cuda.synchronize(device)
        timings[name] = timings.get(name, 0.) + time.perf_counter() - begun

    def labels_for(observation, indices, *, recursive=False):
        # Starting in contact does not cancel a demonstrated retouch event.
        roles = event_consistent_roles(target['role'][indices], observation['current_contact'])
        endpoint_points, persistent = observed_endpoint_targets(anchors_w[indices],
            observation['current_contact'], roles)
        _, cell, valid = contact_points_to_observed_heightmap_map(observation['current_q'],
            roles != 0, endpoint_points, observation['heightmap'])
        return {'role': roles, 'contact_cell': cell, 'contact_cell_valid': valid,
                'endpoint_points_world': endpoint_points, 'contact_keep_intent': persistent}

    def objective(prediction, observation, indices, labels):
        qlocal = to_local(prediction.qpos, indices)
        if cfg.reuse_fk:
            model.fk.begin_link_pose_cache(qlocal)
        try:
            return objective_impl(prediction, observation, indices, labels, qlocal)
        finally:
            if cfg.reuse_fk:
                model.fk.end_link_pose_cache()

    def objective_impl(prediction, observation, indices, labels, qlocal):
        with timed('newton_query'):
            queried = query(prediction.qpos, indices, qlocal if cfg.reuse_fk else None)
        if cfg.gpu_pipeline and cfg.region_plan:
            rows = queried[0]
            rows.contact_region_ids = model.region_geometry.witness_regions(model.fk, qlocal,
                rows.sample, rows.part, rows.points[:, 1])
        points_w = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
        surfaces = audit_intended_surfaces(points_w, prediction.conditioned_contact, queried[1])
        with timed('execution_objective'):
            _, metrics = conditioned_pose_objective(model, ConditionedPosePrediction(qlocal), take(target, indices),
                take(scene, indices), realization_contact=prediction.conditioned_contact, realization_surface=surfaces,
                missing_pair_approach_weight=cfg.missing_pair_approach_weight, query_override=queried)
        compatible = ((prediction.conditioned_contact == target['planned_contact'][indices]).all(-1)
            & ((surfaces == target['planned_surface'][indices]) | ~prediction.conditioned_contact).all(-1))
        with timed('relative_layout'):
            layout, layout_metrics = relative_contact_layout_loss(model, qlocal, points_local(points_w, indices),
                prediction.conditioned_contact, surfaces, queried[0], regions=prediction.planned_regions)
        # An endpoint predictor supplies neither an adjacent-pose trajectory nor
        # executed loads. Do not substitute a whole-action displacement loss.
        metrics['support_motion_evidence_known'] = torch.zeros_like(compatible, dtype=qlocal.dtype)
        supervised = labels.get('supervised', torch.ones(len(indices), device=device, dtype=torch.bool))
        with timed('planner_objective'):
            plan_loss, plan_metrics = (relative_plan_objective(prediction, labels, observation['heightmap'])
                if cfg.relative_plan_supervision else contact_plan_objective(prediction, labels))
        if cfg.execution_only:
            plan_loss = plan_loss.detach() * 0
        imitation = demonstration_loss(compatible * metrics['imitation_loss'], supervised)
        metrics['imitation_supervised'] = (compatible & supervised).float()
        execution_loss = metrics['own_plan_realization_loss']
        approach_loss = cfg.missing_pair_approach_weight * metrics['missing_pair_approach_loss']
        if cfg.unified_contact:
            from contact_solver.contact_regions import unified_region_objective
            with timed('unified_region_objective'):
                execution_loss, region_metrics = unified_region_objective(model, queried[0],
                    prediction.conditioned_contact, surfaces, take(scene, indices), prediction.planned_regions,
                    audit_path=cfg.output/'unknown_normals.jsonl')
            metrics.update(region_metrics)
            metrics['newton_invalid_fullbody_witnesses'] = torch.maximum(metrics['newton_invalid_fullbody_witnesses'], region_metrics['region_invalid_penetrating_normals'])
            metrics['newton_generation_accepted'] *= region_metrics['region_gradient_valid']
            metrics['newton_generation_valid'] *= region_metrics['region_gradient_valid']
            metrics['region_plan_safe'] = region_metrics['region_plan_realized']*(metrics['newton_generation_accepted'] > .5)
            metrics['optimized_execution_loss'] = execution_loss
            approach_loss = execution_loss*0
        if cfg.region_plan:
            bits = (target['region_label'][indices].long() * (2**torch.arange(4, device=device))).sum(-1)
            known = target['region_label_known'][indices]
            ce = torch.nn.functional.cross_entropy(prediction.region_logits.transpose(1, 2), (bits-1).clamp_min(0), reduction='none')
            region_ce = (ce*known).sum(-1)/known.sum(-1).clamp_min(1)
            if not cfg.execution_only:
                plan_loss = plan_loss+region_ce
            metrics['region_classification_loss'] = region_ce
            metrics['region_classification_correct'] = (((prediction.region_logits.argmax(-1)+1 == bits) | ~known).all(-1)).float()
        if cfg.pending_plan_repair:
            # Repair observations supervise the executor of stored intent,
            # never a fresh planner asked to predict the previous event again.
            plan_loss = plan_loss * labels.get('planner_weight', torch.ones_like(plan_loss))
        plan_loss = demonstration_loss(plan_loss, supervised)
        loss = (plan_loss + imitation + execution_loss
            + approach_loss + metrics['safety_loss']
            + cfg.relative_layout_weight * layout)
        if cfg.gpu_pipeline:
            from generator.next_interaction_heightmap import HEIGHTMAP_RESOLUTION_M
            scale = 2 * HEIGHTMAP_RESOLUTION_M
            own_error, own_complete, _ = queried[0].anchored_layout(
                points_w, prediction.conditioned_contact, surfaces, actual=True, regions=prediction.planned_regions)
            task_error, task_complete, _ = queried[0].anchored_layout(
                labels['endpoint_points_world'], target['planned_contact'][indices], target['planned_surface'][indices], actual=True)
            own_ok = own_complete & torch.isfinite(own_error) & (own_error <= scale**2)
            task_ok = task_complete & torch.isfinite(task_error) & (task_error <= scale**2)
            metrics['own_endpoint_layout_accepted'] = own_ok.float()
            metrics['task_endpoint_layout_accepted'] = task_ok.float()
            support_ok = persistent_role_consistent(prediction.conditioned_role, observation['current_contact'])
            metrics['persistent_role_accepted'] = support_ok.float()
            metrics['event_endpoint_accepted'] = (own_ok & task_ok & support_ok).float()
            metrics['issued_plan_position_loss'] = layout
            # The scene-fixed layout term above already optimizes this error.
            # Do not add it twice when the legacy anchored flag is set.
        if cfg.unified_contact:
            loss = loss*metrics['region_gradient_valid']
        if cfg.static_gradient_audit:
            # Expose the exact weighted terms only for frozen-model auditing.
            valid = metrics['region_gradient_valid']
            metrics['audit_planner'] = plan_loss*valid
            metrics['audit_imitation'] = imitation*valid
            metrics['audit_layout'] = cfg.relative_layout_weight*layout*valid
            metrics['audit_safety'] = metrics['safety_loss']*valid
        metrics.update(plan_metrics); metrics.update(layout_metrics)
        # Surface compatibility is loss-only, so replacing the two-way
        # classifier by thousands of cells cannot silence the motion prior.
        emitted_surfaces = audit_intended_surfaces(plan_points_in_pose_frame(observation['current_q'], prediction.contact_points_local),
                                                  prediction.contact, queried[1])
        metrics['plan_exact'] = ((prediction.contact == target['planned_contact'][indices]).all(-1)
            & ((emitted_surfaces == target['planned_surface'][indices]) | ~prediction.contact).all(-1)).float()
        metrics['constraint_prior'] = plan_loss + imitation + cfg.relative_layout_weight * layout
        metrics['loss'] = loss
        return loss, metrics, queried

    @torch.no_grad()
    def realized_layout(prediction, observation, queried):
        points = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
        surfaces = audit_intended_surfaces(points, prediction.conditioned_contact, queried[1])
        active = prediction.conditioned_contact
        if cfg.gpu_pipeline:
            return queried[0].realized_layout(points, active, surfaces)
        witnesses = torch.zeros_like(points); observed = torch.zeros_like(active)
        enabled, faces = active.cpu().tolist(), surfaces.cpu().tolist()
        for i, catalog in enumerate(queried[1]['surface_catalog_by_sample']):
            actual = select_contact_pairs([queried[1]['pairs'][i]], catalog)
            for part, pair in enumerate(representative_pairs(actual['contact_pairs'][0], enabled[i], faces[i])):
                if pair is not None:
                    witnesses[i, part] = points.new_tensor(pair['position_w']); observed[i, part] = True
        error, _, count = relative_layout_statistics(witnesses, points, observed)
        complete = active.any(-1) & ((observed | ~active).all(-1))
        return error, complete, count

    def fresh_observation(q, indices, observed, chosen):
        if cfg.gpu_pipeline:
            return {'current_q': q.detach(), 'current_contact': observed['contact_part_mask'][chosen],
                    'current_anchor': observed['contact_position_w'][chosen], 'heightmap': render(q, indices)}
        masks, points = [], []
        for i in chosen.tolist():
            selected = select_contact_pairs([observed['pairs'][i]], observed['surface_catalog_by_sample'][i])
            masks.append(selected['contact_part_mask'][0]); points.append(selected['contact_position_w'][0])
        return {'current_q': q.detach(), 'current_contact': torch.as_tensor(np.asarray(masks), device=device),
                'current_anchor': torch.as_tensor(np.asarray(points), device=device), 'heightmap': render(q, indices)}

    from contact_solver.predictor_constraints import CanonicalBoxGeometry
    rollout_geometry = CanonicalBoxGeometry() if cfg.parallel_rollouts or cfg.rollout_batch_fraction else None

    @torch.no_grad()
    def continuation(q, indices, queried, metric):
        if cfg.gpu_pipeline:
            actual_contact = queried[1]['contact_part_mask']
        else:
            actual_contact = torch.as_tensor(np.asarray([
                select_contact_pairs([pairs], catalog)['contact_part_mask'][0]
                for pairs, catalog in zip(queried[1]['pairs'], queried[1]['surface_catalog_by_sample'])
            ]), device=device, dtype=torch.bool)
        clear, _, _ = rollout_geometry.evaluate(model.fk, to_local(q.detach(), indices), take(scene, indices))
        depth_cm = torch.maximum(metric['newton_penetration_cm'], 100*(-clear.amin(-1)).clamp_min(0))
        invalid = metric['newton_invalid_fullbody_witnesses'].clone()
        if cfg.gpu_pipeline:
            pair = queried[0].pair
            # Missing solver fields are errors; unallocated activation is an
            # explicit invalid-state reset, never accepted as normal contact.
            active = ((pair['type'].long() & 1) != 0) & (pair['dist'] < pair['includemargin'])
            if not torch.equal(active, pair['active']):
                raise ValueError('Newton activation metadata mismatch')
            bad = active & ~pair['constraint_allocated']
            invalid.scatter_add_(0, pair['sample'], bad.to(invalid))
        continuable, reasons = physical_reset_masks(q, depth_cm, invalid,
            metric['joint_violation_rad'], actual_contact,
            TerminationLimits(penetration_m=cfg.recoverable_penetration_m))
        return continuable, reasons

    starts = {(s['motion_id'], s['current_frame']): i for i, s in enumerate(samples)}
    successor = torch.tensor([starts.get((s['motion_id'], s['target_frame']), -1) for s in samples], device=device)
    allowed_successors = torch.zeros(len(samples), device=device, dtype=torch.bool)
    allowed_successors[train] = True
    successor = torch.where((successor >= 0) & allowed_successors[successor.clamp_min(0)], successor, -1)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}
    dataset = {'schema': 'full1000_position_ablation_v1', 'original_timeline_preserved': bool(manifest.get('fixed_timeline_ablation')),
        'raster_label_semantics': 'nearest_observed_3d_cell_v1', 'forbidden_forward_inputs': ['surface ID', 'event index', 'future q'],
        'own_plan_realization_under_wrong_predicted_plan': True,
        'pose_realization_gradient_reaches_plan_heads': False,
        'execution_plan_gradient_contract': 'detached hard plan and encoder features; independent execution residuals, geometry encoder and output norm',
        'gradient_clipping': 'planner and executor parameter groups clipped independently at norm 10',
        'failure_states_used_for_training': bool(cfg.parallel_rollouts or cfg.rollout_batch_fraction), 'common_horizontal_translation_allowed_in_layout': False,
        'contact_position_contract': 'observed_endpoint_region_geometry_v3',
        'rollout_policy': (f'up to {cfg.rollout_steps} autonomous outputs per batch segment; keep physically continuable failures; no demonstration-end termination'
                           if cfg.rollout_batch_fraction else 'no autonomous rollout'),
        'source_reconstruction': 'archived full1000 config, teacher schedule and loss contract; not a byte-identical source restore',
        'runtime_difference': 'current Newton labels rebuilt on original boundaries',
        'position_interface_loss': 'scene-fixed witness positions plus unchanged Newton activation; no future-only translation exemption',
        'planner_supervision': 'actual demonstrated endpoints for every role; persistent activation is not a fixed point',
        'relative_layout_contract': 'newton_region_matched_witness_xy_v3',
        'execution_only_adaptation': cfg.execution_only,
        'initialization': warm_metadata,
        'inherited_pretraining': 'none; predictor training initializes without predictor checkpoints',
        'samples': len(samples), 'training_samples': len(train), 'validation_samples': len(validation)}
    dataset['reference_warmup'] = {
        'steps': cfg.reference_warmup_steps, 'sampling': 'one shuffled traversal of all training rows per interval',
        'plan': 'visible demonstrated role/contact/cell; region head remains supervised and predicted',
        'newton_objective_unchanged': True, 'generated_state_feedback': False,
        'optimizer_and_scheduler_restarted_at_transition': False,
        'teacher_schedule_clock': 'reporting interval minus reference_warmup_steps',
    }
    from somaforge_core.loaded_material_motion import unknown_material_path
    dataset['support_motion_contract'] = unknown_material_path()
    dataset['forward_progress_contract'] = dict(enabled=cfg.require_forward_progress,
        window_steps=cfg.progress_window, minimum_m=cfg.progress_minimum_m,
        direction='whole task start-to-end XY; fixed through each episode',
        metric='rolling increase in historical furthest root projection; reset history on physical reset',
        loss_changed=False)
    dataset['short_ablation_contract'] = {
        'anchored_plan_execution': cfg.anchored_plan_execution,
        'failure_contract': 'raw prediction feedback while physically continuable; reset on severe penetration, zero actual contacts, or invalid state; failed output contributes loss first',
        'pending_plan_repair': cfg.pending_plan_repair,
        'repair_contract': 'frozen issued world intent; no planner loss on repairs; own-complete wrong-task plans reset' if cfg.pending_plan_repair else None,
        'event_roles': cfg.event_roles, 'event_endpoint_gate': cfg.event_endpoint_gate,
        'event_gate_scope': 'region-matched scene endpoint and persistent role validity; temporal retouch is unverified',
        'event_layout_tolerance': '2 heightmap cells; separate from Newton contact truth',
        'consistent_event_roles': cfg.consistent_event_roles, 'body_geometry': cfg.body_geometry,
        'part_geometry': cfg.part_geometry,
        'region_plan': cfg.region_plan, 'unified_contact': cfg.unified_contact,
        'collision_aggregation': cfg.collision_aggregation, 'recovery_same_stage_until_completed': True,
        'recovery_penetration_limit_is_not_contact_truth': cfg.recoverable_penetration_m,
        'rollout_steps_initial': cfg.rollout_steps, 'rollout_steps_final': cfg.rollout_steps_final,
        'bptt': False, 'raw_q_feedback': True,
        'support_motion_objective': {
            'enabled': False, 'reason': 'endpoint_only_without_executed_trajectory',
            'substitute_endpoint_displacement_loss': False},
        'execution_gap_objective': {
            'schema': 'newton_gap_interval_v1',
            'upper': 'actual pair includemargin; independent of activation flags',
            'lower_m': -DEFAULT_ACCEPTANCE.shallow_penetration_m,
            'lower_penalty': 'full-body safety only; not duplicated in intended-contact term',
            'missing_pair': 'configured margin upper approach plus finite-face guidance',
            'contact_truth': 'unchanged Newton activation/allocation and primary-face selection'}}
    if cfg.parallel_rollouts:
        dataset['rollout_policy']='persistent independent GPU states; one batched endpoint prediction and optimizer update; no six-step unroll'
        dataset['parallel_rollout_contract']={'pool_size':cfg.parallel_pool_size,
            'updates_per_reporting_interval':cfg.parallel_updates_per_step,
            'age_histogram_cap':cfg.parallel_episode_limit,'episode_limit':cfg.parallel_episode_limit,'consecutive_retry_limit':None,
            'resample_interval_predictions':cfg.parallel_episode_limit,
            'resample_counter':'per slot; every prediction including failures; persists across reporting intervals',
            'demo_end_resets':not cfg.parallel_allow_post_demo,'post_demo_supervision':False,
            'allow_post_demo':cfg.parallel_allow_post_demo,
            'termination_penetration_m':cfg.recoverable_penetration_m,
            'epoch_boundaries_reset_pool':False,'cross_step_gradient':False,
            'reference_q_only_on_initialization_or_resample':True,
            'failure_feedback':'retain exact pre-prediction input and target within sampling window; rejected output still trains',
            'progress_history':'accepted transitions only','successor_must_be_training_row':True,
            'old_rollout_steps_and_fraction_are_unused':True,
            'plan_conditioning':'original full1000 teacher probability on each visible pool input; zero by interval 700 unless explicitly overridden',
            'teacher_changes_plan_only':True,
            'teacher_q_never_used_as_feedback':True,
            'loss_weight':'one mean loss over all slots per optimizer update',
            'progress_unit':'reporting intervals, not equivalent to old epochs'}
    if cfg.part_geometry:
        geometry = model.part_geometry_observation
        dataset['part_geometry_observation'] = {'schema': geometry.feature_schema,
            'features_per_part': geometry.feature_count, 'asset_counts': geometry.asset_counts,
            'injection': 'residual into each of six existing part tokens before terrain reasoning',
            'source_vertices_are_not_network_inputs': True, 'gap_is_not_contact_truth': True}
    if cfg.region_plan or cfg.unified_contact:
        dataset['contact_regions'] = {'schema': model.region_geometry.schema,
            'region_names': model.region_geometry.names, 'features_per_part': 32 if cfg.region_plan else 0,
            'raw_mesh_network_input': False, 'region_label_metadata': region_label_metadata,
            'unified_interval_objective': cfg.unified_contact,
            'interval_penalty_aggregation': 'robust_lower_plus_robust_upper_mean_plus_max_v2',
            'all_raw_fullbody_witnesses_checked': True,
            'proxy_is_only_missing_pair_approach_not_contact_truth': True,
            'invalid_penetrating_normal_policy': 'raw audit saved; entire sample excluded from gradients and rollout; never accepted as safe',
            'execution_gradient_reaches_new_region_head': False}
    warm_metadata['initial_model_sha256'] = state_fingerprint(model)
    (cfg.output / 'config.json').write_text(json.dumps(config, indent=2))
    if cfg.constraint_training:
        dataset['constraint_training'] = dict(schema='predictor_constraint_band_v1',
            contact_truth='current Newton CONSTRAINT, dist < includemargin, allocated, primary upward face',
            plan_conditioning='fixed demonstrated contact intent during each parameter proposal',
            geometry='canonical full-shape box/ground support plus Newton; conservative plane bounds',
            sampling='fresh static query at each candidate pose, no integration',
            keep_tangent_tolerance_m=.01, root_relative_cost=cfg.constraint_root_cost,
            max_candidate_queries=cfg.constraint_max_trials, pilot_samples=cfg.pilot_samples,
            optimizer='project actual AdamW increment; reject rolls back weights, moments and AL',
            inference='ordinary network forward; no projector')
        dataset['gradient_clipping'] = 'no gradient clipping; actual AdamW increment projected and backtracked'
    (cfg.output / 'dataset.json').write_text(json.dumps(dataset, indent=2))
    initialization_report = {**warm_metadata, 'loaded_parameters': loaded, 'fresh_parameters': fresh}
    (cfg.output / 'initialization.json').write_text(json.dumps(initialization_report, indent=2))
    (cfg.output / 'warm_start.json').write_text(json.dumps(initialization_report, indent=2))
    if cfg.static_gradient_audit:
        from generator.research.isolated_gradients import audit
        audit(model, cfg, inputs, validation, labels_for, objective)
        return  # Deliberately before optimizer construction, evaluation, or training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.schedule_steps, eta_min=cfg.learning_rate * .1)

    constrained_step = constraint_adapter = None
    if cfg.constraint_training:
        from contact_solver.constraint_training import ConstrainedOptimizerStep, StepConfig
        from contact_solver.predictor_constraints import PredictorConstraintAdapter
        from contact_solver.feasibility_filter import ProtectionRule
        constrained_step = ConstrainedOptimizerStep(model, optimizer,
            [ProtectionRule('contact/', 0.), ProtectionRule('keep/', 0.), ProtectionRule('collision/', 0.), ProtectionRule('joint/', 1e-6)],
            config=StepConfig(max_trials=cfg.constraint_max_trials))
        constraint_adapter = PredictorConstraintAdapter(root_relative_cost=cfg.constraint_root_cost)

    if cfg.constraint_validation and constraint_adapter is None:
        from contact_solver.predictor_constraints import PredictorConstraintAdapter
        constraint_adapter = PredictorConstraintAdapter(root_relative_cost=cfg.constraint_root_cost)

    def save(path, step, metrics, state=None):
        torch.save({'schema': 'full1000_position_predictor_v1', 'model': model.state_dict() if state is None else state,
                    'config': config, 'dataset': dataset, 'step': step, 'metrics': metrics,
                    'condition_contract': dataset, 'robot_asset_json': encode_robot_asset_json(),
                    'constraint_training_state': None if constrained_step is None or state is not None else constrained_step.state_dict(),
                    'optimizer': None if constrained_step is None or state is not None else optimizer.state_dict()}, path)

    @torch.no_grad()
    def evaluate(teacher=False):
        model.eval(); totals, maxima = {}, {}
        layout_errors, layout_safe_errors, visible_count = [], [], 0
        for indices in validation.split(cfg.batch_size):
            observation = take(inputs, indices)
            labels = labels_for(observation, indices)
            if teacher:
                visible = (labels['contact_cell_valid'] | ~target['planned_contact'][indices]).all(-1)
                indices = indices[visible]
                if not len(indices): continue
                observation = take(inputs, indices); labels = labels_for(observation, indices)
                prediction = model(**observation, teacher_contact=target['planned_contact'][indices],
                    teacher_cell=labels['contact_cell'], teacher_role=labels['role'], teacher_mask=torch.ones(len(indices), device=device, dtype=torch.bool))
            else:
                prediction = model(**observation)
            visible_count += len(indices)
            _, metrics, queried = objective(prediction, observation, indices, labels)
            if cfg.constraint_training or cfg.constraint_validation:
                checked = constraint_adapter(model, queried[0], take(scene, indices),
                    active=target['planned_contact'][indices], surface=target['planned_surface'][indices],
                    sample_ids=[str(int(i)) for i in indices], context_ids=['validation']*len(indices),
                    prior=prediction.qpos.sum()*0, nominal_q=target['q'][indices])
                depth = prediction.qpos.new_tensor([v['verified_penetration_m'] for v in checked.diagnostics.values()])
                from contact_solver.device_contact_objective import acceptance
                actual_ok = acceptance(queried[1], target['planned_contact'][indices], target['planned_surface'][indices])['contact_accepted']
                metrics['constraint_verified_penetration_cm'] = 100*depth
                metrics['constraint_contact_satisfied'] = (actual_ok & (depth <= DEFAULT_ACCEPTANCE.shallow_penetration_m)).float()
            error, complete, count = realized_layout(prediction, observation, queried)
            eligible = complete & (count >= 2)
            layout_errors.extend((100 * error[eligible].sqrt()).cpu().tolist())
            safe = eligible & (metrics['newton_generation_accepted'] > .5) & (metrics['joint_violation_rad'] <= .15)
            layout_safe_errors.extend((100 * error[safe].sqrt()).cpu().tolist())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value.sum())
                maxima[key] = max(maxima.get(key, -float('inf')), float(value.max()))
        if not visible_count: raise ValueError('no visible validation plans')
        return {**{key: value / visible_count for key, value in totals.items()}, 'maxima': maxima,
                'evaluated_samples': visible_count, 'excluded_out_of_view_samples': len(validation)-visible_count,
                'complete_actual_layout_samples': len(layout_errors),
                'actual_layout_rms_cm': float(np.mean(layout_errors)) if layout_errors else None,
                'safe_complete_actual_layout_samples': len(layout_safe_errors),
                'safe_actual_layout_rms_cm': float(np.mean(layout_safe_errors)) if layout_safe_errors else None}

    if cfg.compile_fk:
        model.fk._world_poses = torch.compile(model.fk._world_poses, dynamic=True)

    if cfg.compile_modules:
        # Compile pure neural submodules in place: preserve checkpoint keys and
        # keep Newton queries, contact assertions and stage control eager.
        for module in (model.height, model.location_key, model.shared,
                       model.plan_fusion, model.interaction_decoder):
            module.compile(dynamic=True)

    def statistic(value, *, integer=False):
        if cfg.defer_statistics:
            return value.detach().to(torch.int64 if integer else torch.float64)
        return int(value) if integer else float(value)

    fixed_labels = labels_for(inputs, torch.arange(len(samples), device=device))
    if cfg.pilot_visible_only:
        visible = (fixed_labels['contact_cell_valid'][train] | ~target['planned_contact'][train]).all(-1)
        excluded = train[~visible].tolist()
        train = train[visible]
        if not len(train): raise ValueError('Pilot has no visible training plans')
        dataset['training_samples'] = len(train)
        dataset['pilot_visibility_excluded'] = excluded
        (cfg.output/'dataset.json').write_text(json.dumps(dataset, indent=2))
    if cfg.pilot_samples is not None:
        (cfg.output/'pilot_selection.json').write_text(json.dumps(dict(
            train=train.tolist(), validation=validation.tolist(),
            excluded=dataset.get('pilot_visibility_excluded', []),
            motion_ids={str(int(i)):int(samples[int(i)]['motion_id']) for i in torch.cat((train, validation))}), indent=2))
    history, best_score, best_step, best_state = [], float('inf'), None, None
    performance_history = []
    begun = time.perf_counter()
    baseline = {'autonomous': evaluate(), 'teacher_conditioned': evaluate(teacher=True)}
    (cfg.output / 'initial_evaluation.json').write_text(json.dumps(baseline, indent=2))
    save(cfg.output / 'step_0000.pt', 0, baseline['autonomous'])
    print(json.dumps({'initial_evaluation': baseline}), flush=True)
    # Fixed GT observations have fixed labels; only recursive observations need re-rasterizing.
    fixed_labels = labels_for(inputs, torch.arange(len(samples), device=device))
    # Identical minibatch/mask draws across architectural ablations. New
    # module initialization must not silently alter the sampling sequence.
    torch.manual_seed(cfg.seed)
    pool=None
    pending=None
    training_predictions=0
    def make_pool():
        from generator.parallel_rollout_pool import ParallelRolloutPool
        return ParallelRolloutPool(inputs,train,successor,cfg.parallel_pool_size,
            recover_failed_states=cfg.recover_failed_states,max_episode_steps=cfg.parallel_episode_limit,
            max_retries=cfg.parallel_retry_limit,
            allow_post_demo=cfg.parallel_allow_post_demo,
            global_directions=global_directions if cfg.require_forward_progress else None,
            progress_window=cfg.progress_window,progress_minimum_m=cfg.progress_minimum_m)
    def record_training_diagnostics(metrics):
        for name in ('region_gradient_valid', 'imitation_supervised', 'newton_contact_accepted',
                     'newton_generation_accepted', 'plan_contact_exact', 'plan_role_exact'):
            if name in metrics:
                training_diagnostics[name] = training_diagnostics.get(name, 0.) + metrics[name].detach().sum()

    for step in range(1, cfg.steps + 1):
        if device.type == 'cuda': torch.cuda.synchronize(device)
        step_begun = time.perf_counter(); timings = {}
        model.train()
        reference_warmup, probability = training_course(step,
            reference_warmup_steps=cfg.reference_warmup_steps,
            execution_only=cfg.execution_only, fixed=cfg.fixed_teacher_probability)
        if cfg.parallel_rollouts and pool is None and not reference_warmup:
            pool = make_pool()
            if cfg.pending_plan_repair:
                from generator.pending_contact_plan import PendingContactPlans, planner_weights
                pending = PendingContactPlans(cfg.parallel_pool_size, device)
        if cfg.constraint_training:
            probability = 1.
        training_diagnostics = {}
        rollout_depth = (cfg.rollout_steps_final if cfg.rollout_steps_final is not None
            and step > cfg.rollout_curriculum_switch else cfg.rollout_steps)
        total_loss, total_count, teacher_count, rollout_outputs, phase_advances = 0., 0, 0, 0, 0
        combined_total, rollout_weighted_total, gradient_norm_total, batches = 0., 0., 0., 0
        constraint_accepted_updates = 0
        depth_outputs = [0] * rollout_depth
        failure_counts = {'own_contact': 0, 'own_safety': 0, 'task_contact': 0, 'joint_limit': 0,
                          'retries_used': 0, 'advances_used': 0, 'stopped': 0,
                          'severe_penetration_resets': 0, 'no_contact_resets': 0, 'invalid_resets': 0, 'no_forward_progress_resets': 0}
        if pool is not None:
            failure_counts = {key.replace('_resets', '_rejections'): value for key, value in failure_counts.items()}
        prediction_start=training_predictions
        pool_counts={key:torch.zeros((),device=device,dtype=torch.long) for key in
            ('generated_inputs','advances','retries','resets','terminal_resets','invalid_resets','timeout_resets',
             'periodic_resamples','rollbacks','accepted_transitions','invalid_rejections','severe_penetration_rejections',
             'no_contact_rejections','no_forward_progress_rejections',
             'demo_exhausted','autonomous_states','severe_penetration_resets','no_contact_resets','no_forward_progress_resets')}
        if pool is not None:
            from contact_solver.device_contact_objective import acceptance
            age_histogram=torch.zeros(cfg.parallel_episode_limit+1,device=device,dtype=torch.long)
            for update in range(cfg.parallel_updates_per_step):
                remaining=(cfg.training_prediction_budget-training_predictions if cfg.training_prediction_budget is not None else cfg.batch_size)
                if remaining <= 0: break
                count=min(cfg.batch_size,remaining)
                slots,indices,observation=pool.batch(count)
                ages=pool.age[slots]
                generated=(ages > 0).sum()
                age_histogram+=torch.bincount(ages.clamp_max(cfg.parallel_episode_limit),minlength=len(age_histogram))
                pool_counts['generated_inputs']+=generated
                labels=labels_for(observation,indices,recursive=True)
                labels['supervised']=pool.supervised[slots].clone()
                repair = pending.batch(slots) if pending is not None else {}
                if pending is not None:
                    labels['planner_weight'] = planner_weights(repair['repair_mask'])
                teacher=teacher_mask_for_batch(probability,target['planned_contact'][indices],labels['contact_cell_valid'])
                teacher &= labels['supervised']
                if pending is not None: teacher &= ~repair['repair_mask']
                optimizer.zero_grad(set_to_none=True)
                with timed('forward'):
                    prediction=model(**observation,teacher_contact=target['planned_contact'][indices],
                        teacher_cell=labels['contact_cell'],teacher_role=labels['role'],teacher_mask=teacher, **repair)
                loss,rm,queried=objective(prediction,observation,indices,labels)
                record_training_diagnostics(rm)
                combined=loss.mean()
                with timed('backward_optimizer'):
                    combined.backward()
                    gradient_norm=torch.stack(list(model.clip_training_gradients().values())).norm()
                    optimizer.step()
                gradient_norm_total+=statistic(gradient_norm);batches+=1
                total_count+=count;training_predictions+=count
                teacher_count+=statistic(teacher.sum(),integer=True)
                total_loss+=statistic(loss.detach().sum());combined_total+=statistic(loss.detach().sum())
                accepted=acceptance(queried[1],target['planned_contact'][indices],target['planned_surface'][indices])['contact_accepted']
                safe=(rm['newton_generation_accepted'] > .5) & accepted & (rm['joint_violation_rad'] <= .15)
                if cfg.region_plan and cfg.unified_contact: safe &= rm['region_plan_realized'] > .5
                if cfg.event_endpoint_gate: safe &= rm['event_endpoint_accepted'] > .5
                if cfg.gpu_pipeline: safe &= rm['event_endpoint_accepted'] > .5
                if pending is not None: safe &= rm['event_endpoint_accepted'] > .5
                recoverable, reset_reasons = continuation(prediction.qpos, indices, queried, rm)
                if pending is not None:
                    own_complete = ((rm['newton_generation_accepted'] > .5)
                        & (rm['region_plan_realized'] > .5) & (rm['own_endpoint_layout_accepted'] > .5))
                    # Completed wrong-task intent is not an unfinished plan.
                    # Reset rather than relabeling that endpoint as the old event.
                    recoverable &= ~own_complete
                with timed('pool_feedback'):
                    feedback=pool.update(slots,prediction.qpos,queried[1],safe,recoverable,fresh_observation,reset_reasons=reset_reasons)
                    if pending is not None:
                        issued_world = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
                        pending.update(slots, prediction, issued_world, pool.retries[slots] > 0)
                for key in pool_counts:
                    if key!='generated_inputs':pool_counts[key]+=feedback[key]
                phase_advances+=statistic(feedback['advances'],integer=True)
                failure_counts['own_contact']+=statistic((rm['newton_contact_accepted'] <= .5).sum(),integer=True)
                failure_counts['own_safety']+=statistic((rm['newton_generation_accepted'] <= .5).sum(),integer=True)
                failure_counts['task_contact']+=statistic(((~accepted) & labels['supervised']).sum(),integer=True)
                failure_counts['joint_limit']+=statistic((rm['joint_violation_rad'] > .15).sum(),integer=True)
                for reason, mask in reset_reasons.items():
                    failure_counts[reason+'_rejections'] += statistic(mask.sum(), integer=True)
                for key,source in [('retries_used','retries'),('advances_used','advances'),('stopped','resets')]:
                    failure_counts[key]+=statistic(feedback[source],integer=True)
            depth_outputs=age_histogram.cpu().tolist()
            rollout_outputs=sum(depth_outputs[1:])
        order_generator = torch.Generator(device=device).manual_seed(cfg.seed+step) if cfg.comparison_batch_order else None
        shuffled=(train[:0] if pool is not None else train[torch.randperm(len(train), device=device, generator=order_generator)])
        for indices in shuffled.split(cfg.batch_size):
            if not len(indices):continue
            if cfg.training_prediction_budget is not None:
                remaining=cfg.training_prediction_budget-training_predictions
                if remaining <= 0:break
                indices=indices[:remaining]
            optimizer.zero_grad(set_to_none=True)
            observation = take(inputs, indices); labels = take(fixed_labels, indices)
            visible = (labels['contact_cell_valid'] | ~target['planned_contact'][indices]).all(-1)
            teacher = teacher_mask_for_batch(probability,target['planned_contact'][indices],labels['contact_cell_valid'])
            if cfg.execution_only and not bool(visible.all()):
                raise ValueError('adaptation requires fully observed teacher plans')
            if cfg.constraint_training:
                context = hashlib.sha256(cfg.manifest.read_bytes() + encode_robot_asset_json().encode() + b'predictor_constraint_band_v1').hexdigest()
                sample_ids = [str(int(i)) for i in indices]
                if not bool(visible.all()): raise ValueError('Constraint pilot needs observable contact plans; unsupported hidden plans must be selected explicitly')
                fixed_active = target['planned_contact'][indices]
                fixed_surface = target['planned_surface'][indices]
                _, current_actual = query(observation['current_q'], indices)
                keep = dict(q=to_local(observation['current_q'], indices).detach(),
                    anchor=points_local(current_actual['contact_position_w'], indices).detach(),
                    mask=current_actual['contact_part_mask'] & fixed_active & (labels['role'] == 2)
                         & (current_actual['contact_surface'] == fixed_surface))
                query_audit_index = 0
                def constraint_closure():
                    nonlocal query_audit_index
                    pred = model(**observation, teacher_contact=fixed_active,
                        teacher_cell=labels['contact_cell'], teacher_role=labels['role'],
                        teacher_mask=torch.ones_like(teacher))
                    _, metric, queried = objective(pred, observation, indices, labels)
                    if cfg.pilot_samples is not None:
                        audit_dir = cfg.output/'constraint_queries'
                        audit_dir.mkdir(exist_ok=True)
                        def cpu(value):
                            if torch.is_tensor(value): return value.detach().cpu()
                            if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
                            return value
                        torch.save(dict(schema='predictor_constraint_query_v1', sample_ids=sample_ids,
                            context_id=context, robot_asset_json=encode_robot_asset_json(),
                            sampling='fresh static query at candidate pose; no integration',
                            q_world=pred.qpos.detach().cpu(), observed=cpu(queried[1]),
                            current_observed=cpu(current_actual),
                            forward_inputs=cpu(dict(**observation, teacher_contact=fixed_active,
                                teacher_cell=labels['contact_cell'], teacher_role=labels['role'],
                                teacher_mask=torch.ones_like(teacher))), keep=cpu(keep)), audit_dir/f'{step}_{batches}_{query_audit_index}.pt')
                        query_audit_index += 1
                    return constraint_adapter(model, queried[0], take(scene, indices),
                        active=fixed_active, surface=fixed_surface, sample_ids=sample_ids,
                        context_ids=[context]*len(indices), prior=metric['constraint_prior'].sum(),
                        nominal_q=target['q'][indices], payload=(pred, metric), keep=keep)
                batch_result, step_report = constrained_step.step(constraint_closure)
                with (cfg.output/'constraint_steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(epoch=step, sample_ids=sample_ids, **step_report))+'\n')
                measured = step_report.get('loss_after', step_report['loss_before'])
                constraint_accepted_updates += int(step_report['accepted'])
                gradient_norm_total += step_report['gradient_norm']
                total_count += len(indices); training_predictions += len(indices); batches += 1
                total_loss += measured*len(indices); combined_total += measured*len(indices)
                teacher_count += len(indices)
                continue
            with timed('forward'):
                prediction = model(**observation, teacher_contact=target['planned_contact'][indices],
                    teacher_cell=labels['contact_cell'], teacher_role=labels['role'], teacher_mask=teacher)
            loss, first_metrics, first_queried = objective(prediction, observation, indices, labels)
            record_training_diagnostics(first_metrics)
            training_predictions+=len(indices)
            combined = loss.mean()
            rollout_weighted = combined.detach() * 0
            teacher_count += statistic(teacher.sum(), integer=True); total_count += len(indices); total_loss += statistic(loss.detach().sum())
            count = (max(1, round(len(indices) * cfg.rollout_batch_fraction))
                if cfg.rollout_batch_fraction and not reference_warmup else 0)
            positions = torch.randperm(len(indices), device=device)[:count]
            rindices, rinput = indices[positions], take(observation, positions)
            rsupervised = torch.ones(len(rindices), device=device, dtype=torch.bool)
            for depth in range(rollout_depth if count else 0):
                if cfg.training_prediction_budget is not None:
                    remaining=cfg.training_prediction_budget-training_predictions
                    if remaining <= 0:break
                    rindices=rindices[:remaining];rinput={k:v[:remaining] for k,v in rinput.items()};rsupervised=rsupervised[:remaining]
                with timed('forward'):
                    action = model(**rinput)
                training_predictions+=len(rindices)
                rlabels = labels_for(rinput, rindices, recursive=True)
                rlabels['supervised'] = rsupervised
                rloss, rm, queried = objective(action, rinput, rindices, rlabels)
                term = cfg.rollout_batch_fraction * rloss.mean() / rollout_depth
                combined = combined + term
                rollout_weighted = rollout_weighted + term.detach()
                rollout_outputs += len(rindices)
                depth_outputs[depth] += len(rindices)
                if cfg.gpu_pipeline:
                    from contact_solver.device_contact_objective import acceptance
                    accepted = acceptance(queried[1], target['planned_contact'][rindices],
                        target['planned_surface'][rindices])['contact_accepted']
                else:
                    accepted = []
                    for i, catalog in enumerate(queried[1]['surface_catalog_by_sample']):
                        actual = select_contact_pairs([queried[1]['pairs'][i]], catalog)
                        fact = contact_acceptance(actual['contact_pairs'][0], target['planned_contact'][rindices[i]].tolist(), target['planned_surface'][rindices[i]].tolist())
                        accepted.append(fact['contact_accepted'])
                    accepted = torch.as_tensor(accepted, device=device)
                safe = ((rm['newton_generation_accepted'] > .5) & accepted
                        & (rm['joint_violation_rad'] <= .15))
                if cfg.gpu_pipeline:
                    safe &= rm['event_endpoint_accepted'] > .5
                if cfg.region_plan and cfg.unified_contact:
                    safe &= rm['region_plan_realized'] > .5
                failure_counts['own_contact'] += statistic((rm['newton_contact_accepted'] <= .5).sum(), integer=True)
                failure_counts['own_safety'] += statistic((rm['newton_generation_accepted'] <= .5).sum(), integer=True)
                failure_counts['task_contact'] += statistic(((~accepted) & rsupervised).sum(), integer=True)
                failure_counts['joint_limit'] += statistic((rm['joint_violation_rad'] > .15).sum(), integer=True)
                keep, reset_reasons = continuation(action.qpos, rindices, queried, rm)
                for reason, mask in reset_reasons.items():
                    failure_counts[reason+'_resets'] += statistic(mask.sum(), integer=True)
                next_indices, next_supervised, advance, retry, exhausted = rollout_transition(
                    rindices, successor, rsupervised, safe, keep)
                chosen = keep.nonzero(as_tuple=False).flatten()
                phase_advances += statistic(advance.sum(), integer=True)
                failure_counts['stopped'] += statistic((~keep).sum(), integer=True)
                if not len(chosen) or depth + 1 == rollout_depth: break
                failure_counts['retries_used'] += statistic(retry.sum(), integer=True)
                failure_counts['advances_used'] += statistic(advance.sum(), integer=True)
                rsupervised = next_supervised[chosen]
                rindices = next_indices[chosen]
                rinput = fresh_observation(action.qpos[chosen], rindices, queried[1], chosen)
            combined_total += statistic(combined.detach()) * len(indices)
            rollout_weighted_total += statistic(rollout_weighted) * len(indices)
            with timed('backward_optimizer'):
                combined.backward()
                if cfg.performance_audit and step == 1 and batches == 0:
                    torch.save({'indices': indices.cpu(), 'qpos': prediction.qpos.detach().cpu(),
                        'planned_contact': prediction.conditioned_contact.detach().cpu(),
                        'actual_contact': first_queried[1]['contact_part_mask'].cpu(),
                        'actual_surface': first_queried[1]['contact_surface'].cpu(),
                        'base_loss': loss.detach().cpu(), 'optimized_loss': combined.detach().cpu(),
                        'depth_outputs': depth_outputs.copy(),
                        'gradient': {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None}},
                        cfg.output / 'first_batch_audit.pt')
                gradient_norm = torch.stack(list(model.clip_training_gradients().values())).norm()
                gradient_norm_total += statistic(gradient_norm); batches += 1
                optimizer.step()
        if not cfg.constant_learning_rate:
            scheduler.step()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - step_begun
        # One host read per epoch instead of scalar reads at every rollout.
        if cfg.defer_statistics:
            values = [total_loss, teacher_count, phase_advances, combined_total,
                      rollout_weighted_total, gradient_norm_total, *failure_counts.values()]
            values = torch.stack([torch.as_tensor(v, device=device, dtype=torch.float64) for v in values]).cpu().tolist()
            total_loss, teacher_count, phase_advances, combined_total, rollout_weighted_total, gradient_norm_total = values[:6]
            teacher_count, phase_advances = int(teacher_count), int(phase_advances)
            failure_counts = {key: int(value) for key, value in zip(failure_counts, values[6:])}
        train_timings = dict(timings) if cfg.profile_components else None
        actual_predictions=training_predictions-prediction_start
        parallel_metrics=({key:int(value) for key,value in zip(pool_counts,torch.stack(list(pool_counts.values())).cpu().tolist())}
            if pool is not None else None)
        diagnostics = {name: float(value.cpu())/total_count for name, value in training_diagnostics.items()}
        training_phase = 'reference_warmup' if reference_warmup else ('persistent_parallel' if pool is not None else 'nested_rollout')
        performance_history.append({'step': step, 'training_seconds': train_seconds,
            'training_mode':training_phase, 'training_diagnostics': diagnostics,
            'training_predictions':actual_predictions,'total_training_predictions':training_predictions,
            'predictions_per_second':actual_predictions/train_seconds,'optimizer_updates':constraint_accepted_updates if cfg.constraint_training else batches,
                   'optimizer_proposals':batches,
            'parallel_pool':parallel_metrics,
            'optimized_loss': combined_total/total_count, 'base_loss': total_loss/total_count,
            'rollout_depth_outputs': depth_outputs, 'rollout_failure_counts': failure_counts,
            'training_component_seconds': train_timings})
        (cfg.output / 'performance.json').write_text(json.dumps(performance_history, indent=2))
        budget_finished=cfg.training_prediction_budget is not None and training_predictions >= cfg.training_prediction_budget
        if step == 1 or step % cfg.evaluation_every == 0 or step == cfg.steps or budget_finished:
            metrics = evaluate()
            teacher_metrics = evaluate(teacher=True)
            score = (20 * (1 - metrics['plan_exact']) + 40 * (1 - metrics['newton_generation_valid'])
                + 5 * metrics['newton_unrealized_intended_contacts'] + metrics['root_cm'] + metrics['body_cm']
                + 10 * metrics['joint_rmse_rad'] + metrics['penetration_cm'] + metrics['maxima']['penetration_cm'])
            row = {'step': step, 'teacher_probability': probability, 'teacher_fraction': teacher_count / total_count,
                   'training_phase': training_phase, 'teacher_schedule_step': max(0, step-cfg.reference_warmup_steps),
                   'training_diagnostics': diagnostics,
                   'training_predictions':actual_predictions,'total_training_predictions':training_predictions,
                   'predictions_per_second':actual_predictions/train_seconds,'optimizer_updates':constraint_accepted_updates if cfg.constraint_training else batches,
                   'optimizer_proposals':batches,
                   'parallel_pool':parallel_metrics,
                   'loss': total_loss / total_count, 'historical_validation_score': score,
                   'rollout_outputs': rollout_outputs, 'rollout_phase_advances': phase_advances,
                   'optimized_loss': combined_total/total_count,
                   'weighted_rollout_loss': rollout_weighted_total/total_count if pool is None else None,
                   'mean_gradient_norm_before_clip': gradient_norm_total/max(batches, 1),
                   'rollout_depth_outputs': depth_outputs, 'rollout_failure_counts': failure_counts,
                   'learning_rate': optimizer.param_groups[0]['lr'],
                   'training_seconds': train_seconds, 'training_component_seconds': train_timings,
                   'teacher_metrics': teacher_metrics,
                   'metrics': metrics, 'elapsed_seconds': time.perf_counter() - begun}
            history.append(row); print(json.dumps(row), flush=True)
            save(cfg.output / f'step_{step:04d}.pt', step, metrics)
            save(cfg.output / 'last.pt', step, metrics)
            if score < best_score:
                best_score, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
                save(cfg.output / 'best_historical_score.pt', step, metrics, best_state)
            (cfg.output / 'progress.json').write_text(json.dumps({'history': history}, indent=2))
            if pool is not None:
                torch.save({'schema':'persistent_parallel_rollout_training_state_v1','step':step,
                    'training_predictions':training_predictions,'pool':pool.state_dict(),
                    'pending_plan':pending.state_dict() if pending is not None else None,
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},
                    cfg.output/'parallel_training_state.pt')
        if budget_finished:break
    (cfg.output / 'report.json').write_text(json.dumps({'best_historical_score_step': best_step,
        'best_historical_score': best_score, 'history': history,
        'final_selection': 'run independent full-chain evaluator over milestones; historical score is diagnostic'}, indent=2))
    print(json.dumps({'training_complete': True, 'steps': step,'training_predictions':training_predictions}), flush=True)


if __name__ == '__main__':
    main()
