#!/usr/bin/env python3
"""Train the predictor on complete, self-observed demonstration actions."""
import argparse
import atexit
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
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
from somaforge_core.launcher import parse_isaaclab_launcher_args

ROOT = Path(__file__).resolve().parents[3]
from generator.predictor_architecture import build_position_predictor, load_position_predictor, ARCHITECTURE_FIELDS, LEGACY_SCHEMA
from generator.structured_position_predictor import STRUCTURED_SCHEMA
from generator.conditioned_pose_predictor import ConditionedPosePrediction, conditioned_pose_objective
from generator.contact_location_predictor import contact_points_to_observed_heightmap_map
from generator.planned_contact_predictor import (audit_intended_surfaces, contact_plan_objective,
    plan_points_in_pose_frame)
from contact_solver.native_contact_position import native_position_statistics, CONTACT_POSITION_SCHEMA
from contact_solver.contact_point_sets import (CONTACT_SET_SCHEMA as CONTACT_POINT_SCHEMA,
    contact_patch_observation, matched_targets, matched_endpoint_statistics,
    reference_keep_compatible, matched_endpoint_objective_terms)
from contact_solver.newton_witness_loss import query_local_distances
from contact_solver.contact_layout import (relative_plan_objective, observed_endpoint_targets,
    persistent_role_consistent,
    ENDPOINT_POSITION_TOLERANCE_M, endpoint_position_loss)
from generator.next_interaction_heightmap_v2 import _quaternion_multiply_wxyz, render_root_yaw_box_heightmaps
from contact_solver.interaction_acceptance import contact_acceptance, DEFAULT_ACCEPTANCE
from contact_solver.interaction_acceptance import configured_acceptance, acceptance_contract
from contact_solver.endpoint_penetration import (penetration_contract,
    penetration_checks, solid_generation_mask)
from contact_solver.endpoint_task import PENETRATION_ACCEPTANCE_M, PENETRATION_LOSS_SCALE_M, JOINT_EXCESS_RAD
from generator.endpoint_objective import (OBJECTIVE_SCHEMA, LOSS_GROUPS, DIAGNOSTIC_LOSSES,
    STRUCTURE_WEIGHT, compose_endpoint_loss)
from somaforge_core.contact_face_selection import select_contact_pairs
from somaforge_core.robot_assets import encode_robot_asset_json, decode_robot_asset_json
from somaforge_core.newton_contact_data import require_current_newton_manifest
from somaforge_core.published_data import load_published_json, relocate_published_values
from generator.full1000_training_variants import (event_consistent_roles,
    teacher_mask_for_batch, training_course, endpoint_layout_pass, endpoint_feedback_checks, endpoint_rejection_masks,
    reference_batch_teacher_probability, endpoint_validation_score)
from generator.rollout_termination import TerminationLimits, native_continuation_masks, demonstration_loss
from generator.parallel_rollout_pool import rollout_transition
from generator.optimizer_ownership import OptimizerOwnership, prepare_optimizer_gradients
from generator.training_data import prepare, take
from generator.predictor_initialization import initialize_predictor, initialize_training_joint_bias, state_fingerprint
from generator.scene_workers import Workers
from generator.role_validity import validity_inputs, ROLE_VALIDITY_SCHEMA
from contact_solver.keep_patch_loss import (bind_keep_patch, keep_patch_motion,
    keep_patch_loss, calibrated_keep_allowance, keep_allowance_for_observation)
from contact_solver.keep_patch_cache import KeepPatchCache
from somaforge_core.loaded_material_motion import DEFAULT_MATERIAL_RESIDUAL_SCALE_M
import somaforge_core.newton_scene_router as router


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


@dataclass
class Config:
    query_checkpoint: Path = ROOT / 'runtime/current/holosoma/logs/WholeBodyTracking/20260727_085520-g1_29dof_wbt_single_climb00_completionema_horizon50_ncon160_from4k_to10k-locomotion/model_06000.pt'
    query_source_manifest: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/versions/loaded_material_rebuild_20261003/edits207_joint_v12/scenes/0e21fa0c89cb673d/manifest.json'
    query_source_model: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/versions/loaded_material_rebuild_20261003/edits207_joint_v12/scenes/0e21fa0c89cb673d/model.json'
    manifest: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_main_20261022/manifest.json'
    data_cache: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_main_20261022/data.pt'
    q_cache: Path = ROOT / 'runtime/current/motions/climb00_contact_dataset/training/predictor_main_20261022/q_cache_64.npz'
    output: Path = ROOT / 'runtime/current/models/generator/v1_shared_interval_new_run'
    initialization: Literal['scratch'] = 'scratch'
    architecture: Literal['full1000_position_predictor_v1', 'full1000_position_predictor_v2'] = LEGACY_SCHEMA
    scratch_joint_mean_bias: bool = True
    reference_warmup_steps: int = 100
    continuous_demonstration_training: bool = True
    mode: str = 'full'
    overfit_motion_id: int | None = None
    training_stage: Literal["single_segment", "demonstration_rollout", "post_demonstration", "legacy_mixed"] = "single_segment"
    steps: int = 1100  # Absolute ending step; resume only runs the remaining interval.
    schedule_steps: int = 1100
    resume_training_state: Path | None = None
    resume_gradient_isolation: bool = False  # Explicit shared-to-isolated migration; reset mixed shared moments.
    stop_after_step: int | None = None
    capture_training_state_steps: tuple[int, ...] = ()
    isolate_unsupervised_execution_gradients: bool = False
    optimizer_gradient_ownership: OptimizerOwnership = 'planner_supervised_v1'
    audit_checkpoint: Path | None = None  # Read-only gradient audit; never training initialization.
    query_worlds: int | None = 16
    gpu_pipeline: bool = True
    execution_only: bool = False
    event_roles: bool = True
    limb_contact_modes: bool = False
    role_validity: bool = True
    world_surface_scan: bool = True
    depth_observation: bool = True
    depth_scan_profile: str = 'root_depth_87x58_fov89p5_pitch85_yaw_v1'
    event_endpoint_gate: bool = False
    pending_plan_repair: bool = False
    anchored_plan_execution: bool = False
    relative_plan_supervision: bool = False
    location_supervision: Literal['point', 'surface_region'] = 'surface_region'
    location_region_spacing_multiplier: float = 1.0
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
    keep_patch_weight: float = 0.0  # Historical opt-in regularizer; keep does not imply fixed material.
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
    # Applies to unified-contact; independent training modes keep their loss.
    shared_shape_target_interval: bool = True
    recover_failed_states: bool = False
    rollout_steps_final: int | None = None
    rollout_curriculum_switch: int = 10
    recoverable_penetration_m: float = PENETRATION_ACCEPTANCE_M  # Legacy CLI name; severity classification only.
    acceptance_penetration_m: float = PENETRATION_ACCEPTANCE_M
    endpoint_position_tolerance_m: float = ENDPOINT_POSITION_TOLERANCE_M
    joint_excess_rad: float = JOINT_EXCESS_RAD
    penetration_penalty: Literal['log1p', 'huber'] = 'huber'
    penetration_scale_m: float = PENETRATION_LOSS_SCALE_M
    parallel_rollouts: bool = True
    parallel_pool_size: int | None = 128
    parallel_updates_per_step: int | None = 10
    parallel_episode_limit: int = 64
    parallel_retry_limit: int = 8
    parallel_allow_post_demo: bool = True
    training_prediction_budget: int | None = None
    execution_plan_gradients: bool = False
    execution_observation_gradients: bool = False  # Planning owns its representation; legacy control opts in.
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
    official = parse_isaaclab_launcher_args(__doc__)
    cfg = tyro.cli(Config)
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, Path):
            setattr(cfg, field.name, Path(relocate_published_values(str(value))))
    if cfg.training_stage != "legacy_mixed":
        if not cfg.parallel_rollouts or cfg.pending_plan_repair or cfg.constraint_training or cfg.rollout_batch_fraction:
            raise ValueError("Separate stages require parallel infrastructure without mixed nested rollouts/repair/constraint training")
        if cfg.training_stage != "single_segment" and cfg.resume_training_state is None:
            raise ValueError("Recursive stages require the preceding stage full training state")
        if cfg.pilot_samples is not None and cfg.training_stage != "single_segment":
            raise ValueError("Recursive stages require complete demonstration chains, not partial pilot rows")
    runtime_acceptance = configured_acceptance(asdict(cfg))
    penetration_metadata = penetration_contract(runtime_acceptance,
        penalty=cfg.penetration_penalty, scale_m=cfg.penetration_scale_m)
    if cfg.resume_training_state is not None and (not cfg.parallel_rollouts or cfg.keep_patch_weight or cfg.constraint_training):
        raise ValueError('Full-state resume requires parallel rollouts without keep cache or constrained optimizer')
    if cfg.resume_gradient_isolation and (cfg.resume_training_state is None
            or cfg.architecture != LEGACY_SCHEMA or cfg.execution_observation_gradients
            or cfg.isolate_unsupervised_execution_gradients
            or cfg.optimizer_gradient_ownership != 'planner_supervised_v1'):
        raise ValueError('Gradient-isolation migration requires a full v1 resume with isolated execution and planner-owned optimizer updates')
    if cfg.stop_after_step is not None and not 1 <= cfg.stop_after_step <= cfg.steps:
        raise ValueError('Stop step must be within the configured training schedule')
    if cfg.isolate_unsupervised_execution_gradients and (cfg.architecture != LEGACY_SCHEMA or not cfg.execution_observation_gradients):
        raise ValueError('Selective isolation requires v1 with shared observation gradients enabled')
    if not cfg.gpu_pipeline:
        raise ValueError('Endpoint training requires complete solids from the initialized GPU Newton scene')
    if not cfg.unified_contact:
        raise ValueError('New endpoint training requires unified contact and complete-solid penetration')
    if cfg.world_surface_scan and not cfg.gpu_pipeline:
        raise ValueError('World mesh scans require current initialized Newton scene metadata')
    if not cfg.static_gradient_audit and not cfg.role_validity:
        raise ValueError('New Predictor training requires actual-contact role validity')
    if cfg.reference_warmup_steps < 0:
        raise ValueError('Reference warmup steps must be nonnegative')
    if cfg.role_validity and not (cfg.event_roles and cfg.gpu_pipeline):
        raise ValueError('Surface role validity requires event roles and actual GPU Newton observations')
    if cfg.execution_only:
        raise ValueError('Scratch training requires a trainable planner and executor')
    if not cfg.static_gradient_audit and cfg.architecture == STRUCTURED_SCHEMA and (cfg.execution_plan_gradients or cfg.execution_observation_gradients):
        raise ValueError('Structured scratch training forbids shared planner/executor gradients')
    if cfg.training_stage != 'legacy_mixed' and cfg.architecture != LEGACY_SCHEMA:
        raise ValueError('Separate stages currently support the maintained v1 architecture')
    if not np.isfinite(cfg.location_region_spacing_multiplier) or cfg.location_region_spacing_multiplier <= 0:
        raise ValueError('Location region spacing multiplier must be finite positive')
    if cfg.location_supervision == 'surface_region' and not cfg.world_surface_scan:
        raise ValueError('Surface-region location supervision requires a world scan')
    if cfg.audit_checkpoint is not None and not cfg.static_gradient_audit:
        raise ValueError('An audit checkpoint cannot initialize or resume training')
    if cfg.reference_warmup_steps and cfg.constraint_training:
        raise ValueError('Reference warmup is unsupported with constrained optimizer training')
    if cfg.recover_failed_states or cfg.pending_plan_repair:
        raise ValueError('Legacy recovery/plan-repair flags are unsupported; physically continuable raw-state feedback is now automatic.')
    if cfg.anchored_plan_execution and not (cfg.gpu_pipeline and cfg.parallel_rollouts):
        raise ValueError('Anchored endpoint execution requires parallel GPU training')
    if (official.visualizer and not official.headless_explicit) or official.enable_cameras:
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
    if cfg.keep_patch_weight < 0 or (cfg.keep_patch_weight and not cfg.gpu_pipeline):
        raise ValueError('Keep-patch loss requires nonnegative weight and actual GPU Newton witnesses')
    if cfg.depth_observation and (not cfg.gpu_pipeline or cfg.keep_patch_weight):
        raise ValueError('Depth Predictor uses fresh endpoint contact points; GPU queries required and keep-patch weight must be zero')
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
    device = torch.device(official.device)
    manifest = load_published_json(cfg.manifest)
    if cfg.pending_plan_repair and not (cfg.parallel_rollouts and cfg.gpu_pipeline and cfg.event_roles and cfg.region_plan):
        raise ValueError('Pending-plan repair requires parallel GPU regional event-role model')
    if cfg.pending_plan_repair and cfg.relative_plan_supervision:
        raise ValueError('Pending-plan trial requires body-relative cell supervision: --no-relative-plan-supervision')
    if cfg.event_endpoint_gate and not (cfg.event_roles and cfg.gpu_pipeline and cfg.parallel_rollouts):
        raise ValueError('Event endpoint trial requires explicit roles and parallel GPU training')
    require_current_newton_manifest(manifest, context='full1000 position training')
    if cfg.static_gradient_audit:
        checkpoint = torch.load(cfg.audit_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get('dataset', {}).get('penetration_contract') != penetration_metadata:
            raise ValueError('Audit penetration contract differs; use the recorded options or archived source')
        if checkpoint.get('dataset', {}).get('objective_contract', {}).get('schema') != OBJECTIVE_SCHEMA:
            raise ValueError('Audit objective differs; historical checkpoints require their archived source')
        if checkpoint.get('dataset', {}).get('task_acceptance') != acceptance_contract(runtime_acceptance):
            raise ValueError('Audit endpoint tolerances differ from checkpoint')
        if cfg.role_validity != checkpoint['config'].get('role_validity', False):
            raise ValueError('Audit role validity differs from checkpoint; set the recorded contract explicitly')
        if cfg.depth_observation != checkpoint['config'].get('depth_observation', False):
            raise ValueError('Audit depth observation differs from checkpoint; select its recorded observation')
        from somaforge_core.world_mesh_scan import LEGACY_DEPTH_PROFILE
        if cfg.depth_observation and cfg.depth_scan_profile != checkpoint['config'].get('depth_scan_profile', LEGACY_DEPTH_PROFILE):
            raise ValueError('Audit depth camera differs from checkpoint; select its recorded profile')
        if cfg.depth_observation and checkpoint.get('dataset', {}).get('contact_position_contract') != CONTACT_POINT_SCHEMA:
            raise ValueError('Old depth contact objective requires its archived source for a faithful gradient audit')
        for key in ARCHITECTURE_FIELDS:
            if getattr(cfg, key) != checkpoint['config'].get(key, False if key not in ('width', 'layers', 'location_width') else getattr(cfg, key)):
                raise ValueError(f'Audit configuration differs from checkpoint: {key}')
        model = load_position_predictor(checkpoint, device=device)
    else:
        model = build_position_predictor(asdict(cfg), schema=cfg.architecture).to(device)
    initial, loaded, fresh, warm_metadata = initialize_predictor(model,
        initialization=cfg.initialization, seed=cfg.seed)
    if cfg.static_gradient_audit:
        warm_metadata.update(initialization='read_only_audit', source=str(cfg.audit_checkpoint.resolve()),
            source_step=checkpoint['step'], predictor_checkpoints_loaded=[str(cfg.audit_checkpoint.resolve())])
    if cfg.execution_only and cfg.rollout_batch_fraction:
        raise ValueError('execution-only adaptation requires rollout-batch-fraction 0')
    payload = torch.load(cfg.data_cache, map_location='cpu', weights_only=False)
    if cfg.limb_contact_modes:
        from generator.limb_contact import limb_targets
        # Audit actual cached observations without altering their contact truth.
        for key in ('start_contacts', 'end_contact'):
            limb_targets(torch.as_tensor(payload['data'][key], dtype=torch.long))
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
                      tensor_transport=cfg.gpu_pipeline, solid_geometry=True,
                      depth_scan_profile=cfg.depth_scan_profile).__enter__()
    atexit.register(workers.__exit__, None, None, None)
    router.configured_router = lambda: workers.router
    tensor_scene_ids = torch.full((len(samples),), -1, dtype=torch.long, device=device)
    for i in ids:
        height = min(heights, key=lambda h: abs(h - float(samples[i]['height'])))
        target['newton_model_fingerprint'][i] = torch.tensor(list(bytes.fromhex(workers.entries[height]['fp'])), dtype=torch.uint8, device=device)
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
            rows.solid = workers.solid_router(model.fk, qlocal, tensor_scene_ids[indices],
                world_frame=(origin[indices], basis[indices]))
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
    contact_skin = None
    if cfg.unified_contact:
        from contact_solver.contact_material_skin import ContactMaterialSkin
        contact_skin = ContactMaterialSkin(model.region_geometry, workers.tensor_router.link_names)
        training_rows = torch.zeros(len(samples), device=device, dtype=torch.bool)
        training_rows[train] = True
    if cfg.region_plan:
        region_labels = torch.zeros(len(samples), 6, 4, dtype=torch.bool, device=device)
        audited_pairs = []
        with torch.no_grad():
            for indices in torch.tensor(ids, device=device).split(cfg.batch_size):
                rows, observed = query(native_target_q[indices], indices)
                region_labels[indices] = model.region_geometry.actual_mask(model.fk, rows, target['planned_surface'][indices])
                audited_pairs.append({'indices': indices.cpu(), 'observed': cpu_tree(observed)})
                if contact_skin is not None:
                    contact_skin.add(model.fk, model.region_geometry, native_target_q[indices],
                        observed, indices, training_rows[indices])
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

    if contact_skin is not None:
        skin_source_evidence = []
        if not cfg.region_plan:
            with torch.no_grad():
                for indices in train.split(cfg.batch_size):
                    _, observed = query(native_target_q[indices], indices)
                    contact_skin.add(model.fk, model.region_geometry, native_target_q[indices],
                        observed, indices, training_rows[indices])
                    skin_source_evidence.append(dict(indices=indices.cpu(), q_world=native_target_q[indices].cpu(),
                        pairs={name: value.cpu() for name, value in observed['pairs'].items()}))
        contact_skin.finalize()
        model.region_geometry.material_skin = contact_skin
        from somaforge_core.newton_contacts import SCHEMA as native_contact_schema
        from somaforge_core.contact_face_selection import FACE_POLICY
        skin_metadata = dict(**contact_skin.contract(), contact_semantics=native_contact_schema,
            face_selection=FACE_POLICY, sampling='fresh static Newton query at native source q; no integration',
            raw_evidence_reference='region_labels.pt:newton_batches' if cfg.region_plan else 'source_evidence',
            scenes={str(entry['scene_id']):dict(height_scale=height, model_fingerprint=entry['fp'],
                model=str(entry['folder']/'model.json'), binding=str(entry['folder']/'binding.json'))
                for height, entry in workers.entries.items()})
        torch.save(dict(metadata=skin_metadata, robot_asset_json=encode_robot_asset_json(),
            source_evidence=skin_source_evidence,
            buffers={name: value.cpu() for name, value in contact_skin.named_buffers()}),
            cfg.output/'contact_material_skin.pt')
        print(json.dumps({'contact_material_skin': contact_skin.contract()}), flush=True)

    keep_patch_allowance = None
    keep_patch_metadata = {'enabled': False}
    reference_patch_cache = (KeepPatchCache(len(samples), inputs['current_q'], workers.tensor_router.link_names)
        if cfg.keep_patch_weight else None)
    pool_patch_cache = (KeepPatchCache(cfg.parallel_pool_size, inputs['current_q'], workers.tensor_router.link_names)
        if cfg.keep_patch_weight and cfg.parallel_rollouts else None)
    if cfg.keep_patch_weight:
        # A keep label is continuous activation, not a claim that a particular
        # material site remains stationary. Calibrate geometric motion from
        # each real demonstration action, never a projector-generated target.
        reference_motion = torch.zeros(len(samples), 6, device=device)
        reference_present = torch.zeros_like(reference_motion, dtype=torch.bool)
        calibration_ids = torch.cat((train, validation)).unique()
        with torch.no_grad():
            for indices in calibration_ids.split(cfg.batch_size):
                observed = workers.tensor_router(native_q[indices], tensor_scene_ids[indices])
                patch = bind_keep_patch(model.fk, native_q[indices], observed)
                movement, present = keep_patch_motion(model.fk, native_target_q[indices], patch)
                reference_motion[indices], reference_present[indices] = movement, present
        keep = target['role'] == 2
        if bool((keep[calibration_ids] & ~reference_present[calibration_ids]).any()):
            raise ValueError('Demonstration keep role lacks an actual allocated primary-face patch')
        # A predicted keep on a part whose demonstrated role is different has
        # no per-action keep example. Use only training keep examples as its
        # fallback allowance, and explicitly mark unseen parts as unknown.
        keep_patch_allowance, fallback, fallback_known = calibrated_keep_allowance(
            reference_motion, reference_present, target['role'], train)
        keep_patch_metadata = dict(enabled=True, schema='newton_material_patch_keep_intent_v1',
            weight=cfg.keep_patch_weight, role='network conditioned_role == keep',
            geometry='current actual allocated task-eligible primary-face materials; FK to endpoint',
            patch='minimum tangent displacement over convex material patch; permits an interior pivot',
            allowance='own demonstration action for demonstrated keep; training-only per-part median otherwise',
            fallback_m=fallback.cpu().tolist(), fallback_known=fallback_known.cpu().tolist(),
            unknown_policy='reported; no invented material or loaded-support evidence',
            penalty='log1p_squared_residual_v1; same scalar tail as contact and collision',
            scale_m=DEFAULT_MATERIAL_RESIDUAL_SCALE_M, actual_loaded_path_budget_unchanged=True,
            contact_truth_unchanged=True, executed_support_evidence=False)
        torch.save(dict(metadata=keep_patch_metadata, robot_asset_json=encode_robot_asset_json(),
            sample_ids=calibration_ids.cpu(), motion_m=reference_motion.cpu(),
            present=reference_present.cpu(), allowance_m=keep_patch_allowance.cpu()),
            cfg.output/'keep_patch_calibration.pt')
        print(json.dumps({'keep_patch':keep_patch_metadata}), flush=True)

    def render(q, indices):
        if model.depth_observation:
            raise RuntimeError('Heightmap rendering is forbidden for the depth Predictor')
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
    if not model.depth_observation:
        inputs['heightmap'] = render(native_q, torch.arange(len(samples), device=device))
    if cfg.role_validity:
        # Query native INPUT poses; future target contacts cannot define legality.
        raw_inputs = cfg.output / 'role_validity_input_queries'
        raw_inputs.mkdir(exist_ok=False)
        torch.save(dict(schema=ROLE_VALIDITY_SCHEMA, robot_asset_json=encode_robot_asset_json(),
            sampling='same-state fixed-q queries before prediction; no integration',
            scenes={str(k): c.metadata for k, c in workers.tensor_clients.items()}), raw_inputs / 'semantics.pt')
        with torch.no_grad():
            for batch_ids in torch.tensor(ids, device=device).split(cfg.batch_size):
                actual = workers.tensor_router(native_q[batch_ids], tensor_scene_ids[batch_ids])
                metadata = validity_inputs(model, native_q[batch_ids],
                    None if model.depth_observation else inputs['heightmap'][batch_ids], actual,
                    scan_provider=workers.scan, scene_ids=tensor_scene_ids[batch_ids])
                inputs['current_contact'][batch_ids] = actual['contact_part_mask']
                inputs['current_anchor'][batch_ids] = actual['contact_position_w']
                for key, value in metadata.items():
                    if key not in inputs:
                        inputs[key] = value.new_zeros((len(samples), *value.shape[1:]))
                    inputs[key][batch_ids] = value
                torch.save(cpu_tree(dict(sample_ids=batch_ids, current_q=native_q[batch_ids],
                    observed=actual, validity=metadata,
                    depth_scan_audit=(workers.mesh_scan_router.last_audit if model.world_surface_scan else None))), raw_inputs / f'{int(batch_ids[0]):06d}.pt')
    anchors_w = points_world(target['anchor'], torch.arange(len(samples), device=device))
    endpoint_patches = {}
    reference_point_audit = None
    if model.depth_observation:
        batches = (audited_pairs if cfg.region_plan else
            ({'indices': batch_ids, 'observed': workers.tensor_router(native_target_q[batch_ids], tensor_scene_ids[batch_ids])}
             for batch_ids in torch.tensor(ids, device=device).split(cfg.batch_size)))
        for batch in batches:
            batch_ids = batch['indices'].to(device)
            patch = contact_patch_observation(batch['observed'], native_q[batch_ids])
            for key, value in patch.items():
                if key not in endpoint_patches:
                    endpoint_patches[key] = value.new_zeros(len(samples), *value.shape[1:])
                endpoint_patches[key][batch_ids] = value
        audit_ids = torch.tensor(ids, device=device)
        audit_observation = take(inputs, audit_ids)
        same_surface = (audit_observation['current_contact_relations'] &
            (audit_observation['contact_surface_ids'][:, None] == target['planned_surface'][audit_ids, :, None])).any(-1)
        audit_roles = event_consistent_roles(target['role'][audit_ids], same_surface)
        _, conflict = reference_keep_compatible(take(endpoint_patches, audit_ids), audit_roles,
            target['planned_surface'][audit_ids], audit_observation, runtime_acceptance.endpoint_position_m)
        reference_point_audit = dict(schema=CONTACT_POINT_SCHEMA,
            actual_keep_parts=int((audit_roles == 2).sum()), conflicting_keep_parts=int(conflict.sum()),
            conflicting_samples=int(conflict.any(-1).sum()),
            sample_ids=audit_ids[conflict.any(-1)].cpu().tolist(),
            policy='keep labels and contact objectives retained; incompatible pose imitation explicitly masked',
            temporal_continuity='inherited demonstrated event labels; not certified by endpoint snapshots')
        (cfg.output/'reference_contact_point_audit.json').write_text(json.dumps(reference_point_audit, indent=2))
        print(json.dumps({'reference_contact_point_audit': {k:v for k,v in reference_point_audit.items() if k != 'sample_ids'}}), flush=True)
    if getattr(model, 'world_surface_scan', False):
        dataset_scan_contract = workers.mesh_scan_router.contract()
        (cfg.output/'world_mesh_scan_contract.json').write_text(json.dumps(dataset_scan_contract, indent=2))
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
        observed_target_contact = observation['current_contact']
        if cfg.role_validity:
            observed_target_contact = (observation['current_contact_relations']
                & (observation['contact_surface_ids'][:, None] == target['planned_surface'][indices, :, None])).any(-1)
        roles = event_consistent_roles(target['role'][indices], observed_target_contact)
        endpoint_points, persistent = observed_endpoint_targets(anchors_w[indices],
            observation['current_contact'], roles)
        if model.depth_observation:
            endpoint_points = matched_targets(endpoint_points, roles, target['planned_surface'][indices],
                observation, take(endpoint_patches, indices))
        if getattr(model, 'world_surface_scan', False):
            from somaforge_core.world_mesh_scan import mesh_location_labels
            cell, valid = mesh_location_labels(endpoint_points, roles != 0,
                target['planned_surface'][indices], observation,
                radius_m=(observation['scan_spacing_m']*cfg.location_region_spacing_multiplier
                    if cfg.location_supervision == 'surface_region' else None))
        else:
            _, cell, valid = contact_points_to_observed_heightmap_map(observation['current_q'],
                roles != 0, endpoint_points, observation['heightmap'])
        if model.depth_observation:
            # Keep supervision is available from contact observation, not camera coverage.
            valid = valid | (roles == 2)
        labels = {'role': roles, 'contact_cell': cell, 'contact_cell_valid': valid,
                'endpoint_points_world': endpoint_points, 'contact_keep_intent': persistent}
        if cfg.location_supervision == 'surface_region':
            from somaforge_core.heightmap import _root_yaw_basis
            basis, _ = _root_yaw_basis(observation['current_q'])
            radius = observation['scan_spacing_m'] * cfg.location_region_spacing_multiplier
            nearest = observation['scan_points_world'].gather(1, cell[..., None].expand(-1, -1, 3))
            covered = (nearest-endpoint_points).norm(dim=-1) <= radius[:, None]
            labels['contact_cell_valid'] = valid & (covered | (roles == 2))
            labels.update(location_region_points_local=torch.einsum('bji,bpj->bpi', basis,
                endpoint_points-observation['current_q'][:, None, :3]),
                location_region_surface=target['planned_surface'][indices], location_region_radius_m=radius)
        return labels

    def teacher_conditions(indices, labels):
        conditions = dict(teacher_contact=target['planned_contact'][indices],
            teacher_cell=labels['contact_cell'], teacher_role=labels['role'])
        if cfg.region_plan:
            conditions['teacher_regions'] = target['region_label'][indices]
        return conditions

    def teacher_available(indices, labels):
        visible = (labels['contact_cell_valid'] | ~target['planned_contact'][indices]).all(-1)
        if cfg.region_plan:
            visible &= (target['region_label_known'][indices] | ~target['planned_contact'][indices]).all(-1)
        return visible

    def objective(prediction, observation, indices, labels, *, patch_cache=None, patch_keys=None):
        qlocal = to_local(prediction.qpos, indices)
        if cfg.reuse_fk:
            model.fk.begin_link_pose_cache(qlocal)
        try:
            return objective_impl(prediction, observation, indices, labels, qlocal,
                patch_cache=patch_cache, patch_keys=patch_keys)
        finally:
            if cfg.reuse_fk:
                model.fk.end_link_pose_cache()

    def objective_impl(prediction, observation, indices, labels, qlocal, *, patch_cache=None, patch_keys=None):
        supervised = labels.get('supervised', torch.ones(len(indices), device=device, dtype=torch.bool))
        with timed('newton_query'):
            queried = query(prediction.qpos, indices, qlocal if cfg.reuse_fk else None)
        if cfg.gpu_pipeline and cfg.region_plan:
            rows = queried[0]
            rows.contact_region_ids = model.region_geometry.witness_regions(model.fk, qlocal,
                rows.sample, rows.part, rows.points[:, 1])
        points_w = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
        surfaces = (prediction.conditioned_surface if prediction.conditioned_surface is not None
            else audit_intended_surfaces(points_w, prediction.conditioned_contact, queried[1]))
        if model.depth_observation:
            points_w = matched_targets(points_w, prediction.conditioned_role, surfaces, observation)
        with timed('execution_objective'):
            _, metrics = conditioned_pose_objective(model, ConditionedPosePrediction(qlocal), take(target, indices),
                take(scene, indices), realization_contact=prediction.conditioned_contact, realization_surface=surfaces,
                missing_pair_approach_weight=cfg.missing_pair_approach_weight, query_override=queried)
        compatible = ((prediction.conditioned_contact == target['planned_contact'][indices]).all(-1)
            & ((surfaces == target['planned_surface'][indices]) | ~prediction.conditioned_contact).all(-1))
        with timed('relative_layout'):
            target_recovery = target_recovery_mask = None
            if model.depth_observation:
                endpoint_stats,target_recovery,target_recovery_mask = matched_endpoint_objective_terms(model, workers.position_router,
                    queried[0], tensor_scene_ids[indices], points_w, prediction.conditioned_role,
                    surfaces, observation, regions=prediction.planned_regions)
                own_error,own_complete,own_count = endpoint_stats
            else:
                own_error,own_complete,own_count=native_position_statistics(model,workers.position_router,
                    qlocal,tensor_scene_ids[indices],points_w,prediction.conditioned_contact,surfaces,queried[1],
                    regions=prediction.planned_regions,world_frame=(origin[indices],basis[indices]))
            layout=endpoint_position_loss(own_error, runtime_acceptance.endpoint_position_m)
            layout_metrics=dict(relative_layout_rms_cm=100*own_error.detach().clamp_min(0).sqrt(),
                relative_layout_observed_parts=own_count.to(qlocal))
        # An endpoint predictor supplies neither an adjacent-pose trajectory nor
        # executed loads. Do not substitute a whole-action displacement loss.
        metrics['support_motion_evidence_known'] = torch.zeros_like(compatible, dtype=qlocal.dtype)
        hold_loss = prediction.qpos.sum(-1)*0
        if cfg.keep_patch_weight:
            with timed('keep_patch_objective'):
                cache = reference_patch_cache if patch_cache is None else patch_cache
                keys = indices if patch_keys is None else patch_keys
                # Legacy nested rollouts may contain different generated poses
                # for the same demonstration row; those have no stable slot key.
                if patch_cache is None and not torch.equal(observation['current_q'], inputs['current_q'][indices]):
                    current_actual = workers.tensor_router(observation['current_q'].detach(), tensor_scene_ids[indices])
                    patch = bind_keep_patch(model.fk, observation['current_q'], current_actual)
                else:
                    patch = cache.get(keys, observation['current_q'], tensor_scene_ids[indices],
                        model.fk, workers.tensor_router,
                        source=reference_patch_cache if patch_cache is not None else None, source_keys=indices)
                keep_allowance = keep_allowance_for_observation(
                    keep_patch_allowance[indices], fallback, fallback_known, supervised)
                hold_loss, hold_metrics = keep_patch_loss(model.fk, prediction.qpos,
                    prediction.conditioned_role, patch, keep_allowance)
                # Scalar-per-sample summaries match the existing training log.
                active = (prediction.conditioned_role == 2) & hold_metrics['keep_patch_present']
                metrics['keep_patch_loss'] = hold_loss
                metrics['keep_patch_missing'] = hold_metrics['keep_patch_missing']
                metrics['keep_patch_motion_cm'] = 100*(hold_metrics['keep_patch_motion_m']*active).amax(-1)
                metrics['keep_patch_excess_cm'] = 100*hold_metrics['keep_patch_excess_m'].amax(-1)
                metrics['keep_patch_unknown_allowance'] = ((prediction.conditioned_role == 2)
                    & ~torch.isfinite(keep_allowance)).sum(-1).to(qlocal)
        with timed('planner_objective'):
            plan_loss, plan_metrics = (relative_plan_objective(prediction, labels, observation.get('heightmap'))
                if cfg.relative_plan_supervision else contact_plan_objective(prediction, labels))
        if cfg.execution_only:
            plan_loss = plan_loss.detach() * 0
        if model.depth_observation:
            keep_compatible, conflict = reference_keep_compatible(take(endpoint_patches, indices),
                prediction.conditioned_role, surfaces, observation, runtime_acceptance.endpoint_position_m)
            metrics['reference_keep_point_conflict_parts'] = conflict.sum(-1).to(qlocal)
            metrics['reference_keep_pose_imitation_masked'] = (compatible & supervised & ~keep_compatible).to(qlocal)
            compatible &= keep_compatible
        imitation = demonstration_loss(compatible * metrics['imitation_loss'], supervised)
        metrics['imitation_supervised'] = (compatible & supervised).float()
        if cfg.unified_contact:
            from contact_solver.contact_regions import unified_region_objective
            with timed('unified_region_objective'):
                interval=(workers.target_interval(model,queried[0],prediction.conditioned_contact,
                    surfaces,take(scene,indices),tensor_scene_ids[indices]) if cfg.shared_shape_target_interval else None)
                execution_loss, region_metrics = unified_region_objective(model, queried[0],
                    prediction.conditioned_contact, surfaces, take(scene, indices), prediction.planned_regions,
                    audit_path=cfg.output/'unknown_normals.jsonl', surface_interval=interval,
                    penetration_penalty=cfg.penetration_penalty, penetration_scale_m=cfg.penetration_scale_m,
                    target_recovery=target_recovery, target_recovery_mask=target_recovery_mask)
            metrics.update(region_metrics)
            metrics['newton_invalid_fullbody_witnesses'] = torch.maximum(metrics['newton_invalid_fullbody_witnesses'], region_metrics['region_invalid_penetrating_normals'])
            metrics['optimized_execution_loss'] = execution_loss
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
        loss, contributions = compose_endpoint_loss(planner=plan_loss, imitation=imitation,
            contact=execution_loss, endpoint=cfg.relative_layout_weight*layout,
            joint_limit=metrics['safety_loss'],
            legacy_keep=cfg.keep_patch_weight*hold_loss if cfg.keep_patch_weight else None)
        metrics.update(contributions)
        with torch.no_grad():
            if model.depth_observation:
                task_error,task_complete,task_count = matched_endpoint_statistics(model, workers.position_router,
                    queried[0], tensor_scene_ids[indices], labels['endpoint_points_world'], labels['role'],
                    target['planned_surface'][indices], observation)
            else:
                task_error,task_complete,task_count=native_position_statistics(model,workers.position_router,
                    qlocal,tensor_scene_ids[indices],labels['endpoint_points_world'],target['planned_contact'][indices],
                    target['planned_surface'][indices],queried[1],world_frame=(origin[indices],basis[indices]))
            own_ok = endpoint_layout_pass(own_error, own_complete, own_count, runtime_acceptance.endpoint_position_m)
            task_ok = endpoint_layout_pass(task_error, task_complete, task_count, runtime_acceptance.endpoint_position_m)
            metrics['own_endpoint_layout_accepted'] = own_ok.float()
            metrics['task_endpoint_layout_accepted'] = torch.where(supervised, task_ok.float(), torch.nan)
            support_ok = persistent_role_consistent(prediction.conditioned_role, observation['current_contact'])
            if cfg.role_validity:
                same_relation = (observation['current_contact_relations'] &
                    (observation['contact_surface_ids'][:, None] == surfaces[..., None])).any(-1)
                illegal = (prediction.conditioned_role == 2) & ~same_relation
                if bool(illegal.any()):
                    raise ValueError('Issued keep violated the actual part/surface action space')
                support_ok &= ~illegal.any(-1)
                metrics['role_validity_invalid_keep'] = illegal.sum(-1).float()
                metrics['role_validity_absent_keep_probability'] = torch.where(
                    observation['current_contact'], 0., prediction.role_logits.softmax(-1)[..., 2]).amax(-1)
            metrics['persistent_role_accepted'] = support_ok.float()
            metrics['event_endpoint_accepted'] = (own_ok & (~supervised | task_ok) & support_ok).float()
            metrics['issued_plan_position_loss'] = layout
            # The scene-fixed layout term above already optimizes this error.
            # Do not add it twice when the legacy anchored flag is set.
        solid_depth = queried[0].solid.depths()
        metrics['solid_penetration_cm'] = 100*solid_depth.amax(-1)
        metrics['solid_terrain_penetration_cm'] = 100*solid_depth[:, 0]
        metrics['solid_self_penetration_cm'] = 100*solid_depth[:, 1]
        metrics['execution_penetration_cm'] = metrics['solid_penetration_cm']
        metrics['penetration_cm'] = metrics['solid_penetration_cm']
        metrics['newton_generation_accepted'] = solid_generation_mask(metrics, limits=runtime_acceptance).float()
        metrics['native_generation_valid_diagnostic'] = metrics.pop('newton_generation_valid')
        metrics['region_plan_safe'] = metrics['region_plan_realized'] * metrics['newton_generation_accepted']
        metrics.update({key: value.float() for key, value in penetration_checks(metrics, limits=runtime_acceptance).items()})
        if cfg.static_gradient_audit:
            # Expose the exact weighted terms only for frozen-model auditing.
            metrics['audit_planner'] = plan_loss
            metrics['audit_imitation'] = imitation
            metrics['audit_layout'] = cfg.relative_layout_weight*layout
            metrics['audit_safety'] = metrics['safety_loss']
            metrics['audit_keep_patch'] = cfg.keep_patch_weight*hold_loss
        metrics.update(plan_metrics); metrics.update(layout_metrics)
        # Surface compatibility is loss-only, so replacing the two-way
        # classifier by thousands of cells cannot silence the motion prior.
        emitted_surfaces = audit_intended_surfaces(plan_points_in_pose_frame(observation['current_q'], prediction.contact_points_local),
                                                  prediction.contact, queried[1])
        metrics['plan_exact'] = ((prediction.contact == target['planned_contact'][indices]).all(-1)
            & ((emitted_surfaces == target['planned_surface'][indices]) | ~prediction.contact).all(-1)).float()
        metrics['constraint_prior'] = (plan_loss + imitation + cfg.relative_layout_weight * layout
            + cfg.keep_patch_weight * hold_loss)
        for name in DIAGNOSTIC_LOSSES:
            if name in metrics:
                metrics['diagnostic_'+name] = metrics.pop(name)
        metrics['loss'] = loss
        return loss, metrics, queried

    @torch.no_grad()
    def realized_layout(prediction, observation, queried, indices):
        points = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
        surfaces = (prediction.conditioned_surface if prediction.conditioned_surface is not None
            else audit_intended_surfaces(points, prediction.conditioned_contact, queried[1]))
        active = prediction.conditioned_contact
        if model.depth_observation:
            points = matched_targets(points, prediction.conditioned_role, surfaces, observation)
            return matched_endpoint_statistics(model, workers.position_router, queried[0],
                tensor_scene_ids[indices], points, prediction.conditioned_role, surfaces, observation,
                regions=prediction.planned_regions)
        return native_position_statistics(model,workers.position_router,to_local(prediction.qpos,indices),
            tensor_scene_ids[indices],points,active,surfaces,queried[1],regions=prediction.planned_regions,
            world_frame=(origin[indices],basis[indices]))

    def fresh_observation(q, indices, observed, chosen):
        if cfg.gpu_pipeline:
            heightmap = None if model.depth_observation else render(q, indices)
            result = {'current_q': q.detach(), 'current_contact': observed['contact_part_mask'][chosen],
                    'current_anchor': observed['contact_position_w'][chosen]}
            if heightmap is not None:
                result['heightmap'] = heightmap
            if cfg.role_validity:
                # Subset the same-state query and remap its pair sample IDs.
                from generator.role_validity import subset_observed
                result.update(validity_inputs(model, q, heightmap, subset_observed(observed, chosen),
                    scan_provider=workers.scan, scene_ids=tensor_scene_ids[indices]))
            return result
        masks, points = [], []
        for i in chosen.tolist():
            selected = select_contact_pairs([observed['pairs'][i]], observed['surface_catalog_by_sample'][i])
            masks.append(selected['contact_part_mask'][0]); points.append(selected['contact_position_w'][0])
        return {'current_q': q.detach(), 'current_contact': torch.as_tensor(np.asarray(masks), device=device),
                'current_anchor': torch.as_tensor(np.asarray(points), device=device), 'heightmap': render(q, indices)}

    @torch.no_grad()
    def continuation(q, indices, queried, metric):
        if cfg.gpu_pipeline:
            actual_contact = queried[1]['contact_part_mask']
        else:
            actual_contact = torch.as_tensor(np.asarray([
                select_contact_pairs([pairs], catalog)['contact_part_mask'][0]
                for pairs, catalog in zip(queried[1]['pairs'], queried[1]['surface_catalog_by_sample'])
            ]), device=device, dtype=torch.bool)
        return native_continuation_masks(q, metric, actual_contact,
            pair=queried[0].pair if cfg.gpu_pipeline else None,
            limits=TerminationLimits(penetration_m=cfg.recoverable_penetration_m,
                joint_excess_rad=runtime_acceptance.joint_excess_rad))

    starts = {(s['motion_id'], s['current_frame']): i for i, s in enumerate(samples)}
    successor = torch.tensor([starts.get((s['motion_id'], s['target_frame']), -1) for s in samples], device=device)
    allowed_successors = torch.zeros(len(samples), device=device, dtype=torch.bool)
    allowed_successors[train] = True
    successor = torch.where((successor >= 0) & allowed_successors[successor.clamp_min(0)], successor, -1)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in asdict(cfg).items()}
    if cfg.static_gradient_audit:
        config['architecture'] = checkpoint['schema']
    dataset = {'schema': 'full1000_position_ablation_v1', 'original_timeline_preserved': bool(manifest.get('fixed_timeline_ablation')),
        'raster_label_semantics': 'nearest_observed_3d_cell_v1',
        'forbidden_network_encoder_inputs': ['surface ID', 'event index', 'future q'],
        'role_validity': ROLE_VALIDITY_SCHEMA if cfg.role_validity else 'legacy_unconstrained',
        'deterministic_validity_inputs': ['actual part/surface relations', 'actual upward surface catalog raster association'] if cfg.role_validity else [],
        'own_plan_realization_under_wrong_predicted_plan': True,
        'pose_realization_gradient_reaches_plan_heads': False,
        'optimizer_gradient_ownership': cfg.optimizer_gradient_ownership,
        'unsupervised_optimizer_updates': ('skip planner parameters and their moments; also skip isolated shared parameters'
            if cfg.optimizer_gradient_ownership == 'planner_supervised_v1' else 'historical zero-gradient AdamW updates'),
        'execution_plan_gradient_contract': ('legacy checkpoint flags, read-only audit' if cfg.static_gradient_audit and checkpoint['schema'] != STRUCTURED_SCHEMA
            else ('detached hard contact intent; shared v1 observation encoder' if cfg.architecture == LEGACY_SCHEMA and cfg.execution_observation_gradients
            else 'detached hard contact intent; independent planner/executor observation gradients')),
        'architecture_contract': model.architecture_contract() if hasattr(model, 'architecture_contract') else {'schema': checkpoint['schema'], 'legacy_read_only': True},
        'gradient_clipping': 'actual model parameter groups clipped independently at norm 10: '+', '.join(model.training_parameter_groups()),
        'failure_states_used_for_training': bool(cfg.parallel_rollouts or cfg.rollout_batch_fraction), 'common_horizontal_translation_allowed_in_layout': False,
        'contact_position_contract': CONTACT_POINT_SCHEMA if model.depth_observation else CONTACT_POSITION_SCHEMA,
        'reference_contact_point_audit': reference_point_audit,
        'rollout_policy': (f'up to {cfg.rollout_steps} autonomous outputs per batch segment; keep physically continuable failures; no demonstration-end termination'
                           if cfg.rollout_batch_fraction else 'no autonomous rollout'),
        'source_reconstruction': 'archived full1000 config, teacher schedule and loss contract; not a byte-identical source restore',
        'runtime_difference': 'current Newton labels rebuilt on original boundaries',
        'position_interface_loss': ({'schema': CONTACT_POINT_SCHEMA,
            'representative': 'minimum XYZ distance over eligible same-part/surface endpoint pairs',
            'keep_target': 'all actual starting points on the same part/surface; no deepest-point identity',
            'other_target': 'planned endpoint point', 'metric': 'XYZ RMS',
            'missing_contact_gradient': 'closest intended region native XYZ target interval replaces generic attraction in unified contact only; other regions retain surface intervals',
            'pose_imitation_conflict': 'mask and count samples whose reference keep point sets exceed configured endpoint tolerance; keep labels unchanged',
            'temporal_continuity_verified_by_endpoint_loss': False}
            if model.depth_observation else workers.position_router.contract()),
        'endpoint_position_loss': {'schema': CONTACT_POINT_SCHEMA if model.depth_observation else CONTACT_POSITION_SCHEMA,
            'tolerance_m': runtime_acceptance.endpoint_position_m,
            'zero_gradient_inside_accepted_range': True,
            'spatial_metric_changed': True,
            'newton_contact_rule_changed': False},
        'task_acceptance': acceptance_contract(runtime_acceptance),
        'objective_contract': dict(schema=OBJECTIVE_SCHEMA,
            weighted_groups=list(LOSS_GROUPS), diagnostic_only_losses=list(DIAGNOSTIC_LOSSES),
            joint_imitation='mean squared joint error in pose only; FK structure has no maximum joint penalty',
            missing_contact='one closest-region XYZ target interval in unified contact; other required regions retain surface intervals; observed endpoint loss has no recovery',
            endpoint_position='actual same-part/surface XYZ contact point sets; keep compares start/end',
            structure_weight=STRUCTURE_WEIGHT, endpoint_weight=cfg.relative_layout_weight,
            legacy_keep_weight=cfg.keep_patch_weight),
        'penetration_contract': penetration_metadata,
        'validation_score_contract': 'complete_solid_endpoint_acceptance_score_v1',
        'optimization_penetration_target_m': 0.,
        'planner_supervision': ('keep: actual starting point on same surface; other active roles: demonstrated endpoint'
            if model.depth_observation else 'actual demonstrated endpoints for every role'),
        'relative_layout_contract': CONTACT_POINT_SCHEMA if model.depth_observation else CONTACT_POSITION_SCHEMA,
        'execution_only_adaptation': cfg.execution_only,
        'initialization': warm_metadata,
        'inherited_pretraining': 'none; predictor training initializes without predictor checkpoints',
        'samples': len(samples), 'training_samples': len(train), 'validation_samples': len(validation)}
    if model.world_surface_scan:
        dataset['world_mesh_scan'] = dataset_scan_contract
        dataset['raster_label_semantics'] = None
        dataset['location_label_semantics'] = ('same_surface_local_region_probability_mass_v1'
            if cfg.location_supervision == 'surface_region' else 'nearest_visible_same_surface_world_xyz_v1')
        dataset['location_supervision'] = dict(mode=cfg.location_supervision,
            radius='scan_spacing_m * location_region_spacing_multiplier',
            spacing_multiplier=cfg.location_region_spacing_multiplier,
            replaces_point_classification=True, keep_excluded=True,
            surface_ids='training grouping only; no learned ID embedding or inference restriction',
            contact_truth_unchanged=True)
        dataset['deterministic_validity_inputs'] = ['actual part/surface relations', 'registered scanned world surface identity']
        dataset['heightmap_usage'] = ('none; 2D optical-depth image features and raw-pixel world candidates'
            if model.depth_observation else 'context only; interaction XYZ and surface identity come from raw depth/tactile mesh')
        if model.depth_observation:
            dataset['environment_observation'] = model.architecture_contract()['environment_observation']
    if cfg.unified_contact and cfg.shared_shape_target_interval:
        dataset['shape_target_interval_objective']=workers.target_interval_provider().contract()
    dataset['reference_warmup'] = {
        'steps': cfg.reference_warmup_steps, 'sampling': 'one shuffled traversal of all training rows per interval',
        'plan': 'visible demonstrated role/contact/cell and actual Newton endpoint regions; region logits remain independently supervised',
        'newton_objective_unchanged': True, 'generated_state_feedback': False,
        'optimizer_and_scheduler_restarted_at_transition': False,
        'teacher_schedule_clock': 'reporting interval minus reference_warmup_steps',
    }
    dataset['continuous_demonstration_training'] = {
        'enabled': cfg.continuous_demonstration_training and cfg.parallel_rollouts,
        'sampling': 'one shuffled traversal of the training demonstration rows each feedback interval',
        'conditioning': 'complete demonstrated role/contact/cell/native regions',
        'loss': 'same planner, pose, contact, layout, keep and safety objective',
        'pool_commit': False, 'autonomous_teacher_schedule_changed': False,
        'post_demo_reference_target_fabricated': False,
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
        'failure_contract': 'commit only accepted and physically continuable predictions; otherwise retain input and target; every candidate contributes loss before rollback; periodically resample after the attempt budget',
        'pending_plan_repair': cfg.pending_plan_repair,
        'repair_contract': 'frozen issued world intent; no planner loss on repairs; own-complete wrong-task plans reset' if cfg.pending_plan_repair else None,
        'event_roles': cfg.event_roles, 'event_endpoint_gate': cfg.event_endpoint_gate,
        'event_gate_scope': 'region-matched scene endpoint and persistent role validity; temporal retouch is unverified',
        'event_layout_tolerance': '4 cm RMS spatial task tolerance; separate from Newton contact truth',
        'consistent_event_roles': cfg.consistent_event_roles, 'body_geometry': cfg.body_geometry,
        'part_geometry': cfg.part_geometry,
        'region_plan': cfg.region_plan, 'unified_contact': cfg.unified_contact,
        'collision_aggregation': cfg.collision_aggregation, 'recovery_same_stage_until_completed': True,
        'severe_penetration_classification_m': cfg.recoverable_penetration_m,
        'rollout_steps_initial': cfg.rollout_steps, 'rollout_steps_final': cfg.rollout_steps_final,
        'bptt': False, 'raw_q_feedback': True,
        'support_motion_objective': {
            'enabled': False, 'reason': 'endpoint_only_without_executed_trajectory',
            'substitute_endpoint_displacement_loss': False},
        'keep_intent_geometry_objective': keep_patch_metadata,
        'execution_gap_objective': {
            'schema': 'newton_contact_upper_complete_solid_lower_v1',
            'upper': 'actual pair includemargin; independent of activation flags',
            'lower_m': 0.,
            'lower_penalty': 'complete-solid penetration and coherent target lower bound; identical material queries deduplicated',
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
            'severe_penetration_classification_m':cfg.recoverable_penetration_m,
            'epoch_boundaries_reset_pool':False,'cross_step_gradient':False,
            'reference_q_only_on_initialization_or_resample':True,
            'failure_feedback':'retain exact pre-prediction input and target within sampling window; rejected output still trains',
            'progress_history':'accepted transitions only','successor_must_be_training_row':True,
            'old_rollout_steps_and_fraction_are_unused':True,
            'plan_conditioning':'teacher plans only while demonstration-supervised; no teacher or future labels after exhaustion',
            'autonomous_commit':'own actual contact, region-filtered position, keep-role consistency and safety; no final demonstration target',
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
        if cfg.unified_contact:
            dataset['collision_distance_objective'] = workers.solid_router.contract()
        from contact_solver.contact_surface_interval import CONTACT_SURFACE_INTERVAL_SCHEMA
        dataset['contact_regions'] = {'schema': model.region_geometry.schema,
            'region_names': model.region_geometry.names, 'features_per_part': 32 if cfg.region_plan else 0,
            'raw_mesh_network_input': False, 'region_label_metadata': region_label_metadata,
            'unified_interval_objective': cfg.unified_contact,
            'interval_penalty_aggregation': ('huber_solid_lower_log1p_contact_mean_plus_max_v1'
                if cfg.penetration_penalty == 'huber' else 'log1p_squared_lower_plus_upper_mean_plus_max_v3'),
            'material_face_interval': CONTACT_SURFACE_INTERVAL_SCHEMA,
            'native_candidate_disables_material_field': False,
            'contact_material_skin': None if contact_skin is None else contact_skin.contract(),
            'all_raw_fullbody_witnesses_checked': True,
            'proxy_is_loss_geometry_not_contact_truth': True,
            'witness_distribution_requirement': 'audit only; endpoint events do not require every observed quadrant to recur',
            'invalid_penetrating_normal_policy': 'raw audit saved; invalid native distance rows excluded; verified complete-solid and other gradients retained; rollout rejected',
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
        torch.save({'schema': cfg.architecture, 'architecture_contract': model.architecture_contract(), 'model': model.state_dict() if state is None else state,
                    'config': config, 'dataset': dataset, 'step': step, 'metrics': metrics,
                    'condition_contract': dataset, 'robot_asset_json': encode_robot_asset_json(),
                    'constraint_training_state': None if constrained_step is None or state is not None else constrained_step.state_dict(),
                    'optimizer': None if state is not None else optimizer.state_dict()}, path)

    @torch.no_grad()
    def evaluate(teacher=False):
        model.eval(); totals, maxima = {}, {}
        layout_errors, layout_safe_errors, visible_count = [], [], 0
        for indices in validation.split(cfg.batch_size):
            observation = take(inputs, indices)
            labels = labels_for(observation, indices)
            if teacher:
                visible = teacher_available(indices, labels)
                indices = indices[visible]
                if not len(indices): continue
                observation = take(inputs, indices); labels = labels_for(observation, indices)
                prediction = model(**observation, **teacher_conditions(indices, labels),
                    teacher_mask=torch.ones(len(indices), device=device, dtype=torch.bool))
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
            error, complete, count = realized_layout(prediction, observation, queried, indices)
            eligible = complete & (count >= 2)
            layout_errors.extend((100 * error[eligible].sqrt()).cpu().tolist())
            safe = eligible & (metrics['newton_generation_accepted'] > .5) & (metrics['joint_violation_rad'] <= runtime_acceptance.joint_excess_rad)
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
        for module in model.compilable_modules():
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
    dataset['training_stage'] = cfg.training_stage
    if cfg.training_stage != 'legacy_mixed':
        dataset['continuous_demonstration_training']['enabled'] = False
        dataset['reference_warmup']['steps'] = None
        dataset['reference_warmup']['teacher_schedule_clock'] = 'explicit stage; no scheduled mixing'
        if 'parallel_rollout_contract' in dataset:
            dataset['parallel_rollout_contract'].update(
                demo_end_resets=cfg.training_stage == 'demonstration_rollout',
                allow_post_demo=cfg.training_stage == 'post_demonstration')
        dataset['stage_contract'] = dict(schema='separate_training_stages_v1',
            reference_batches=cfg.training_stage == 'single_segment',
            demonstration_supervision=cfg.training_stage != 'post_demonstration',
            teacher_conditioning=cfg.training_stage == 'single_segment',
            starts=('verified_stage2_terminals' if cfg.training_stage == 'post_demonstration' else
                    'random_demonstration_states' if cfg.training_stage == 'demonstration_rollout' else 'demonstrated_segments'),
            execution_only_updates=cfg.training_stage == 'post_demonstration',
            rejection_policy=('reset_episode' if cfg.training_stage == 'post_demonstration' else
                              'rollback' if cfg.training_stage == 'demonstration_rollout' else 'not_applicable'),
            validation='single-segment diagnostic, separate from recursive stage statistics')
        (cfg.output/'dataset.json').write_text(json.dumps(dataset, indent=2))
    history, best_score, best_step, best_state = [], float('inf'), None, None
    performance_history = []
    begun = time.perf_counter()
    if cfg.resume_training_state is not None:
        from generator.training_state import restore_training_state, SCHEMA as TRAINING_STATE_SCHEMA
        resume = torch.load(cfg.resume_training_state, map_location='cpu', weights_only=False)
        if resume.get('schema') != TRAINING_STATE_SCHEMA:
            raise ValueError('Historical partial state cannot resume this experiment')
        model.load_state_dict(resume['model'], strict=True)
    else:
        resume = None
    baseline = {'autonomous': evaluate(), 'teacher_conditioned': evaluate(teacher=True)}
    (cfg.output / 'initial_evaluation.json').write_text(json.dumps(baseline, indent=2))
    if resume is None:
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
    def make_pool(stage=None, bank=None):
        from generator.parallel_rollout_pool import ParallelRolloutPool
        from generator.staged_rollout_pool import StagedRolloutPool
        stage = stage or cfg.training_stage
        cls = ParallelRolloutPool if stage == "legacy_mixed" else StagedRolloutPool
        extra = {} if stage == "legacy_mixed" else dict(stage=stage, terminal_bank=bank)
        return cls(inputs,train,successor,cfg.parallel_pool_size, **extra,
            recover_failed_states=cfg.recover_failed_states,max_episode_steps=cfg.parallel_episode_limit,
            max_retries=cfg.parallel_retry_limit,
            allow_post_demo=cfg.parallel_allow_post_demo,
            global_directions=global_directions if cfg.require_forward_progress else None,
            progress_window=cfg.progress_window,progress_minimum_m=cfg.progress_minimum_m)
    def record_training_diagnostics(metrics):
        for name in ('region_gradient_valid', 'imitation_supervised', 'newton_contact_accepted',
                     'newton_generation_accepted', 'plan_contact_exact', 'plan_role_exact',
                     'solid_penetration_cm', 'penetration_exceeded', 'penetration_unknown',
                     'role_validity_invalid_keep', 'role_validity_absent_keep_probability',
                     'reference_keep_point_conflict_parts', 'reference_keep_pose_imitation_masked',
                     *(f'loss_contribution_{name}' for name in LOSS_GROUPS), 'loss_contribution_legacy_keep'):
            if cfg.training_stage == 'post_demonstration' and name in (
                    'plan_contact_exact', 'plan_role_exact', 'reference_keep_point_conflict_parts',
                    'reference_keep_pose_imitation_masked'):
                continue
            if name in metrics:
                training_diagnostics[name] = training_diagnostics.get(name, 0.) + metrics[name].detach().sum()

    start_step = 1
    if resume is not None:
        from generator.staged_rollout_pool import validate_stage_transition
        stage_changed = validate_stage_transition(resume['config'], config)
        if resume['pool'] is not None:
            old_stage = resume['config'].get('training_stage', 'legacy_mixed')
            bank = resume['pool'].get('staged', {}).get('terminal_bank')
            pool = make_pool(old_stage, bank)
        if resume['pending_plan'] is not None:
            from generator.pending_contact_plan import PendingContactPlans
            pending = PendingContactPlans(cfg.parallel_pool_size, device)
        restored = restore_training_state(resume, model=model, optimizer=optimizer, scheduler=scheduler,
            pool=pool, pending=pending, config=config, robot_asset_json=encode_robot_asset_json())
        if restored['optimizer_migration'] is not None:
            (cfg.output/'resume_gradient_isolation.json').write_text(json.dumps(restored['optimizer_migration'], indent=2))
            print(json.dumps({'optimizer_migration': restored['optimizer_migration']}), flush=True)
        training_predictions = restored['training_predictions']
        history, best_score, best_step, best_state = (restored[k] for k in
            ('history', 'best_score', 'best_step', 'best_state'))
        from generator.training_state import capture_training_state
        # Audit the exact fork state before the first resumed update. This also
        # distinguishes restore errors from CUDA/solver nondeterminism later.
        fork_state = capture_training_state(step=restored['step'], model=model, optimizer=optimizer,
            scheduler=scheduler, pool=pool, pending=pending, config=config,
            robot_asset_json=encode_robot_asset_json(), training_predictions=training_predictions,
            history=history, best_score=best_score, best_step=best_step, best_state=best_state)
        if stage_changed:
            fork_state['config'] = dict(config, training_stage=resume['config']['training_stage'])
        torch.save(fork_state, cfg.output/'restored_initial_state.pt')
        del fork_state
        if stage_changed:
            bank = None if pool is None else pool.terminal_bank
            pool = make_pool(cfg.training_stage, bank)
            best_score, best_step, best_state = float('inf'), None, None
            (cfg.output/'stage_transition.json').write_text(json.dumps(dict(
                source=str(cfg.resume_training_state), from_stage=resume['config']['training_stage'],
                to_stage=cfg.training_stage, at_step=restored['step'],
                terminal_seed_count=len(pool.terminal_bank)), indent=2))
            from generator.training_state import save_epoch_training_state
            entry = capture_training_state(step=restored['step'], model=model, optimizer=optimizer,
                scheduler=scheduler, pool=pool, pending=pending, config=config,
                robot_asset_json=encode_robot_asset_json(), training_predictions=training_predictions,
                history=history, best_score=best_score, best_step=best_step, best_state=best_state)
            save(cfg.output/f"step_{restored['step']:04d}.pt", restored['step'], baseline['autonomous'])
            save_epoch_training_state(entry, cfg.output)
            del entry
        start_step = restored['step'] + 1
        if start_step > (cfg.stop_after_step or cfg.steps):
            raise ValueError('Resume state is already at or beyond the requested stop step')
        print(json.dumps(dict(resumed_from_step=restored['step'],
            isolate_unsupervised_execution_gradients=cfg.isolate_unsupervised_execution_gradients)), flush=True)
        del resume
    else:
        from generator.training_state import capture_training_state, save_epoch_training_state
        initial_state = capture_training_state(step=0, model=model, optimizer=optimizer,
            scheduler=scheduler, pool=pool, pending=pending, config=config,
            robot_asset_json=encode_robot_asset_json(), training_predictions=training_predictions,
            history=history, best_score=best_score, best_step=best_step, best_state=best_state)
        save_epoch_training_state(initial_state, cfg.output)
        del initial_state
    for step in range(start_step, (cfg.stop_after_step or cfg.steps) + 1):
        if device.type == 'cuda': torch.cuda.synchronize(device)
        step_begun = time.perf_counter(); timings = {}
        model.train()
        reference_warmup, probability = training_course(step,
            reference_warmup_steps=cfg.reference_warmup_steps,
            execution_only=cfg.execution_only, fixed=cfg.fixed_teacher_probability)
        if cfg.training_stage != 'legacy_mixed':
            reference_warmup = cfg.training_stage == 'single_segment'
            probability = 1. if reference_warmup else 0.
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
        demonstration_predictions = 0
        combined_total, rollout_weighted_total, gradient_norm_total, batches = 0., 0., 0., 0
        constraint_accepted_updates = 0
        depth_outputs = [0] * rollout_depth
        failure_counts = {'own_contact': 0, 'own_safety': 0, 'task_contact': 0, 'joint_limit': 0,
                          'penetration_exceeded': 0, 'penetration_unknown': 0,
                          'penetration_only': 0, 'penetration_and_contact': 0,
                          'retries_used': 0, 'advances_used': 0, 'stopped': 0,
                          'severe_penetration_resets': 0, 'no_contact_resets': 0, 'invalid_resets': 0, 'no_forward_progress_resets': 0}
        if pool is not None:
            failure_counts = {key.replace('_resets', '_rejections'): value for key, value in failure_counts.items()}
        prediction_start=training_predictions
        pool_counts={key:torch.zeros((),device=device,dtype=torch.long) for key in
            ('generated_inputs','advances','retries','resets','terminal_resets','invalid_resets','timeout_resets',
             'periodic_resamples','rejection_resets','rollbacks','accepted_transitions','invalid_rejections','severe_penetration_rejections',
             'no_contact_rejections','no_forward_progress_rejections',
             'demo_exhausted','autonomous_states','autonomous_inputs','severe_penetration_resets','no_contact_resets','no_forward_progress_resets')}
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
                pool_counts['autonomous_inputs']+=(~labels['supervised']).sum()
                repair = pending.batch(slots) if pending is not None else {}
                if pending is not None:
                    labels['planner_weight'] = planner_weights(repair['repair_mask'])
                teacher=teacher_mask_for_batch(probability,target['planned_contact'][indices],labels['contact_cell_valid'],
                    region_known=target['region_label_known'][indices] if cfg.region_plan else None)
                teacher &= labels['supervised']
                if pending is not None: teacher &= ~repair['repair_mask']
                optimizer.zero_grad(set_to_none=True)
                with timed('forward'):
                    execution_mask = (dict(execution_observation_supervised=labels['supervised'])
                        if cfg.isolate_unsupervised_execution_gradients or cfg.training_stage == 'post_demonstration' else {})
                    prediction=model(**observation, **teacher_conditions(indices, labels), teacher_mask=teacher, **repair, **execution_mask)
                loss,rm,queried=objective(prediction,observation,indices,labels,
                    patch_cache=pool_patch_cache, patch_keys=slots)
                record_training_diagnostics(rm)
                combined=loss.mean()
                with timed('backward_optimizer'):
                    combined.backward()
                    prepare_optimizer_gradients(model, labels['supervised'],
                        policy=("planner_supervised_v1" if cfg.training_stage == "post_demonstration" else cfg.optimizer_gradient_ownership),
                        isolate_unsupervised_execution_gradients=(cfg.isolate_unsupervised_execution_gradients or cfg.training_stage == "post_demonstration"))
                    gradient_norm=torch.stack(list(model.clip_training_gradients().values())).norm()
                    optimizer.step()
                gradient_norm_total+=statistic(gradient_norm);batches+=1
                total_count+=count;training_predictions+=count
                teacher_count+=statistic(teacher.sum(),integer=True)
                total_loss+=statistic(loss.detach().sum());combined_total+=statistic(loss.detach().sum())
                accepted=acceptance(queried[1],target['planned_contact'][indices],target['planned_surface'][indices])['contact_accepted']
                checks=endpoint_feedback_checks(rm, accepted, labels['supervised'], limits=runtime_acceptance)
                safe=checks['accepted']
                recoverable, reset_reasons = continuation(prediction.qpos, indices, queried, rm)
                if pending is not None:
                    own_complete = ((rm['newton_generation_accepted'] > .5)
                        & (rm['own_endpoint_layout_accepted'] > .5))
                    # Completed wrong-task intent is not an unfinished plan.
                    # Reset rather than relabeling that endpoint as the old event.
                    recoverable &= ~own_complete
                with timed('pool_feedback'):
                    stage_reasons = (dict(reset_reasons, **endpoint_rejection_masks(checks, reset_reasons))
                        if cfg.training_stage != 'legacy_mixed' else reset_reasons)
                    feedback=pool.update(slots,prediction.qpos,queried[1],safe,recoverable,fresh_observation,reset_reasons=stage_reasons)
                    if pool_patch_cache is not None:
                        # Only store committed states that survived periodic resampling.
                        # Rejected states retain their patch; changed reset states miss
                        # the exact pose/scene check on their next lookup.
                        with torch.no_grad():
                            committed = (safe & recoverable
                                & (pool.observation['current_q'][slots] == prediction.qpos).all(-1)
                                & (tensor_scene_ids[pool.indices[slots]] == tensor_scene_ids[indices]))
                            chosen = committed.nonzero().flatten()
                            if len(chosen):
                                patch = bind_keep_patch(model.fk, prediction.qpos.detach(), queried[1],
                                    sample_indices=chosen)
                                pool_patch_cache.put(slots[chosen], prediction.qpos[chosen],
                                    tensor_scene_ids[indices[chosen]], patch)
                    if pending is not None:
                        issued_world = plan_points_in_pose_frame(observation['current_q'], prediction.conditioned_points_local)
                        pending.update(slots, prediction, issued_world, pool.retries[slots] > 0)
                for key in pool_counts:
                    if key not in ('generated_inputs','autonomous_inputs'):pool_counts[key]+=feedback[key]
                for key, mask in endpoint_rejection_masks(checks, reset_reasons).items():
                    failure_counts[key]+=statistic(mask.sum(),integer=True)
                phase_advances+=statistic(feedback['advances'],integer=True)
                failure_counts['own_contact']+=statistic((rm['newton_contact_accepted'] <= .5).sum(),integer=True)
                failure_counts['own_safety']+=statistic((rm['newton_generation_accepted'] <= .5).sum(),integer=True)
                failure_counts['task_contact']+=statistic(((~accepted) & labels['supervised']).sum(),integer=True)
                failure_counts['joint_limit']+=statistic((rm['joint_violation_rad'] > runtime_acceptance.joint_excess_rad).sum(),integer=True)
                for reason, mask in reset_reasons.items():
                    failure_counts[reason+'_rejections'] += statistic(mask.sum(), integer=True)
                for key,source in [('retries_used','retries'),('advances_used','advances'),('stopped','resets')]:
                    failure_counts[key]+=statistic(feedback[source],integer=True)
            depth_outputs=age_histogram.cpu().tolist()
            rollout_outputs=sum(depth_outputs[1:])
        order_generator = torch.Generator(device=device).manual_seed(cfg.seed+step) if cfg.comparison_batch_order else None
        shuffled=(train[:0] if pool is not None and (cfg.training_stage != "legacy_mixed" or not cfg.continuous_demonstration_training)
                  else train[torch.randperm(len(train), device=device, generator=order_generator)])
        for indices in shuffled.split(cfg.batch_size):
            if not len(indices):continue
            if cfg.training_prediction_budget is not None:
                remaining=cfg.training_prediction_budget-training_predictions
                if remaining <= 0:break
                indices=indices[:remaining]
            optimizer.zero_grad(set_to_none=True)
            observation = take(inputs, indices); labels = take(fixed_labels, indices)
            visible = teacher_available(indices, labels)
            reference_probability = reference_batch_teacher_probability(probability,
                parallel_feedback=pool is not None, continuous_demonstrations=cfg.continuous_demonstration_training)
            teacher = teacher_mask_for_batch(reference_probability,target['planned_contact'][indices],labels['contact_cell_valid'],
                region_known=target['region_label_known'][indices] if cfg.region_plan else None)
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
                    pred = model(**observation, **teacher_conditions(indices, labels), teacher_mask=torch.ones_like(teacher))
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
                            forward_inputs=cpu(dict(**observation, **teacher_conditions(indices, labels),
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
                prediction = model(**observation, **teacher_conditions(indices, labels), teacher_mask=teacher)
            loss, first_metrics, first_queried = objective(prediction, observation, indices, labels)
            record_training_diagnostics(first_metrics)
            training_predictions+=len(indices)
            demonstration_predictions+=len(indices)
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
                    action = model(**rinput, **(dict(execution_observation_supervised=rsupervised)
                        if cfg.isolate_unsupervised_execution_gradients or cfg.training_stage == 'post_demonstration' else {}))
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
                checks = endpoint_feedback_checks(rm, accepted, rsupervised, limits=runtime_acceptance)
                safe = checks['accepted']
                failure_counts['own_contact'] += statistic((rm['newton_contact_accepted'] <= .5).sum(), integer=True)
                failure_counts['own_safety'] += statistic((rm['newton_generation_accepted'] <= .5).sum(), integer=True)
                failure_counts['task_contact'] += statistic(((~accepted) & rsupervised).sum(), integer=True)
                failure_counts['joint_limit'] += statistic((rm['joint_violation_rad'] > runtime_acceptance.joint_excess_rad).sum(), integer=True)
                keep, reset_reasons = continuation(action.qpos, rindices, queried, rm)
                for key, mask in endpoint_rejection_masks(checks, reset_reasons).items():
                    failure_counts[key] += statistic(mask.sum(), integer=True)
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
        # CosineAnnealingLR rises again beyond T_max. Extended staged runs
        # retain the terminal scheduler state and its minimum learning rate.
        if not cfg.constant_learning_rate and scheduler.last_epoch < scheduler.T_max:
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
        training_phase = (cfg.training_stage if cfg.training_stage != 'legacy_mixed' else
            ('reference_warmup' if reference_warmup else ('persistent_parallel' if pool is not None else 'nested_rollout')))
        stage_rollout = pool.stage_metrics() if pool is not None and hasattr(pool, 'stage_metrics') else None
        if stage_rollout is not None:
            with (cfg.output/'stage_rollouts.jsonl').open('a') as log:
                log.write(json.dumps(dict(step=step, **stage_rollout))+'\n')
            pool.finished_episodes.clear()
        performance_history.append({'step': step, 'training_seconds': train_seconds,
            'training_mode':training_phase, 'training_diagnostics': diagnostics,
            'stage_rollout': stage_rollout,
            'training_predictions':actual_predictions,'total_training_predictions':training_predictions,
            'demonstration_predictions':demonstration_predictions,
            'predictions_per_second':actual_predictions/train_seconds,'optimizer_updates':constraint_accepted_updates if cfg.constraint_training else batches,
                   'optimizer_proposals':batches,
            'parallel_pool':parallel_metrics,
            'keep_patch_cache': {name: dict(query_samples=cache.query_samples, hit_samples=cache.hit_samples,
                source_samples=cache.source_samples, storage_bytes=cache.storage_bytes()) for name, cache in
                (('reference', reference_patch_cache), ('pool', pool_patch_cache)) if cache is not None},
            'optimized_loss': combined_total/total_count, 'base_loss': total_loss/total_count,
            'rollout_depth_outputs': depth_outputs, 'rollout_failure_counts': failure_counts,
            'training_component_seconds': train_timings})
        (cfg.output / 'performance.json').write_text(json.dumps(performance_history, indent=2))
        budget_finished=cfg.training_prediction_budget is not None and training_predictions >= cfg.training_prediction_budget
        if step == 1 or step % cfg.evaluation_every == 0 or step == cfg.steps or step == cfg.stop_after_step or budget_finished:
            metrics = evaluate()
            teacher_metrics = evaluate(teacher=True)
            score = endpoint_validation_score(metrics)
            row = {'step': step, 'teacher_probability': probability, 'teacher_fraction': teacher_count / total_count,
                   'training_phase': training_phase, 'stage_rollout': stage_rollout,
                   'validation_kind': 'single_segment_diagnostic_not_recursive_stage_success', 'teacher_schedule_step': max(0, step-cfg.reference_warmup_steps),
                   'training_diagnostics': diagnostics,
                   'training_predictions':actual_predictions,'total_training_predictions':training_predictions,
                   'demonstration_predictions':demonstration_predictions,
                   'predictions_per_second':actual_predictions/train_seconds,'optimizer_updates':constraint_accepted_updates if cfg.constraint_training else batches,
                   'optimizer_proposals':batches,
                   'parallel_pool':parallel_metrics,
                   'loss': total_loss / total_count, 'validation_score': score,
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
                save(cfg.output / 'best_validation_score.pt', step, metrics, best_state)
            (cfg.output / 'progress.json').write_text(json.dumps({'history': [r for r in history if r.get('training_phase') == training_phase]}, indent=2))
            from generator.training_state import capture_training_state, save_epoch_training_state
            training_state = capture_training_state(step=step, model=model, optimizer=optimizer,
                scheduler=scheduler, pool=pool, pending=pending, config=config,
                robot_asset_json=encode_robot_asset_json(), training_predictions=training_predictions,
                history=history, best_score=best_score, best_step=best_step, best_state=best_state)
            save_epoch_training_state(training_state, cfg.output)
            del training_state
        if budget_finished:break
    (cfg.output / 'report.json').write_text(json.dumps({'best_validation_score_step': best_step,
        'best_validation_score': best_score, 'history': history,
        'final_selection': 'run independent full-chain evaluator over milestones; historical score is diagnostic'}, indent=2))
    print(json.dumps({'training_complete': True, 'steps': step,'training_predictions':training_predictions}), flush=True)


if __name__ == '__main__':
    main()
