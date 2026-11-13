"""Explicit terrain-conditioned SuperMimic experiment using verified SomaForge inputs.

Installed before the training/evaluation entrypoint imports its runtime classes.
The original flat-ground execution path is unchanged when --task_manifest is absent.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from functools import partial

import numpy as np
import torch

_SPEC = None


def collision_masks_match(a_type, a_affinity, b_type, b_affinity):
    """MuJoCo allows a pair when either direction's type/affinity bits overlap."""
    return bool((a_type & b_affinity) or (b_type & a_affinity))


def spawn_floating_canonical(prim_path, cfg, translation=None, orientation=None, *, collision_profile="legacy_v1", **kwargs):
    """Match SomaForge's explicit root-joint override on the composed stage."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import find_matching_prims, get_current_stage
    from pxr import UsdPhysics
    from .lowstep_colliders import remove_duplicate_collision_meshes
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    count = 0
    for robot_prim in find_matching_prims(prim_path, stage=stage):
        report = remove_duplicate_collision_meshes(stage, robot_prim)
        expected_removed=29 if collision_profile=="legacy_v1" else 0
        if len(report["removed"]) != expected_removed or len(report["retained"]) != 29:
            raise RuntimeError(f"Unexpected canonical collision mesh layout: {report}")
        if collision_profile=="surface_v2":
            from somaforge_core.collision_profiles import audit_surface_stage
            audit_surface_stage(stage,robot_prim)
        print(f"Low-step: removed {len(report['removed'])} duplicate collision meshes on the runtime stage", flush=True)
        joint_prim = stage.GetPrimAtPath(f"{robot_prim.GetPath()}/Physics/root_joint")
        if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.FixedJoint):
            raise RuntimeError(f"Canonical root fixed joint missing: {robot_prim.GetPath()}")
        UsdPhysics.Joint(joint_prim).CreateJointEnabledAttr().Set(False)
        count += 1
    if count == 0:
        raise RuntimeError(f"No canonical robot found at {prim_path}")
    print(f"Low-step: disabled {count} canonical root fixed joint(s) on the runtime stage", flush=True)
    return prim


from .lowstep_observations import HeightContactContext, HeightForceContext, OriginalContext, CONTEXT_TYPES


def install(path, contact_model="binary2", reset_sampler="return_bins", probes_per_motion=3, probe_mode="all_long", policy_force_observation="measured", perception="heightmap", perception_buffer_steps=1000, depth_layout="full", depth_encoder="cnn", collision_profile=None, contact_response="legacy", wm_contact_architecture="independent"):
    """Validate a frozen experiment manifest and install matching runtime components."""
    global _SPEC
    from .contact_response import contract as response_contract
    response_contract(contact_response)
    panorama_mode = perception in ("panorama_asymmetric", "panorama_rolling", "panorama_oracle")
    if wm_contact_architecture not in ('independent','shared'):
        raise ValueError('Unknown force WM architecture')
    if wm_contact_architecture=='shared' and (contact_model not in ('force6','force_only') or perception!='heightmap'):
        raise ValueError('Shared force WM currently supports force6/force_only with heightmap')
    if perception not in ("heightmap", "panorama_asymmetric", "panorama_rolling", "panorama_oracle"):
        raise ValueError("Unknown perception mode")
    if panorama_mode and (contact_model != "force6" or policy_force_observation != "none"):
        raise ValueError("Asymmetric panorama requires force6 WM and no actor force observation")
    if policy_force_observation not in ("measured", "none"):
        raise ValueError("Unknown policy force observation mode")
    if policy_force_observation == "none" and contact_model != "force6":
        raise ValueError("Force observation ablation requires force6")
    if contact_model not in CONTEXT_TYPES:
        raise ValueError(f"Unknown contact model: {contact_model}")
    if reset_sampler not in ("return_bins", "completion_recent30_failure_window"):
        raise ValueError(f"Unknown reset sampler: {reset_sampler}")
    if probes_per_motion < 1:
        raise ValueError("At least one statistics probe per motion is required")
    path = Path(path).resolve()
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    if _SPEC is not None:
        if (_SPEC["_sha256"] != signature or _SPEC["_contact_model"] != contact_model
                or _SPEC["_reset_sampler"] != reset_sampler
                or _SPEC.get("_policy_force_observation", "measured") != policy_force_observation
                or _SPEC.get("_perception", "heightmap") != perception
                or _SPEC.get("_depth_layout", "full") != depth_layout
                or _SPEC.get("_depth_encoder", "cnn") != depth_encoder
                or _SPEC.get("_wm_contact_architecture", "independent") != wm_contact_architecture):
            raise RuntimeError("Cannot mix two task manifests in one process")
        if collision_profile is not None and _SPEC["_collision_profile"]!=collision_profile:
            raise RuntimeError("Cannot change collision profile in a running process")
        if _SPEC.get("_contact_response", "legacy") != contact_response:
            raise RuntimeError("Cannot change contact response in a running process")
        return _SPEC
    from .task_manifest import load_task_manifest
    spec = load_task_manifest(path)
    if spec["schema"] != "supermimic_somaforge_lowstep_v1" or not spec["tasks"]:
        raise ValueError("Unsupported or empty low-step experiment")
    if spec.get("terrain_layout", {}).get("collider_representation") != "primitive_box_pose_size_v1":
        raise ValueError("Invalid legacy mesh terrain: regenerate primitive-box terrains before training")
    from supermimic.paths import workspace_root
    soma = workspace_root()
    # The manifest root is historical provenance; installed core owns execution.
    from somaforge_core.robot_assets import canonical_g1_asset_metadata
    from somaforge_core.collision_profiles import profile_metadata
    profile=collision_profile or spec.get("collision_profile","legacy_v1")
    profile_metadata(profile)
    # Task motions retain their FK/base-asset identity. Physical execution is
    # versioned separately and never inherits the source contact/force truth.
    base_asset=canonical_g1_asset_metadata()
    actual = canonical_g1_asset_metadata(collision_profile=profile)
    for field in ("asset_id", "urdf_sha256", "asset_bundle_sha256", "usd_bundle_sha256"):
        if base_asset[field] != spec["robot_asset"][field]:
            raise ValueError(f"Canonical robot changed: {field}")
    for task in spec["tasks"]:
        for field, hash_field in (("csv", "csv_sha256"), ("terrain_usd", "terrain_usd_sha256"),
                                  ("source_motion", "source_motion_sha256"), ("source_terrain", "source_terrain_sha256")):
            if hashlib.sha256(Path(task[field]).read_bytes()).hexdigest() != task[hash_field]:
                raise ValueError(f"Task input changed: {task[field]}")
    spec["_sha256"] = signature
    spec["_contact_model"] = contact_model
    spec["_wm_contact_architecture"] = wm_contact_architecture
    spec["_collision_profile"] = profile
    spec["_contact_response"] = contact_response
    spec["_execution_robot_asset"] = actual
    spec["_reset_sampler"] = reset_sampler
    spec["_policy_force_observation"] = policy_force_observation
    spec["_perception"] = perception
    spec["_depth_layout"] = depth_layout
    spec["_depth_encoder"] = depth_encoder
    force_mode = contact_model in ("force6", "force_only")
    original_mode = contact_model == "original"
    from .fast_mujoco_init import install as install_fast_mujoco_init
    install_fast_mujoco_init()

    # Canonical USD is used verbatim. No collision margins or original asset bytes change.
    from isaaclab.utils import configclass, replace
    from . import robot
    robot.URDF_PATH = actual["urdf_path"]
    robot.USD_PATH = actual["usd_path"]
    robot.MJCF_PATH = spec["visualization_mjcf"]
    robot.G1_CFG = replace(robot.G1_CFG, spawn=replace(
        robot.G1_CFG.spawn, usd_path=actual["usd_path"], func=partial(spawn_floating_canonical,collision_profile=profile),
        activate_contact_sensors=True,
        articulation_props=replace(robot.G1_CFG.spawn.articulation_props, fix_root_link=False),
    ))
    from . import kinematics
    from .lowstep_kinematics import CanonicalKinematics
    kinematics.G1Kinematics = CanonicalKinematics

    from . import motion as motion_module
    original_motion = motion_module.MotionLib

    class TaskMotionLib(original_motion):
        @staticmethod
        def from_dir(directory, kin, device):
            if Path(directory).resolve() != Path(spec["motion_dir"]).resolve():
                raise ValueError("Task manifest and requested motion directory disagree")
            return TaskMotionLib([task["csv"] for task in spec["tasks"]], kin, device, src_fps=50.0)

    motion_module.MotionLib = TaskMotionLib
    if panorama_mode:
        from .panorama import PanoramaContext
        context = PanoramaContext(spec["tasks"], perception_buffer_steps, rolling=perception=="panorama_rolling", oracle=perception=="panorama_oracle", depth_layout=depth_layout, depth_encoder=depth_encoder)
    else:
        context = CONTEXT_TYPES[contact_model](spec["tasks"])
    has_contact = context.contact_dim > 0
    from . import mdp as mdp_module
    original_mdp = mdp_module.MDP
    original_normalize_xi = mdp_module.normalize_xi
    original_obs_dim = mdp_module.OBS_DIM
    original_context_dim = mdp_module.XI_DIM
    mdp_module.XI_DIM = original_context_dim if original_mode else original_context_dim + 1 + context.contact_dim
    mdp_module.XI_DIM += getattr(context, "extra_xi_dim", 0)
    policy_context_dim = context.height_dim if policy_force_observation == "none" else context.dim
    if panorama_mode:
        policy_context_dim = context.policy_dim
    mdp_module.OBS_DIM += policy_context_dim
    mdp_module.CRITIC_EXTRA += context.dim

    def normalize_xi(xi):
        return torch.cat((original_normalize_xi(xi[..., :original_context_dim]), xi[..., original_context_dim:]), -1)

    class TerrainMDP(original_mdp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            zeros = torch.zeros(policy_context_dim, device=self.default_q.device)
            self.obs_noise = torch.cat((self.obs_noise, zeros))
            self.obs_scale = torch.cat((self.obs_scale, torch.ones_like(zeros)))

        def obs(self, state, a_prev, xi, frames):
            base = super().obs(state, a_prev, xi, frames)
            if panorama_mode:
                return torch.cat((base, context.policy_features(state, xi)), -1)
            if policy_force_observation == "none":
                return torch.cat((base, context.heights(state, xi[..., context.task_column])), -1)
            return base if original_mode else torch.cat((base, context.features(state, xi)), -1)

        def critic_base_features(self, state, frames, xi):
            return super().critic_features(state, frames, xi[..., :original_context_dim])

        def critic_features(self, state, frames, xi):
            base = self.critic_base_features(state, frames, xi)
            return base if original_mode else torch.cat((base, context.features(state, xi)), -1)

        def termination(self, state, frames):
            horizontal = (state[0][..., :2] - self.motion.root_p[frames, :2]).norm(dim=-1)
            return super().termination(state, frames) | (horizontal > 0.5)

        def soft_termination(self, state, frames, tz=0.02, tt=0.04):
            base = super().soft_termination(state, frames, tz, tt)
            horizontal = (state[0][..., :2] - self.motion.root_p[frames, :2]).norm(dim=-1)
            return 1 - (1 - base) * (1 - torch.sigmoid((horizontal - 0.5) / 0.04))

    mdp_module.MDP = TerrainMDP
    if not original_mode:
        mdp_module.normalize_xi = normalize_xi
    from . import networks
    original_world_model = networks.WorldModel

    class TerrainWorldModel(original_world_model):
        def __init__(self, hidden=(2048, 2048, 2048)):
            super().__init__(hidden)
            if panorama_mode:
                from .panorama import encoder
                self.depth_encoder = encoder(context.encoder_kind, context.resolution)
                context.depth_encoder = self.depth_encoder
            self.in_dim = mdp_module.INV_DIM + mdp_module.NUM_J + original_context_dim + context.dim
            self.net = networks.mlp(self.in_dim, list(hidden), networks.WM_OUT,
                                    act=torch.nn.SiLU, layer_norm=True)
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)
            self.in_norm = networks.RunningNorm(self.in_dim)
            if has_contact:
                self.contact_head = networks.mlp(self.in_dim, [256, 256], context.contact_dim, act=torch.nn.SiLU)
            if force_mode:
                torch.nn.init.zeros_(self.contact_head[-1].weight)
                torch.nn.init.zeros_(self.contact_head[-1].bias)

        def inputs(self, state, action, xi):
            return torch.cat((mdp_module.invariant_features(state), action,
                              xi[..., :original_context_dim], context.features(state, xi)), -1)

        def contact_logits(self, state, action, xi):
            return networks.run_mlp(self.contact_head, self.in_norm(self.inputs(state, action, xi)))

        def next_context(self, state, action, xi, xi_n):
            if not has_contact:
                return xi
            prediction = self.contact_logits(state, action, xi_n)
            contacts = context.world_forces(state, prediction) if force_mode else prediction.sigmoid()
            return torch.cat((xi[..., :34], contacts, xi[..., 34+context.contact_dim:]), -1)

        def joint_step(self, state, action, xi, xi_n):
            # Both heads use exactly the same current-state encoding, in parallel.
            inputs = self.in_norm(self.inputs(state, action, xi_n))
            next_state = self.step_from_inputs(state, inputs)
            prediction = networks.run_mlp(self.contact_head, inputs)
            contacts = context.world_forces(state, prediction)
            return next_state, torch.cat((xi[..., :34], contacts, xi[..., 52:]), -1)

        def training_predictions(self, window, normalized_inputs):
            """Paired one-step state output and the unchanged contact objective."""
            return (networks.run_mlp(self.net, normalized_inputs),
                    self.contact_loss_from_inputs(window, normalized_inputs[:, None]))

        def contact_loss_from_inputs(self, window, normalized_inputs):
            """Reuse the state head's encoding for paired one-step supervision."""
            return self.contact_loss(window, normalized_inputs=normalized_inputs)

        def contact_loss(self, window, normalized_inputs=None):
            if not has_contact:
                return window["s"].new_zeros(())
            logits = (self.contact_logits(mdp_module.unflatten_state(window["s"]), window["a"],
                                         normalize_xi(window["xi"])) if normalized_inputs is None
                      else networks.run_mlp(self.contact_head, normalized_inputs))
            if force_mode:
                target = context.local_forces(mdp_module.unflatten_state(window["s"]), window["contact_next"])
                loss = torch.nn.functional.smooth_l1_loss(logits, target, reduction="none").mean(-1)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, window["contact_next"], reduction="none").mean(-1)
            return (loss * window["mask"]).sum() / window["mask"].sum().clamp(min=1)

    if not original_mode:
        if wm_contact_architecture=='shared':
            from .shared_force_wm import shared_force_world_model
            networks.WorldModel=shared_force_world_model(TerrainWorldModel,context)
        else:
            networks.WorldModel = TerrainWorldModel
    if panorama_mode:
        from .panorama import policy_class
        networks.Policy = policy_class(networks, context)
    from . import buffer as buffer_module
    if has_contact:
        buffer_module.FIELDS["contact_next"] = (context.contact_dim, torch.float32)
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner import CloneCfg, InclusionSet
    from isaaclab.cloner.cloner_strategies import sequential
    from isaaclab.scene import InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ContactSensorCfg
    from . import env as env_module

    attributes = {}
    annotations = {}
    sensor_groups = []
    for side in (("left", "right") if has_contact else ()):
        foot = (f"{{ENV_REGEX_NS}}/Robot/Geometry/pelvis/{side}_hip_pitch_link/"
                f"{side}_hip_roll_link/{side}_hip_yaw_link/{side}_knee_link/"
                f"{side}_ankle_pitch_link/{side}_ankle_roll_link")
        foot_sensors=[("base",foot)]
        if profile=="legacy_v1":foot_sensors.append(("spheres",foot+f"/{side}_ankle_roll_sphere_.*_link"))
        names=[]
        for suffix, path in foot_sensors:
            name = f"{side}_foot_{suffix}"
            attributes[name] = ContactSensorCfg(prim_path=path, update_period=0.0, history_length=0,
                                               track_friction_forces=force_mode)
            annotations[name] = ContactSensorCfg
            names.append(name)
        sensor_groups.append(tuple(names))
    if force_mode:
        for side in ("left", "right"):
            torso = "{ENV_REGEX_NS}/Robot/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link"
            hand = (f"{torso}/{side}_shoulder_pitch_link/{side}_shoulder_roll_link/"
                    f"{side}_shoulder_yaw_link/{side}_elbow_link/{side}_wrist_roll_link/"
                    f"{side}_wrist_pitch_link/{side}_wrist_yaw_link/{side}_sphere_hand_link")
            names = []
            hand_sensors=[("base",hand)]
            if profile=="legacy_v1":hand_sensors.append(("tip",hand+f"/{side}_sphere_hand_tip_link"))
            for suffix, path in hand_sensors:
                name = f"{side}_hand_{suffix}"
                attributes[name] = ContactSensorCfg(prim_path=path, update_period=0.0, history_length=0,
                                                   track_friction_forces=True)
                annotations[name] = ContactSensorCfg
                names.append(name)
            sensor_groups.append(tuple(names))
        for side in ("left", "right"):
            knee = (f"{{ENV_REGEX_NS}}/Robot/Geometry/pelvis/{side}_hip_pitch_link/"
                    f"{side}_hip_roll_link/{side}_hip_yaw_link/{side}_knee_link")
            name = f"{side}_knee_base"
            attributes[name] = ContactSensorCfg(prim_path=knee, update_period=0.0, history_length=0,
                                               track_friction_forces=True)
            annotations[name] = ContactSensorCfg
            sensor_groups.append((name,))
    for index, task in enumerate(spec["tasks"]):
        name = f"terrain_{index:04d}"
        attributes[name] = AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/Terrain_{index:04d}",
                                      spawn=sim_utils.UsdFileCfg(usd_path=task["terrain_usd"]))
        annotations[name] = AssetBaseCfg
    attributes["robot"] = replace(robot.G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    attributes["clone_cfg"] = CloneCfg(clone_combinations=[
        InclusionSet(assets=[f"terrain_{index:04d}"]) for index in range(context.count)
    ])
    attributes["__annotations__"] = annotations
    scene_type = configclass(type("LowstepSceneCfg", (InteractiveSceneCfg,), attributes))
    env_module.TrackingSceneCfg = scene_type
    original_sim = env_module.TrackingSim
    original_make_sim = env_module.make_sim

    class TerrainSim(original_sim):
        def state(self):
            state = super().state()
            if hasattr(self, "contact_sensors"):
                contacts = []
                for sensors in self.contact_sensors:
                    force = sum((sensor.data.net_forces_w if force_mode else sensor.data.net_normal_forces_w).torch.sum(1)
                                for sensor in sensors)
                    contacts.append(force / context.force_scale if force_mode else (force.norm(dim=-1) > 1.0).float())
                self._latest_contacts = torch.stack(contacts, -2).flatten(-2) if force_mode else torch.stack(contacts, -1)
                self.xi[:, context.contact_slice] = self._latest_contacts
            return state

        def step(self, action, *, precomputed_state=None, tracking_env_ids=None):
            out = super().step(action,precomputed_state=precomputed_state,tracking_env_ids=tracking_env_ids)
            if has_contact:
                out["contact_next"] = self._latest_contacts.clone()
            if panorama_mode:
                self.perception_step += 1
            return out

        def update_perception(self):
            if panorama_mode and perception != "panorama_oracle":
                self.historical_sensor.update(self.perception_step)

        def __init__(self, sim_cfg, num_envs, *args, **kwargs):
            self.collision_profile = profile
            self.execution_robot_asset = actual
            task_world_counts = kwargs.pop('task_world_counts', None)
            if num_envs < context.count:
                raise ValueError("Use at least one environment per task")
            eval_mode = kwargs.get("eval_mode", args[3] if len(args) > 3 else False)
            weights = np.array([1.0 if eval_mode else task.get("sampling_weight", 1.0)
                                for task in spec["tasks"]])
            if task_world_counts is not None:
                counts = np.asarray(task_world_counts)
                if (not eval_mode or counts.shape != (context.count,) or
                        not np.issubdtype(counts.dtype, np.integer) or (counts < 1).any() or counts.sum() != num_envs):
                    raise ValueError('Explicit evaluation world counts must be positive integers summing to num_envs')
                weights = counts.astype(float)
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Task sampling weights must be finite and positive")
            training_env_count = num_envs
            use_retry = reset_sampler != "return_bins" and not eval_mode
            training_mask = np.ones(num_envs, dtype=bool)
            if use_retry:
                base_assignments = sequential(weights, num_envs)
                counts = np.bincount(base_assignments, minlength=context.count)
                if (counts == 0).any():
                    raise ValueError("Every task needs at least one training environment")
                training_mask = np.concatenate([np.r_[np.ones(n, dtype=bool),
                                                       np.zeros(probes_per_motion, dtype=bool)] for n in counts])
                weights = counts + probes_per_motion
                num_envs = int(weights.sum())
            # Constructed below by the base simulator; keep physical allocation
            # and the reference task IDs identical. Evaluation visits every clip.
            env_module.TrackingSceneCfg = partial(scene_type, clone_cfg=CloneCfg(clone_combinations=[
                InclusionSet(assets=[f"terrain_{i:04d}"], weight=float(weight))
                for i, weight in enumerate(weights)
            ]))
            assignments = sequential(weights, num_envs)
            if len(np.unique(assignments)) != context.count:
                raise ValueError("Not enough environments to represent every weighted task")
            self.task_ids = torch.tensor(assignments, dtype=torch.long, device=sim_cfg.device)
            motion = args[0] if args else kwargs["motion"]
            self.training_env_count = training_env_count
            self.training_env_ids = torch.tensor(np.flatnonzero(training_mask), dtype=torch.long, device=sim_cfg.device)
            if use_retry:
                from .retry_sampler import Recent30FailureSampler
                self.reset_sampler_controller = Recent30FailureSampler(
                    self.task_ids, torch.tensor(training_mask, device=sim_cfg.device), motion, probes_per_motion, probe_mode)
                print(f"Statistics-only probes: {num_envs-training_env_count}; training environments: {training_env_count}", flush=True)
            groups = [(motion.bin_clip == index).nonzero(as_tuple=False).flatten()
                      for index in range(context.count)]
            width = max(len(group) for group in groups)
            self.task_bins = torch.stack([torch.cat((group, group[:1].repeat(width - len(group))))
                                          for group in groups])
            self.task_bin_valid = torch.stack([
                torch.arange(width, device=sim_cfg.device) < len(group) for group in groups
            ])
            super().__init__(sim_cfg, num_envs, *args, **kwargs)
            from .contact_response import apply as apply_contact_response
            apply_contact_response(self, contact_response)
            if self.robot.is_fixed_base:
                raise RuntimeError("Low-step requires a floating root; canonical USD root_joint must be disabled at spawn")
            layout = self.scene.clone_plan.topology.world_prototype_layout
            if hasattr(layout, "numpy"):
                layout = layout.numpy()
            if not np.array_equal(np.asarray(layout), assignments):
                raise ValueError("Physics terrain allocation differs from reference motion allocation")
            if spec["terrain_layout"].get("bounding_volume_schema") == "per_world_box_aabb_rbound_v1":
                self._refresh_terrain_bounds()
            self._validate_terrain_colliders()
            if spec.get("group_counts"):
                from collections import Counter
                counts = Counter(spec["tasks"][int(i)]["group"] for i in assignments)
                print(f"Task group environment allocation (eval={eval_mode}): {dict(counts)}", flush=True)
                if use_retry:
                    train_counts = Counter(spec["tasks"][int(i)]["group"] for i in assignments[training_mask])
                    print(f"Training-only group allocation: {dict(train_counts)}", flush=True)
            if not original_mode:
                self.xi = torch.cat((self.xi, self.task_ids.float().unsqueeze(-1),
                                     torch.zeros(num_envs, context.contact_dim, device=sim_cfg.device)), -1)
            if panorama_mode:
                from .panorama import HistoricalSensor
                self.xi = torch.cat((self.xi, torch.zeros(num_envs, context.extra_xi_dim, device=sim_cfg.device)), -1)
                self.perception_step = 0
                if perception != "panorama_oracle":
                    self.historical_sensor = HistoricalSensor(self, context)
            if has_contact:
                self.contact_sensors = [tuple(self.scene[name] for name in group) for group in sensor_groups]
                self.contact_parts = context.parts if force_mode else ("left_foot", "right_foot")
                self.force_scale = context.force_scale if force_mode else 1.0
                self._latest_contacts = self.xi[:, context.contact_slice].clone()
            print(f"Low-step: {context.count} paired terrains, {num_envs} environments; "
                  f"local heightmap {context.height_dim} + {contact_model} {context.contact_dim}; policy force {policy_force_observation}; obs {original_obs_dim + policy_context_dim}", flush=True)

        def _refresh_terrain_bounds(self):
            """Match broadphase bounds to per-world primitive sizes before graph capture.

            Newton updates geom_size from each world's shapes but leaves the
            compiled prototype's geom_aabb/geom_rbound unchanged. MJWarp supports
            batched bounds; retaining prototype bounds can discard real contacts.
            Narrowphase geometry, contact parameters and source assets stay fixed.
            """
            import warp as wp
            solver = self.sim.backend_manager._solver
            model = solver.mjw_model
            indices = [i for i in range(solver.mj_model.ngeom)
                       if "/Terrain_" in solver.mj_model.geom(i).name]
            if not indices or any(solver.mj_model.geom_type[i] != 6 for i in indices):
                raise RuntimeError("Per-world terrain bounds require primitive boxes")
            sizes = wp.to_torch(model.geom_size).expand(self.N, -1, -1)
            aabb = wp.to_torch(model.geom_aabb).expand(self.N, -1, -1, -1).clone()
            radius = wp.to_torch(model.geom_rbound).expand(self.N, -1).clone()
            previous_error = float((aabb[:, indices, 1] - sizes[:, indices]).abs().max())
            aabb[:, indices, 0] = 0
            aabb[:, indices, 1] = sizes[:, indices]
            radius[:, indices] = sizes[:, indices].norm(dim=-1)
            self._terrain_bounds_storage = (aabb, radius)
            model.geom_aabb = wp.from_torch(aabb, dtype=wp.vec3)
            model.geom_rbound = wp.from_torch(radius, dtype=wp.float32)
            print(f"Low-step: refreshed per-world box AABBs/radii; previous max half-size mismatch {previous_error:.6g} m", flush=True)

        def _validate_terrain_colliders(self):
            import re
            import warp as wp
            solver = self.sim.backend_manager._solver
            slots = {}
            for index in range(solver.mj_model.ngeom):
                match = re.search(r"/Terrain_\d+/box_(\d+)_", solver.mj_model.geom(index).name)
                if match:
                    slots[int(match.group(1))] = index
            width = max(len(task["boxes"]) for task in spec["tasks"])
            if set(slots) != set(range(spec["terrain_layout"]["shapes_per_world"])):
                raise RuntimeError("Missing terrain collision shapes in the live solver")
            indices = [slots[i+1] for i in range(width)]
            robot_geoms = [i for i in range(solver.mj_model.ngeom)
                           if "/Robot/" in solver.mj_model.geom(i).name]
            if any(solver.mj_model.geom_type[i] != 6 or not any(
                collision_masks_match(int(solver.mj_model.geom_contype[i]), int(solver.mj_model.geom_conaffinity[i]),
                                      int(solver.mj_model.geom_contype[j]), int(solver.mj_model.geom_conaffinity[j]))
                for j in robot_geoms) for i in indices):
                raise RuntimeError("Terrain boxes must permit collisions with the robot")
            expected_robot_types = {7: 29, 5: 4}
            if self.collision_profile == 'legacy_v1':
                expected_robot_types[2] = 14
            robot_types = [int(solver.mj_model.geom_type[i]) for i in robot_geoms]
            actual_robot_types = {kind: robot_types.count(kind) for kind in set(robot_types)}
            if actual_robot_types != expected_robot_types:
                raise RuntimeError(f"Unexpected {self.collision_profile} robot colliders: "
                                   f"expected {expected_robot_types}, got {actual_robot_types}")
            expected = torch.zeros(context.count, width, 10, device=self.device)
            valid = torch.zeros(context.count, width, device=self.device)
            for i, task in enumerate(spec["tasks"]):
                expected[i, :len(task["boxes"])] = torch.tensor(task["boxes"], device=self.device).reshape(-1, 10)
                valid[i, :len(task["boxes"])] = 1
            expected, valid = expected[self.task_ids], valid[self.task_ids]
            actual = [wp.to_torch(getattr(solver.mjw_model, name)).expand(self.N, -1, -1)[:, indices]
                      for name in ("geom_pos", "geom_size", "geom_quat")]
            errors = [((actual[0] - expected[..., :3]).abs() * valid[..., None]).max(),
                      ((actual[1] - expected[..., 7:]*.5).abs() * valid[..., None]).max(),
                      ((1-(actual[2]*expected[..., 3:7]).sum(-1).abs()).abs()*valid).max()]
            if any(float(error) > 1e-5 for error in errors):
                raise RuntimeError(f"Live terrain collision geometry disagrees with task: {errors}")
            if spec["terrain_layout"].get("bounding_volume_schema") == "per_world_box_aabb_rbound_v1":
                all_indices = [slots[i] for i in range(len(slots))]
                sizes = wp.to_torch(solver.mjw_model.geom_size).expand(self.N, -1, -1)[:, all_indices]
                bounds = wp.to_torch(solver.mjw_model.geom_aabb)[:, all_indices]
                radii = wp.to_torch(solver.mjw_model.geom_rbound)[:, all_indices]
                if (not torch.allclose(bounds[..., 0, :], torch.zeros_like(sizes), atol=1e-6)
                        or not torch.allclose(bounds[..., 1, :], sizes, atol=1e-6)
                        or not torch.allclose(radii, sizes.norm(dim=-1), atol=1e-6)):
                    raise RuntimeError("Per-world terrain collision bounds disagree with box sizes")
                print("PASS per-world terrain AABBs and bounding radii, including floor/padding", flush=True)
            print("PASS live terrain collision poses, dimensions, orientations and masks for every environment", flush=True)

        def reset(self, env_ids, frames=None, *, state=None):
            if len(env_ids) == 0:
                return
            if frames is None and hasattr(self, "reset_sampler_controller"):
                frames = self.reset_sampler_controller.sample(env_ids)
            if frames is None:
                m = self.motion
                soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
                self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
                probabilities = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
                task_ids = self.task_ids[env_ids]
                allowed = self.task_bins[task_ids]
                weights = probabilities[allowed] * self.task_bin_valid[task_ids]
                choice = torch.multinomial(weights, 1)
                bins = allowed.gather(1, choice).squeeze(-1)
                start = m.bin_start[bins]
                end = torch.minimum(start + 50, m.clip_end[task_ids])
                frames = start + (torch.rand(len(env_ids), device=self.device) * (end - start).float()).long()
            if not torch.equal(self.motion.clip_id[frames], self.task_ids[env_ids]):
                raise ValueError("Reset attempted to pair a motion with a different terrain")
            super().reset(env_ids, frames, state=state)
            if not original_mode and self.xi.shape[-1] == mdp_module.XI_DIM:
                self.xi[env_ids, context.contact_slice] = 0
            if panorama_mode and hasattr(self, "historical_sensor"):
                self.historical_sensor.reset(env_ids)
                if self.eval_mode and len(env_ids) == self.N:
                    self.perception_step = 0
                    self.historical_sensor.last_step = -1

    def make_sim(device, args):
        if response_contract(contact_response)["manifold_normal"] != "upstream":
            from .contact_manifold_normals import install as install_manifold_normals
            install_manifold_normals(response_contract(contact_response)["manifold_normal"])
        cfg = original_make_sim(device, args)
        if contact_response in ('newton_contacts_v1', 'newton_contacts_v2', 'newton_contacts_v3'):
            cfg.physics.solver_cfg.use_mujoco_contacts = False
        if contact_response in ('newton_contacts_v2','newton_contacts_v3'):
            cfg.physics.collision_decimation = 1
        if contact_response != "legacy":
            cfg.physics.num_substeps = response_contract(contact_response)["substeps"]
            cfg.physics.solver_cfg.enable_multiccd = response_contract(contact_response)["enable_multiccd"]
        if contact_response=='newton_contacts_v3':
            cfg.physics.solver_cfg.solver='cg'
        cfg.physics.solver_cfg.njmax = 1000
        cfg.physics.solver_cfg.nconmax = 256
        return cfg

    env_module.TrackingSim = TerrainSim
    env_module.make_sim = make_sim
    _SPEC = spec
    return spec


def checkpoint_metadata(spec):
    from .lowstep_colliders import COLLISION_SCHEMA
    context_type = CONTEXT_TYPES[spec.get("_contact_model", "binary2")]
    metadata = {"task_manifest_sha256": spec["_sha256"],
            "robot_asset_json": json.dumps(spec.get("_execution_robot_asset",spec["robot_asset"]), sort_keys=True),
            "task_schema": spec["schema"],
            "observation_schema": context_type.schema,
            "robot_collision_schema": COLLISION_SCHEMA}
    if spec.get("_collision_profile","legacy_v1")!="legacy_v1":
        from somaforge_core.collision_profiles import profile_metadata
        metadata.update(profile_metadata(spec["_collision_profile"]))
        metadata["robot_collision_schema"]="canonical_continuous_surface_meshes_v2"
    if spec.get("_policy_force_observation") == "none":
        metadata["observation_schema"] += "_policy_without_force_v1"
    if spec.get("_perception", "").startswith("panorama_"):
        from .panorama import PanoramaContext, PERCEPTION_SCHEMA, ROLLING_SCHEMA, ORACLE_SCHEMA
        rolling = spec["_perception"] == "panorama_rolling"
        oracle = spec["_perception"] == "panorama_oracle"
        metadata["observation_schema"] = ORACLE_SCHEMA if oracle else ROLLING_SCHEMA if rolling else PERCEPTION_SCHEMA
        metadata["perception_contract"] = json.dumps(PanoramaContext.contract(rolling, oracle, spec.get("_depth_layout", "full"), spec.get("_depth_encoder", "cnn")), sort_keys=True)
    if spec.get("_contact_model") == "original":
        metadata["world_model_schema"] = "supermimic_original_133_to_70_v1"
    if spec.get("_contact_model") in ("force6", "force_only"):
        metadata["force_contract"] = json.dumps({
            "parts": HeightForceContext.parts, "components": "Fx,Fy,Fz",
            "measurement": "world-frame total contact force (normal + friction), N",
            "scale_N": HeightForceContext.force_scale,
            "head_frame": "current pelvis heading",
            "loss": "SmoothL1 beta=1 on scaled forces; auxiliary weight=0.1",
        }, sort_keys=True)
    if spec.get("_contact_response", "legacy") != "legacy":
        from .contact_response import metadata as response_metadata
        metadata["contact_response_contract"] = response_metadata(spec["_contact_response"])
    bounds_schema = spec["terrain_layout"].get("bounding_volume_schema")
    if bounds_schema:
        metadata["terrain_bounding_volume_schema"] = bounds_schema
    return metadata


def validate_checkpoint(checkpoint, metadata, *, allow_collision_transfer=False):
    from .contact_response import validate_identity
    validate_identity(checkpoint, metadata)
    mismatched={key for key,value in metadata.items() if checkpoint.get(key)!=value}
    if not mismatched:return None
    if "robot_asset_json" in mismatched:
        # Asset locations change when a checkout moves. Identity is the complete
        # fingerprint/profile contract, excluding only the two location fields.
        from somaforge_core.robot_assets import validate_g1_asset_metadata
        source_asset=json.loads(checkpoint["robot_asset_json"])
        target_asset=json.loads(metadata["robot_asset_json"])
        validate_g1_asset_metadata(source_asset,context="checkpoint robot asset")
        validate_g1_asset_metadata(target_asset,context="task robot asset")
        locations={"urdf_path","usd_path"}
        if ({k:v for k,v in source_asset.items() if k not in locations}
                =={k:v for k,v in target_asset.items() if k not in locations}):
            mismatched.remove("robot_asset_json")
    if not mismatched:return None
    collision_keys={"robot_asset_json","robot_collision_schema","collision_profile","collision_profile_sha256"}
    if allow_collision_transfer and mismatched<=collision_keys:
        from somaforge_core.robot_assets import validate_g1_asset_metadata
        source=json.loads(checkpoint["robot_asset_json"])
        target=json.loads(metadata["robot_asset_json"])
        validate_g1_asset_metadata(source,context="explicit collision-transfer initialization")
        if source.get('collision_profile','legacy_v1')=='legacy_v1' and target.get('collision_profile')=='surface_v2':
            return dict(source='legacy_v1',target='surface_v2',role='initialization_only',
                        actual_execution_validated=False)
    raise ValueError("Checkpoint/task collision contract mismatch: "+", ".join(sorted(mismatched))+
                     "; new collision physics needs fresh native data; initialization transfer must be explicit")
