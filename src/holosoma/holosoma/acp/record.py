#!/usr/bin/env python3
"""Manually record ONE expert rollout as a native-contact reference dataset.

This command only records data. It does not start expert/student training,
choose a checkpoint, replace an existing dataset, or rerun fixed-q physics.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config


@dataclass(frozen=True)
class Options:
    checkpoint: Path
    manifest: Path
    output_dir: Path
    motion_index: int = 0
    max_steps: int = 0
    allow_partial: bool = False
    terrain_observation: Literal["ground_truth", "fused_raw", "fused_neural"] = "ground_truth"
    mapping_checkpoint: str | None = None
    tracker_inputs: Literal["native", "sensor_interface"] = "native"


def main():
    launcher = parse_isaaclab_launcher_args("Record one WBT expert's actual contacts and executed trajectory")
    options = tyro.cli(Options)
    if options.tracker_inputs == "sensor_interface" and options.terrain_observation == "ground_truth":
        raise ValueError("sensor_interface requires fused_raw or fused_neural terrain observations")
    import numpy as np
    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
    from holosoma.acp.contacts import measured_contacts
    from holosoma.acp import observations as acp_observations
    from holosoma.acp.labels import PARTS, REFERENCE_SCHEMA
    from holosoma.utils.wbt_experiment import checkpoint_hash
    from somaforge_core.contact_face_selection import FACE_POLICY
    from somaforge_core.motion_schema import NEWTON_ROLLOUT_KINEMATICS_BACKEND, newton_rollout_kinematics_provenance
    from somaforge_core.robot_assets import canonical_g1_asset_metadata

    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw_contacts"
    raw_dir.mkdir()
    manifest = load_motion_terrain_manifest(str(options.manifest))
    selected = manifest["motion_files"][options.motion_index]
    source = Path(selected["motion_file"])
    with np.load(source, allow_pickle=False) as bundle:
        original = {k: bundle[k].copy() for k in bundle.files}
    single = {k: v for k, v in manifest.items() if k not in {"path", "base_dir"}}
    single["motion_files"] = [selected]
    input_manifest = output / "source_manifest.json"
    input_manifest.write_text(json.dumps(single, indent=2)+"\n")
    config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(options.checkpoint)))
    motion = config.command.setup_terms["motion_command"]
    motion_cfg = dataclasses.replace(motion.params["motion_config"], motion_manifest=str(input_manifest),
        reset_sampler="uniform", use_start_probe_envs=False, use_group_probe_envs=False,
        start_at_timestep_zero_prob=1., freeze_at_timestep_zero_prob=0.,
        fixed_motion_assignment_in_eval=True, hold_at_motion_end_in_eval=True,
        noise_to_initial_pose=dataclasses.replace(motion.params["motion_config"].noise_to_initial_pose, overall_noise_scale=0.))
    config = dataclasses.replace(config,
        training=dataclasses.replace(config.training, num_envs=1, checkpoint=None),
        command=dataclasses.replace(config.command, setup_terms={**config.command.setup_terms,
            "motion_command": dataclasses.replace(motion, params={**motion.params, "motion_config": motion_cfg})}),
        terrain=dataclasses.replace(config.terrain, terrain_term=dataclasses.replace(config.terrain.terrain_term,
            motion_matched_manifest=str(input_manifest),
            spawn=dataclasses.replace(config.terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.))),
        simulator=dataclasses.replace(config.simulator, config=dataclasses.replace(config.simulator.config,
            sim=dataclasses.replace(config.simulator.config.sim,
                max_episode_length_s=max(config.simulator.config.sim.max_episode_length_s,
                    (len(original["joint_pos"])+2) / float(np.asarray(original["fps"]).item()))))),
    )
    config = sync_launcher_headless_config(config, launcher)
    class RecordingContext(TrainingContext):
        def __exit__(self, exc_type, exc_value, exc_tb):
            if exc_value is not None:
                import traceback
                traceback.print_exception(exc_type, exc_value, exc_tb)
            return super().__exit__(exc_type, exc_value, exc_tb)

    with RecordingContext(config, launcher_args=launcher):
        from holosoma.utils.helpers import get_class
        env = get_class(config.env_class)(get_tyro_env_config(config), device=launcher.device)
        env.set_is_evaluating()
        algo = get_class(config.algo._target_)(env=env, config=config.algo.config,
            log_dir=str(output / "inference"), device=launcher.device)
        algo.setup()
        algo.load_for_inference(str(options.checkpoint))
        algo._eval_mode()
        command = env.command_manager.get_state("motion_command")
        sim = env.simulator
        names = [str(x) for x in original["body_names"]]
        body_ids, found_names = sim._robot.find_bodies(names, preserve_order=True)
        if list(found_names) != names:
            raise ValueError(f"Executed articulation is missing canonical source bodies: {set(names)-set(found_names)}")
        frames = []
        control_frames = []
        lidar_frames = []
        last_lidar_time = -float("inf")
        # Mapping affects saved planner observations; the selected dense expert
        # continues to receive its original configured observations.
        env.acp_settings = {**getattr(env, "acp_settings", {}),
            "terrain_observation": options.terrain_observation, "mapping_checkpoint": options.mapping_checkpoint}
        env.acp_perception = None
        env.acp_sensor_time = 0.0
        history = torch.zeros(1, 5, 6+3*env.num_dof, device=env.device)
        command.acp_goal_lookahead = torch.full((1,), 50, dtype=torch.long, device=env.device)
        native_reward = env._compute_reward

        def capture():
            nonlocal last_lidar_time
            native_reward()
            data = measured_contacts(env)
            index = len(frames)
            env.acp_sensor_time = (index+1)*float(env.dt)
            if index == 0:
                (output / "binding.json").write_text(json.dumps(env._acp_contact_reader.binding, indent=2)+"\n")
            evidence = env._acp_contact_reader.read_evidence()
            np.savez_compressed(raw_dir / f"frame_{index:06d}.npz", **evidence.export())
            origin = command._reference_env_origins[0]
            root = sim.robot_root_states[0]
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = acp_observations.proprio(env)
            scan, scan_valid = acp_observations.terrain(env)
            if env.acp_perception is not None:
                latest = env.acp_perception.latest_scan
                timestamp = float(latest["timestamps"][0])
                if timestamp > last_lidar_time:
                    lidar_frames.append({key: value[0].cpu().numpy().copy()
                                         for key, value in latest.items() if key != "sampled_envs"})
                    last_lidar_time = timestamp
            frames.append({
                "joint_pos": torch.cat((root[:3]-origin, root[[6, 3, 4, 5]], sim.dof_pos[0])).cpu().numpy(),
                "joint_vel": torch.cat((root[7:13], sim.dof_vel[0])).cpu().numpy(),
                "body_pos_w": (sim._robot.data.body_pos_w.torch[0, body_ids]-origin).cpu().numpy(),
                "body_quat_w": sim._robot.data.body_quat_w.torch[0, body_ids][:, [3, 0, 1, 2]].cpu().numpy(),
                "body_lin_vel_w": sim._robot.data.body_lin_vel_w.torch[0, body_ids].cpu().numpy(),
                "body_ang_vel_w": sim._robot.data.body_ang_vel_w.torch[0, body_ids].cpu().numpy(),
                "observed_contact_active": data["active"][0].cpu().numpy(),
                "observed_contact_known": data["known"][0].cpu().numpy(),
                "observed_contact_position_w": (data["position_w"][0]-origin).cpu().numpy(),
                "observed_contact_force_w": data["force_w"][0].cpu().numpy(),
                "acp_history": history[0].flatten().cpu().numpy().copy(),
                "acp_goal": acp_observations.goal(env)[0].cpu().numpy(),
                "acp_terrain": scan[0].cpu().numpy(),
                "acp_terrain_valid": scan_valid[0].cpu().numpy(),
            })

        env._compute_reward = capture
        obs = env.reset_all(step_after_reset=False)
        portable = None
        if options.tracker_inputs == "sensor_interface":
            from holosoma.acp.robot_io import G1TrackerInput, RobotSample
            from holosoma.acp.mapping import rotate
            if list(algo.actor_obs_keys) != ["actor_obs"]:
                raise ValueError("Portable tracker supports a single actor_obs group")
            group = config.observation.groups["actor_obs"]
            if group.history_length != 1:
                raise ValueError("Portable tracker requires history_length=1")
            portable = G1TrackerInput(env.dof_names, command.motion_cfg.body_names_to_track,
                command.motion_cfg.body_name_ref[0], env.default_dof_pos, group.terms)
        limit = options.max_steps or len(original["joint_pos"])
        succeeded = False
        failed = False
        with torch.inference_mode():
            for _ in range(limit):
                if portable is None:
                    actor_input = torch.cat([obs[key] for key in algo.actor_obs_keys], dim=1)
                else:
                    # This adapter models ideal encoders/IMU/odometry. All body
                    # transforms used by the actor are reconstructed by URDF FK.
                    root = sim.robot_root_states
                    inverse = root[:, 3:7].clone()
                    inverse[:, :3] *= -1
                    sample = RobotSample(len(frames)*float(env.dt), root[:, :3], root[:, 3:7],
                        root[:, 7:10], rotate(root[:, 10:13], inverse), sim.dof_pos, sim.dof_vel)
                    acp_observations.terrain(env)  # acquire the current LiDAR packet
                    height, known = portable.mapped_height_scan(sample, env.acp_perception.pipeline)
                    actor_input, _ = portable.build(sample, reference_joints=command.joint_pos,
                        reference_velocities=command.joint_vel, reference_anchor_position_w=command.ref_pos_w,
                        reference_anchor_quaternion_xyzw=command.ref_quat_w,
                        previous_action=env.action_manager.action, terrain_height_scan=height)
                    clip = config.observation.clip_observations
                    actor_input = actor_input.clamp(-clip, clip)
                    control_frames.append({"timestamp": np.asarray(sample.timestamp),
                        "actor_input": actor_input[0].cpu().numpy().copy(),
                        "height_valid": known[0].cpu().numpy().copy()})
                actor_input = algo._normalize_actor_obs(actor_input, update=False)
                action = algo.actor.act_inference({"actor_obs": actor_input})
                if portable is not None:
                    control_frames[-1]["action"] = action[0].cpu().numpy().copy()
                obs, _, dones, _ = env.step({"actions": action})
                if bool(dones[0]):
                    ends = env.termination_manager.term_dones.get("motion_ends")
                    bad = env.termination_manager.term_dones.get("bad_tracking")
                    succeeded = bool(ends is not None and ends[0]) and not bool(bad is not None and bad[0])
                    failed = not succeeded
                    break
        complete = not failed and (succeeded or len(frames) >= len(original["joint_pos"])-1)
        if control_frames:
            np.savez_compressed(output/"control_inputs.npz", **{
                key: np.stack([frame[key] for frame in control_frames]) for key in control_frames[0]})
        if lidar_frames:
            np.savez_compressed(output/"lidar_scans.npz", **{
                key: np.stack([frame[key] for frame in lidar_frames]) for key in lidar_frames[0]})
            env.acp_perception.pipeline.map.export(output/"perception_map.npz")
        if len(frames) < 2 or (not complete and not options.allow_partial):
            raise RuntimeError(f"Expert rollout did not complete ({len(frames)} frames); raw audit retained, no reference published")
        data = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
        fps = 1. / float(env.dt)
        data.update(fps=np.asarray(fps, np.float32), body_names=original["body_names"], joint_names=original["joint_names"],
            robot_asset_json=np.asarray(json.dumps(canonical_g1_asset_metadata(collision_profile=getattr(config.robot.asset, "collision_profile", "legacy_v1")))))
        if "vel_cmd" in original:
            data["vel_cmd"] = original["vel_cmd"][:len(frames)]
        meta = dict(schema=REFERENCE_SCHEMA, parts=PARTS, face_selection=FACE_POLICY,
            activation="CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc",
            sampling="last_native_solve_before_policy_observation",
            pose_sampling="control_step_post_physics", strict_pose_contact_same_state=False,
            planner_observations=f"measured_post_physics_noiseless_history_goal_1s_{options.terrain_observation}",
            perception_mode=options.terrain_observation,
            tracker_inputs=options.tracker_inputs,
            sensor_model="ideal_encoders_imu_odometry; instantaneous_terrain_lidar_without_self_occlusion",
            reference_origin_w=command._reference_env_origins[0].cpu().tolist(),
            mapping_checkpoint=options.mapping_checkpoint,
            robot_asset=canonical_g1_asset_metadata(collision_profile=getattr(config.robot.asset, "collision_profile", "legacy_v1")), expert_checkpoint=str(options.checkpoint.resolve()),
            expert_sha256=checkpoint_hash(options.checkpoint), complete=complete,
            raw_contacts=str(raw_dir), source_motion=str(source))
        meta["binding"] = str(output / "binding.json")
        data["observed_contact_provenance_json"] = np.asarray(json.dumps(meta))
        data["kinematics_provenance_json"] = np.asarray(json.dumps(newton_rollout_kinematics_provenance(
            source_path=str(source), source_sha256=checkpoint_hash(source), output_fps=fps, body_names=names)))
        data["kinematics_backend"] = np.asarray(NEWTON_ROLLOUT_KINEMATICS_BACKEND)
        destination = output / "motion.npz"
        np.savez_compressed(destination, **data)
        row = {**selected, "motion_file": str(destination), "motion_sha256": checkpoint_hash(destination),
            "source_file": str(source), "source_sha256": checkpoint_hash(source),
            "kinematics_backend": NEWTON_ROLLOUT_KINEMATICS_BACKEND}
        single.update(motion_files=[row], kinematics_backend=NEWTON_ROLLOUT_KINEMATICS_BACKEND,
            source_kind="native_expert_contact_rollout", reference_complete=complete)
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(single, indent=2)+"\n")
        load_motion_terrain_manifest(str(manifest_path))
        print(json.dumps({"manifest": str(manifest_path), "frames": len(frames), "complete": complete}))


if __name__ == "__main__":
    main()
