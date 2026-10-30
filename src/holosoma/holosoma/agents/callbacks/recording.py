"""Eval callback that records per-step trajectory data to an NPZ file.

Records joint positions, velocities, torques, body poses, contact forces, and
root state for canonical rollout extraction and diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from somaforge_core import canonical_g1_asset_metadata
from somaforge_core.contact_schema import NEWTON_COLLISION_PIPELINE, NEWTON_CONTACT_BACKEND
from somaforge_core.contact_record_storage import compact_snapshot, stack_contact_channel

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import RecordingConfig
from holosoma.utils.safe_torch_import import torch


class EvalRecordingCallback(RLEvalCallback):
    """Records per-step data during evaluation and saves to .npz on completion."""

    def __init__(
        self,
        config: RecordingConfig,
        training_loop: Any = None,
    ):
        super().__init__(config, training_loop)
        self.env_id = config.env_id
        self.profile = config.profile
        if self.profile != "full":
            raise ValueError("Only the full recording profile is supported")
        self.stop_when_done = bool(config.stop_when_done)

        output_path = config.output_path
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        if training_loop is not None and hasattr(training_loop, "log_dir"):
            output_path = str(Path(training_loop.log_dir) / output_path)
        self.output_path = output_path

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict[str, Any] = {}
        self._step_count = 0
        self._initial_state_recorded = False
        self._bootstrap_state_recorded = False
        self._stopped_after_done = False
        self._chunks: list[dict[str, Any]] = []
        self._chunk_start = 0
        if config.chunk_frames < 0:
            raise ValueError("chunk_frames must be nonnegative")
        if config.chunk_frames:
            import faulthandler
            # A stalled first native snapshot must leave actionable evidence.
            faulthandler.dump_traceback_later(60, repeat=True)

    def _records_initial_state(self) -> bool:
        """Respect the requested reset/bootstrap phase origin recording."""
        return bool(self.config.record_initial_state)

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _save(self) -> None:
        """Save recorded data to NPZ."""
        if self._step_count == 0:
            return
        if self.config.chunk_frames:
            self._flush_chunk()
            self._publish_chunks(complete=True)
            return

        arrays: dict[str, np.ndarray] = {}
        for name, values in self._buffers.items():
            if values:
                if name.startswith(('solver_contact_', 'raw_contact_')) and not name.endswith('_count'):
                    arrays[name] = stack_contact_channel(values)
                else:
                    arrays[name] = np.stack(values, axis=0)

        arrays["_metadata_json"] = np.array(json.dumps(self._metadata))

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)

        channel_summary = ", ".join(
            f"{name}{list(arr.shape)}" for name, arr in arrays.items() if name != "_metadata_json"
        )
        logger.info(f"EvalRecordingCallback: saved {self._step_count} steps to {path}\n  Channels: {channel_summary}")

    def _publish_chunks(self, *, complete: bool = False) -> None:
        from somaforge_core.trajectory_refinement import write_json
        path = Path(self.output_path + ".json")
        staged = path.with_suffix(".json.writing")
        write_json(staged, dict(schema="somaforge_chunked_recording_v1", complete=complete,
                   frames=self._chunk_start, metadata=self._metadata, chunks=self._chunks))
        staged.replace(path)

    def _flush_chunk(self) -> None:
        from somaforge_core.trajectory_refinement import fingerprint
        arrays = {}
        for name, values in self._buffers.items():
            if values:
                arrays[name] = (stack_contact_channel(values) if
                    name.startswith(("solver_contact_", "raw_contact_")) and not name.endswith("_count")
                    else np.stack(values))
        if not arrays:
            return
        lengths = {len(arrays[key]) for key in ("root_pos", "motion_id", "motion_time_step", "solver_contact_count")}
        if lengths != {self._step_count - self._chunk_start}:
            raise ValueError("Recording channels do not share the chunk clock")
        directory = Path(self.output_path + ".parts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"chunk_{len(self._chunks):05d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        self._chunks.append(dict(**fingerprint(path), start=self._chunk_start, end=self._step_count))
        self._chunk_start = self._step_count
        for values in self._buffers.values():
            values.clear()
        self._publish_chunks()
        if len(self._chunks) == 1:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        logger.info(f"Recording chunk saved: frames={self._chunk_start}, chunks={len(self._chunks)}")

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator

        self._metadata["dt"] = float(env.dt)
        self._metadata["fps"] = round(1.0 / float(env.dt))
        self._metadata["sim_dt"] = float(env.sim_dt)
        self._metadata["sim_fps"] = round(1.0 / float(env.sim_dt))
        self._metadata["control_decimation"] = env.simulator.simulator_config.sim.control_decimation
        self._metadata["contact_sensor_history_length"] = int(
            env.simulator.simulator_config.contact_sensor_history_length
        )
        self._metadata["contact_force_history_order"] = "latest_first"
        self._metadata["raw_contact_sample"] = "latest_physics_step"
        self._metadata["contact_force_raw_contact_time_aligned"] = True
        self._metadata["env_id"] = self.env_id
        self._metadata["num_envs"] = int(env.num_envs)
        simulator_cfg = env.simulator.simulator_config
        mjwarp_cfg = simulator_cfg.mujoco_warp
        self._metadata["robot_asset"] = canonical_g1_asset_metadata(
            collision_profile=getattr(env.robot_config.asset, "collision_profile", "legacy_v1"))
        self._metadata["contact_source_backend"] = NEWTON_CONTACT_BACKEND
        self._metadata["contact_collision_pipeline"] = NEWTON_COLLISION_PIPELINE
        self._metadata["contact_use_mujoco_contacts"] = False
        self._metadata["contact_force_semantics"] = "environment_on_robot_world_n"
        self._metadata["newton_solver_config"] = {
            "solver": "mjwarp",
            "collision_pipeline": NEWTON_COLLISION_PIPELINE,
            "use_mujoco_contacts": False,
            "nconmax_per_env": int(mjwarp_cfg.nconmax_per_env),
            "njmax_per_env": int(mjwarp_cfg.njmax_per_env or max(mjwarp_cfg.nconmax_per_env * 16, 2048)),
            "substeps": int(simulator_cfg.sim.substeps),
            "sim_dt": float(env.sim_dt),
            "control_decimation": int(simulator_cfg.sim.control_decimation),
        }
        if self.env_id < 0:
            self._metadata["env_done_steps"] = [None for _ in range(env.num_envs)]
            self._metadata["env_done_is_timeout"] = [False for _ in range(env.num_envs)]
            self._metadata["env_done_terms"] = [[] for _ in range(env.num_envs)]
        if hasattr(sim, "dof_names"):
            self._metadata["dof_names"] = list(sim.dof_names)
        if hasattr(sim, "body_names"):
            self._metadata["body_names"] = list(sim.body_names)
        contact_sensor = getattr(sim, "contact_sensor", None)
        if contact_sensor is not None and hasattr(contact_sensor, "body_names"):
            self._metadata["contact_sensor_body_names"] = list(contact_sensor.body_names)
        if self.profile == "full":
            try:
                from isaaclab_newton.physics.newton_manager import NewtonManager

                model = NewtonManager.get_model()
                if model is not None:
                    from somaforge_core.newton_runtime_compat import runtime_provenance
                    from scripts.serve_newton_contact_queries import inspect_model, native_binding
                    self._metadata['newton_runtime'] = runtime_provenance()
                    self._metadata['newton_scene_binding'] = native_binding(model,inspect_model(model))
                    self._metadata["newton_body_labels"] = [str(x) for x in getattr(model, "body_label", [])]
                    self._metadata["newton_shape_labels"] = [str(x) for x in getattr(model, "shape_label", [])]
                    shape_body = getattr(model, "shape_body", None)
                    if shape_body is not None:
                        self._metadata["newton_shape_body"] = np.asarray(
                            shape_body.numpy(), dtype=np.int32
                        ).tolist()
            except Exception as exc:
                self._metadata['newton_scene_binding_status'] = dict(status='unknown',reason=str(exc))
                logger.debug(f"EvalRecordingCallback: Newton contact metadata unavailable: {exc}")

        # Static robot properties
        robot_cfg = env.robot_config
        self._metadata["effort_limits"] = list(robot_cfg.dof_effort_limit_list)
        self._metadata["dof_pos_lower_limits"] = list(robot_cfg.dof_pos_lower_limit_list)
        self._metadata["dof_pos_upper_limits"] = list(robot_cfg.dof_pos_upper_limit_list)
        self._metadata["velocity_limits"] = list(robot_cfg.dof_vel_limit_list)
        asset_cfg = robot_cfg.asset
        self._metadata["urdf_path"] = str(Path(asset_cfg.asset_root) / asset_cfg.urdf_file)

        base_channel_names = [
            "dof_pos",
            "dof_vel",
            "root_pos",
            "root_quat_xyzw",
            "root_lin_vel",
            "root_ang_vel",
            "body_pos_w",
            "body_quat_xyzw",
            "motion_id",
            "motion_time_step",
            "terminated",
            "timeout",
        ]
        full_channel_names = [
            "dof_pos_target",
            "torques",
            "torques_substep",
            "dof_pos_substep",
            "dof_vel_substep",
            "actions",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "contact_forces",
            "contact_forces_history",
            "contact_sensor_forces",
            "contact_sensor_forces_history",
            "raw_contact_count",
            "raw_contact_shape0",
            "raw_contact_shape1",
            "raw_contact_body0",
            "raw_contact_body1",
            "raw_contact_point0_w",
            "raw_contact_point1_w",
            "raw_contact_normal_w",
            "raw_contact_force_w",
            "commanded_velocity",
        ]
        channel_names = base_channel_names + (
            full_channel_names if self.profile == "full" else []
        )
        for name in channel_names:
            self._buffers[name] = []
        if self.config.chunk_frames:
            for name in ("env_origins", "motion_start_step", "solver_contact_task_group",
                         "solver_contact_task_surface", "solver_contact_task_eligible"):
                self._buffers[name] = []

        if self.profile == "full":
            from holosoma.simulator.isaaclab3_newton.contact_snapshot import FIELDS
            from somaforge_core.newton_support import SUPPORT_FIELDS
            from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
            self._metadata['support_assessment_schema']=SUPPORT_ASSESSMENT_SCHEMA
            self._metadata['support_judgment_contract']=dict(
                activation='actual constraint type, includemargin and allocation; shared terrain face selection',
                load='same-solve per-constraint wrench; zero is unloaded, missing is unknown',
                motion='relative tangent velocity at actual loaded application points; sliding remains load-bearing',
                role='declared keep intent is separate from observed load and slip',
                scope='last physics solve per recorded interval; post-integration pose stored separately')
            for field in (*FIELDS,*SUPPORT_FIELDS):
                self._buffers[f"solver_contact_{field}"] = []
            for field in ('contact_activated','load_known','load_bearing','load_state',
                          'normal_force_n','loaded_motion_known','rms_tangent_speed_m_s','slip_state'):
                self._buffers[f'observed_support_{field}'] = []

        self._metadata["recording_profile"] = self.profile
        self._metadata["record_initial_state_requested"] = bool(self.config.record_initial_state)
        self._metadata["record_initial_state_effective"] = self._records_initial_state()
        logger.info(
            "EvalRecordingCallback: recording "
            f"env_id={self.env_id}, profile={self.profile}, output={self.output_path}"
        )

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        return actor_state

    def on_pre_eval_reset_bootstrap(self, actor_state: dict) -> dict:
        if self.env_id >= 0 and self._stopped_after_done:
            return actor_state
        if self._records_initial_state() and not self._initial_state_recorded:
            self._record_step(actor_state)
            self._initial_state_recorded = True
        return actor_state

    def on_post_eval_reset(self, actor_state: dict) -> dict:
        if self.env_id >= 0 and self._stopped_after_done:
            return actor_state
        if not self._records_initial_state():
            return actor_state
        if not self._initial_state_recorded:
            self._record_step(actor_state)
            self._initial_state_recorded = True
            return actor_state
        if not self._bootstrap_state_recorded:
            self._record_step(actor_state)
            self._bootstrap_state_recorded = True
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        self._current_actor_state = actor_state
        if self.env_id < 0:
            self._update_all_env_done_metadata(actor_state)
            if self.stop_when_done and self._all_envs_done():
                self._stopped_after_done = True
                return actor_state
            self._record_step(actor_state)
            return actor_state
        if self._stopped_after_done:
            return actor_state
        if self.stop_when_done and self._actor_state_done(actor_state):
            self._metadata["episode_done"] = True
            self._metadata["done_step"] = int(self._step_count)
            self._metadata["done_is_timeout"] = self._actor_state_timeout(actor_state)
            self._metadata["done_terms"] = self._active_done_terms()
            self._stopped_after_done = True
            return actor_state
        self._record_step(actor_state)
        return actor_state

    def _update_all_env_done_metadata(self, actor_state: dict) -> None:
        dones = actor_state.get("dones")
        if dones is None:
            return
        try:
            done_values = dones.detach().cpu().numpy().astype(bool).reshape(-1)
        except AttributeError:
            done_values = np.asarray(dones, dtype=bool).reshape(-1)
        done_steps = self._metadata.setdefault("env_done_steps", [None for _ in range(done_values.size)])
        done_timeouts = self._metadata.setdefault("env_done_is_timeout", [False for _ in range(done_values.size)])
        done_terms = self._metadata.setdefault("env_done_terms", [[] for _ in range(done_values.size)])
        for env_id, is_done in enumerate(done_values.tolist()):
            if not is_done or done_steps[env_id] is not None:
                continue
            done_steps[env_id] = int(self._step_count)
            done_timeouts[env_id] = self._actor_state_timeout_for_env(actor_state, env_id)
            done_terms[env_id] = self._active_done_terms_for_env(env_id)

    def _all_envs_done(self) -> bool:
        done_steps = self._metadata.get("env_done_steps")
        return isinstance(done_steps, list) and bool(done_steps) and all(step is not None for step in done_steps)

    def _actor_state_done(self, actor_state: dict) -> bool:
        dones = actor_state.get("dones")
        if dones is None:
            return False
        try:
            return bool(dones[self.env_id].detach().cpu().item())
        except (AttributeError, IndexError, TypeError):
            return bool(dones[self.env_id])

    def _actor_state_timeout(self, actor_state: dict) -> bool:
        return self._actor_state_timeout_for_env(actor_state, self.env_id)

    def _actor_state_timeout_for_env(self, actor_state: dict, env_id: int) -> bool:
        extras = actor_state.get("extras")
        if not isinstance(extras, dict):
            return False
        time_outs = extras.get("time_outs")
        if time_outs is None:
            return False
        try:
            return bool(time_outs[env_id].detach().cpu().item())
        except (AttributeError, IndexError, TypeError):
            return bool(time_outs[env_id])

    def _active_done_terms(self) -> list[str]:
        return self._active_done_terms_for_env(self.env_id)

    def _active_done_terms_for_env(self, env_id: int) -> list[str]:
        actor_state = getattr(self, "_current_actor_state", None)
        if isinstance(actor_state, dict):
            extras = actor_state.get("extras")
            if isinstance(extras, dict):
                active_terms = self._active_done_terms_from_mapping(extras.get("term_dones"), env_id)
                if active_terms:
                    return active_terms

        env = self._get_env()
        termination_manager = getattr(env, "termination_manager", None)
        term_dones = getattr(termination_manager, "term_dones", None)
        return self._active_done_terms_from_mapping(term_dones, env_id)

    def _active_done_terms_from_mapping(self, term_dones: Any, env_id: int) -> list[str]:
        if not isinstance(term_dones, dict):
            return []
        active_terms: list[str] = []
        for name, values in term_dones.items():
            try:
                is_active = bool(values[env_id].detach().cpu().item())
            except (AttributeError, IndexError, TypeError):
                is_active = bool(values[env_id])
            if is_active:
                active_terms.append(str(name))
        return active_terms

    def _record_step(self, actor_state: dict) -> None:
        env = self._get_env()
        sim = env.simulator
        eid = slice(None) if self.env_id < 0 else self.env_id

        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().copy()

        self._buffers["dof_pos"].append(_to_np(sim.dof_pos[eid]))  # post_eval_env_step, so after 4 decimation
        self._buffers["dof_vel"].append(_to_np(sim.dof_vel[eid]))
        if self.profile == "full":
            self._buffers["torques"].append(
                _to_np(self._extract_torques(env, eid))
            )  # pre_eval_env_step, so the torques is the last decimation

        # robot_root_states: [num_envs, 13] = pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)
        root = sim.robot_root_states[eid]
        if self.env_id < 0:
            self._buffers["root_pos"].append(_to_np(root[:, :3]))
            self._buffers["root_quat_xyzw"].append(_to_np(root[:, 3:7]))
            self._buffers["root_lin_vel"].append(_to_np(root[:, 7:10]))
            self._buffers["root_ang_vel"].append(_to_np(root[:, 10:13]))
        else:
            self._buffers["root_pos"].append(_to_np(root[:3]))
            self._buffers["root_quat_xyzw"].append(_to_np(root[3:7]))
            self._buffers["root_lin_vel"].append(_to_np(root[7:10]))
            self._buffers["root_ang_vel"].append(_to_np(root[10:13]))

        self._buffers["body_pos_w"].append(_to_np(sim._rigid_body_pos[eid]))
        self._buffers["body_quat_xyzw"].append(_to_np(sim._rigid_body_rot[eid]))
        self._buffers["body_lin_vel_w"].append(_to_np(sim._rigid_body_vel[eid]))
        self._buffers["body_ang_vel_w"].append(_to_np(sim._rigid_body_ang_vel[eid]))
        if hasattr(sim, "contact_forces"):
            self._buffers["contact_forces"].append(_to_np(sim.contact_forces[eid]))
        if hasattr(sim, "contact_forces_history"):
            self._buffers["contact_forces_history"].append(_to_np(sim.contact_forces_history[eid]))
        contact_sensor = getattr(sim, "contact_sensor", None)
        contact_sensor_data = getattr(contact_sensor, "data", None)
        if contact_sensor_data is not None and hasattr(contact_sensor_data, "net_forces_w"):
            self._buffers["contact_sensor_forces"].append(_to_np(contact_sensor_data.net_forces_w[eid]))
        if contact_sensor_data is not None and hasattr(contact_sensor_data, "net_forces_w_history"):
            self._buffers["contact_sensor_forces_history"].append(_to_np(contact_sensor_data.net_forces_w_history[eid]))
        if hasattr(sim, "get_raw_rigid_contacts"):
            if self.config.chunk_frames and self._step_count == 0:
                logger.info("Initial native recording: reading raw Newton contacts")
            try:
                raw_contacts = sim.get_raw_rigid_contacts()
            except Exception as exc:
                logger.debug(f"EvalRecordingCallback: failed to read raw rigid contacts: {exc}")
                raw_contacts = None
            if raw_contacts is not None:
                raw_contacts = compact_snapshot(raw_contacts)
                self._buffers["raw_contact_count"].append(np.asarray(raw_contacts["count"], dtype=np.int32))
                self._buffers["raw_contact_shape0"].append(np.asarray(raw_contacts["shape0"], dtype=np.int32))
                self._buffers["raw_contact_shape1"].append(np.asarray(raw_contacts["shape1"], dtype=np.int32))
                self._buffers["raw_contact_body0"].append(np.asarray(raw_contacts["body0"], dtype=np.int32))
                self._buffers["raw_contact_body1"].append(np.asarray(raw_contacts["body1"], dtype=np.int32))
                self._buffers["raw_contact_point0_w"].append(np.asarray(raw_contacts["point0_w"], dtype=np.float32))
                self._buffers["raw_contact_point1_w"].append(np.asarray(raw_contacts["point1_w"], dtype=np.float32))
                self._buffers["raw_contact_normal_w"].append(np.asarray(raw_contacts["normal_w"], dtype=np.float32))
                self._buffers["raw_contact_force_w"].append(np.asarray(raw_contacts["force_w"], dtype=np.float32))

        if self.profile == "full":
            if self.config.chunk_frames and self._step_count == 0:
                logger.info("Initial native recording: reading same-solve solver/contact load snapshot")
            if not callable(getattr(sim, "get_solver_contact_snapshot", None)):
                raise RuntimeError(
                    "Full recording requires Newton/MJWarp solver contact snapshots; "
                    "raw contacts and forces cannot replace activation labels."
                )
            snapshot = compact_snapshot(sim.get_solver_contact_snapshot())
            from somaforge_core.newton_support import SUPPORT_FIELDS
            if not all(name in snapshot for name in SUPPORT_FIELDS):
                raise RuntimeError('Full support recording requires same-solve per-contact force and velocity')
            for field, value in snapshot.items():
                self._buffers[f"solver_contact_{field}"].append(np.asarray(value))
            from somaforge_core.newton_support import observed_support, select_support_contacts
            if 'newton_scene_binding' not in self._metadata or 'newton_body_labels' not in self._metadata:
                raise RuntimeError('Full support recording requires actual Newton scene/body mapping')
            selection = None
            if self.config.chunk_frames:
                if self._step_count == 0:
                    logger.info("Initial native recording: resolving actual contact primary faces")
                selection = select_support_contacts(snapshot, self._metadata['newton_scene_binding'],
                                                    self._metadata['newton_body_labels'])
                for key in ("groups", "surfaces", "eligible"):
                    output = {"groups": "group", "surfaces": "surface", "eligible": "eligible"}[key]
                    self._buffers["solver_contact_task_" + output].append(selection[key].copy())
            support = observed_support(snapshot, self._metadata['newton_scene_binding'],
                                       self._metadata['newton_body_labels'], env.num_envs, selection=selection)
            for field, value in support.items():
                self._buffers[f'observed_support_{field}'].append(np.asarray(value[eid]))
            self._metadata["solver_contact_semantics"] = {
                "schema": "newton_mjwarp_constraint_snapshot_v1",
                "activation": "CONSTRAINT type and dist < includemargin",
                "allocation": "active and every expected efc row allocated (constraint_rows)",
                "indexing": "solver contacts; not raw Newton contact indices",
                "time": "last solver constraint evaluation; motion poses may be post-integration",
                "force_is_separate": True,
                "storage": "valid rows retained in original order; padded to observed maximum count, not solver capacity",
            }
            from somaforge_core.newton_support import SUPPORT_SCHEMA
            self._metadata['solver_support_semantics'] = dict(
                schema=SUPPORT_SCHEMA,
                force='MJWarp contact_force from actual efc rows, world force/torque on contact body1',
                position='solver contact position: constraint force application point',
                local_position='application point in actual mapped Newton body frame, from retained MJWarp xpos/xmat at constraint evaluation; never post-integration pose',
                velocity='resolved qvel through retained cdof/body ancestry at constraint-evaluation geometry; cvel incoming velocity recorded separately',
                sampling='same last constraint solve, resolved velocity at pre-integration contact geometry; separate from post-integration motion pose',
                activation='unchanged constraint type, dist < includemargin and allocation',
                zero_load='unknown support location; never an automatic static-support certificate')

        self._record_motion_and_termination(env, eid, _to_np)
        if self.config.chunk_frames:
            self._buffers["env_origins"].append(_to_np(env.terrain_manager.get_state("locomotion_terrain").env_origins[eid]))
            motion_command = env.command_manager.get_state("motion_command")
            self._buffers["motion_start_step"].append(_to_np(motion_command.motion.motion_start_idx[motion_command.motion_ids[eid]]))

        # substep tensors: [decimation, num_dof] — one row per physics sub-step
        torques_substep, dof_pos_substep, dof_vel_substep = self._extract_substep_data(env, eid)
        self._buffers["torques_substep"].append(_to_np(torques_substep))
        self._buffers["dof_pos_substep"].append(_to_np(dof_pos_substep))
        self._buffers["dof_vel_substep"].append(_to_np(dof_vel_substep))

        if "actions" in actor_state and actor_state["actions"] is not None:
            self._buffers["actions"].append(_to_np(actor_state["actions"][eid]))

        # Record desired target joint positions (PD setpoint)
        self._buffers["dof_pos_target"].append(_to_np(self._extract_dof_pos_target(env, eid)))

        # Record commanded velocity [lin_vel_x, lin_vel_y, ang_vel_yaw]
        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                self._buffers["commanded_velocity"].append(_to_np(env.command_manager.commands[eid]))
            except (AttributeError, IndexError):
                pass

        self._step_count += 1
        if self.config.chunk_frames and self._step_count - self._chunk_start >= self.config.chunk_frames:
            self._flush_chunk()


    def _record_motion_and_termination(self, env: Any, eid: int | slice, to_np: Any) -> None:
        if hasattr(env, "command_manager") and env.command_manager is not None:
            try:
                motion_command = env.command_manager.get_state("motion_command")
                self._buffers["motion_id"].append(to_np(motion_command.motion_ids[eid]))
                self._buffers["motion_time_step"].append(to_np(motion_command.time_steps[eid]))
            except (AttributeError, KeyError, IndexError):
                pass
        if hasattr(env, "termination_manager") and env.termination_manager is not None:
            try:
                self._buffers["terminated"].append(to_np(env.termination_manager.terminated[eid]))
                self._buffers["timeout"].append(to_np(env.termination_manager.time_outs[eid]))
            except (AttributeError, IndexError):
                pass

    def _extract_dof_pos_target(self, env: Any, env_id: int | slice) -> torch.Tensor:
        """Extract desired target joint positions from the action manager's joint control term.

        The PD target is: actions_after_delay * action_scales + default_dof_pos.
        Returns shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "_actions_after_delay") and hasattr(term, "action_scales"):
                return term._actions_after_delay[env_id] * term.action_scales + env.default_dof_pos[env_id]
        raise RuntimeError("No action term with _actions_after_delay found")

    def _extract_torques(self, env: Any, env_id: int | slice) -> torch.Tensor:
        """Extract torques from the action manager's joint control term.

        Returns torques, shape [num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques"):
                return term.torques[env_id]
        raise RuntimeError("No action term with torques found")

    def _extract_substep_data(self, env: Any, env_id: int | slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract sub-step torques, dof_pos, and dof_vel from the action manager's joint control term.

        Returns (torques_substep, dof_pos_substep, dof_vel_substep), each shape [decimation, num_dof].
        """
        for _term_name, term in env.action_manager.iter_terms():
            if hasattr(term, "torques_substep"):
                return term.torques_substep[env_id], term.dof_pos_substep[env_id], term.dof_vel_substep[env_id]
        raise RuntimeError("No action term with torques_substep found")

    def on_post_evaluate_policy(self) -> None:
        self._save()
