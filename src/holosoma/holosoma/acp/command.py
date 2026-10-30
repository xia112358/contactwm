"""Reference contacts and the paper's backward state-initialization curriculum."""
from __future__ import annotations

import json

import numpy as np
import torch

from holosoma.acp.labels import PARTS, REFERENCE_SCHEMA, hindsight_labels
from holosoma.managers.command.terms.wbt import MotionCommand
from somaforge_core.contact_face_selection import FACE_POLICY
from somaforge_core.robot_assets import decode_robot_asset_json


class ACPCommand(MotionCommand):
    def __init__(self, cfg, env):
        self.acp_settings = cfg.params["acp"]
        self.csi_phase = None
        super().__init__(cfg, env)

    def setup(self):
        super().setup()
        targets, occupancy, known, active, position, reference_known = [], [], [], [], [], []
        for path in self.motion.motion_files:
            with np.load(path, allow_pickle=False) as data:
                decode_robot_asset_json(data["robot_asset_json"], context=str(path))
                required = {"observed_contact_active", "observed_contact_force_w", "observed_contact_position_w",
                            "observed_contact_known", "observed_contact_provenance_json"}
                if not required.issubset(data.files):
                    raise ValueError(f"{path}: record an ACP nine-body native reference first; missing {required-set(data.files)}")
                meta = json.loads(str(data["observed_contact_provenance_json"].item()))
                if (meta.get("schema") != REFERENCE_SCHEMA or tuple(meta.get("parts", ())) != PARTS
                        or meta.get("face_selection") != FACE_POLICY or not meta.get("complete")
                        or meta.get("sampling") != "last_native_solve_before_policy_observation"
                        or meta.get("strict_pose_contact_same_state") is not False
                        or meta.get("activation") != "CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc"):
                    raise ValueError(f"{path}: incompatible or incomplete ACP reference")
                if not np.isclose(float(np.asarray(data["fps"]).item()), 50.0):
                    raise ValueError("ACP reference must be sampled at 50 Hz")
                if len(data["observed_contact_active"]) != len(data["joint_pos"]):
                    raise ValueError("ACP native reference and motion frame counts differ")
                labels = hindsight_labels(data["observed_contact_active"], data["observed_contact_force_w"],
                                           data["observed_contact_position_w"], data["observed_contact_known"])
                targets.append(labels["target_w"]); occupancy.append(labels["occupancy"]); known.append(labels["known"])
                active.append(data["observed_contact_active"]); position.append(data["observed_contact_position_w"])
                reference_known.append(data['observed_contact_known'])
        def tensor(rows, dtype):
            return torch.as_tensor(np.concatenate(rows), dtype=dtype, device=self.device)
        self.acp_target_w = tensor(targets, torch.float32)
        self.acp_target_occupancy = tensor(occupancy, torch.bool)
        self.acp_target_known = tensor(known, torch.bool)
        self.acp_reference_active = tensor(active, torch.bool)
        self.acp_reference_position = tensor(position, torch.float32)
        self.acp_reference_known = tensor(reference_known, torch.bool)
        initial = 0.9 if self.acp_settings["stage"] == 1 else 0.0
        self.csi_phase = torch.full((self.num_envs,), initial, device=self.device)
        self.acp_goal_lookahead = torch.full((self.num_envs,), 50, dtype=torch.long, device=self.device)

    def sample_reset_phase(self, env_ids):
        if self.csi_phase is None or self.acp_settings["stage"] != 1:
            return super().sample_reset_phase(env_ids)
        if self._env.is_evaluating:
            return torch.zeros(len(env_ids), device=self.device)
        return self.csi_phase[env_ids].clone()

    def reset(self, env_ids):
        ids = self._ensure_index_tensor(env_ids)
        if self.csi_phase is not None and self.acp_settings["stage"] == 1 and len(ids):
            mids = self.motion_ids[ids]
            starts, ends = self.motion.motion_start_idx[mids], self.motion.motion_end_idx[mids]
            completed = ((self.time_steps[ids] - starts).float() / (ends-starts-1).clamp_min(1)) >= 0.99
            valid = self._env._pending_episode_lengths[ids] > 0
            terminated = self._env.termination_manager.terminated[ids]
            if not self._env.is_evaluating:
                selected = ids[valid]
                success = (completed & ~terminated)[valid]
                delta = torch.where(success, -0.05, 0.05)
                self.csi_phase[selected] = (self.csi_phase[selected] + delta).clamp(0, 0.99)
        super().reset(env_ids)
        from holosoma.acp.randomization import dynamics_enabled
        if dynamics_enabled(self._env):
            self._env.simulator.robot_root_states[ids, 2] += torch.rand(len(ids), device=self.device)*0.05
        if hasattr(self, "acp_goal_lookahead"):
            self.acp_goal_lookahead[ids] = torch.randint(25, 76, (len(ids),), device=self.device)

    def get_checkpoint_state(self):
        return {**super().get_checkpoint_state(), "acp_csi_phase": self.csi_phase.clone(),
                "acp_csi_version": 2}

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        if state and "acp_csi_phase" in state:
            if state.get("acp_csi_version") != 2:
                raise ValueError("Legacy ACP CSI used motion-level phase bounds; start an aligned Stage-1 run")
            phases = state["acp_csi_phase"].to(self.device)
            if phases.shape != self.csi_phase.shape:
                raise ValueError("ACP curriculum environment count changed")
            if not torch.isfinite(phases).all() or ((phases < 0) | (phases > 0.99)).any():
                raise ValueError("Invalid saved ACP curriculum phases")
            self.csi_phase.copy_(phases)
