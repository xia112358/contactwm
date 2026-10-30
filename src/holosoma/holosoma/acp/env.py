"""ACP observations and episode recording on the native WBT environment."""
from __future__ import annotations

import torch

from holosoma.acp import observations as obs
from holosoma.acp.contacts import measured_contacts
from holosoma.acp.models import ContactPlanner
from holosoma.acp.replay import HindsightReplay
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


class ACPEnvironment(WholeBodyTrackingManager):
    def __init__(self, tyro_config, *, device):
        self.acp_settings = tyro_config.command.setup_terms["motion_command"].params["acp"]
        self.acp_iteration = 0
        self.acp_planner = None
        self.acp_recordings = {}
        self.acp_recording_starts = {}
        self.acp_admitted = 0
        self.acp_completed = 0
        self.acp_replay = HindsightReplay(self.acp_settings["replay_capacity"], tyro_config.training.seed)
        super().__init__(tyro_config, device=device)
        if self.acp_settings["stage"] > 1:
            self.acp_planner = ContactPlanner(self.num_dof).to(device)
        # PPO discovers observation widths before its own first reset. Put
        # articulations on their motion-matched terrain before scanning it.
        self.reset_all(step_after_reset=False)

    def _init_buffers(self):
        super()._init_buffers()
        self._holosoma_step_cache = {}
        self.acp_history = torch.zeros(self.num_envs, 5, 6+3*self.num_dof, device=self.device)
        self.acp_previous_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.acp_second_previous_actions = torch.zeros_like(self.acp_previous_actions)
        self.acp_previous_body_velocity = torch.zeros(self.num_envs, 9, 3, device=self.device)
        self.acp_body_velocity_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        from holosoma.acp.sensing import PoseHistory
        self.acp_pose_history = PoseHistory(self.num_envs, self.dt, self.device)
        self.acp_sensor_step = 0
        self._acp_step_inputs = None
        self.acp_sensor_time = 0.0
        self.acp_perception = None

    def _update_counters_each_step(self):
        super()._update_counters_each_step()
        self.acp_sensor_time += self.dt

    def _compute_observations(self):
        self._holosoma_step_cache.pop("acp_command", None)
        self.acp_history[:, :-1] = self.acp_history[:, 1:].clone()
        measured = obs.proprio(self)
        from holosoma.acp.randomization import observation_noise_enabled
        self.acp_pose_history.update(self.simulator.robot_root_states, self.acp_sensor_step)
        if observation_noise_enabled(self):
            scales = torch.tensor([0.05]*3+[0.0125]*3+[0.01]*self.num_dof
                                  +[1.5]*self.num_dof+[0.0]*self.num_dof, device=self.device)
            measured = measured + (2*torch.rand_like(measured)-1)*scales
        self.acp_history[:, -1] = measured
        super()._compute_observations()

    def _compute_final_observations(self):
        self._holosoma_step_cache.pop("acp_command", None)
        history = self.acp_history
        self.acp_history = torch.cat((history[:, 1:], obs.proprio(self)[:, None]), 1)
        try:
            return super()._compute_final_observations()
        finally:
            self.acp_history = history

    def _pre_physics_step(self, actions):
        self._acp_step_inputs = None
        obs.contact_targets(self)
        predicted = self._holosoma_step_cache["acp_command"][1]
        root = self.simulator.robot_root_states[:]
        self.acp_issued_plan_w = obs.rotate_yaw(predicted, obs.yaw(root[:, 3:7]))+root[:, None, :3]
        self.acp_issued_plan_valid = predicted.abs().sum(-1) > 0
        if self.acp_settings["stage"] > 1:
            scan, valid = obs.terrain(self)
            count = min(self.num_envs, self.acp_settings["record_envs"])
            self._acp_step_inputs = {"history": self.acp_history[:count].flatten(1).detach().cpu().numpy().copy(),
                "goal": obs.goal(self)[:count].detach().cpu().numpy().copy(),
                "terrain": scan[:count].detach().cpu().numpy().copy(), "valid": valid[:count].cpu().numpy().copy(),
                "root": root[:count, :3].cpu().numpy().copy(),
                "yaw": obs.yaw(root[:count, 3:7]).cpu().numpy().copy()}
        self.acp_sensor_step += 1
        super()._pre_physics_step(actions)

    def _compute_reward(self):
        super()._compute_reward()
        from holosoma.acp.rewards import contact_body_velocities
        self.acp_previous_body_velocity.copy_(contact_body_velocities(self))
        self.acp_body_velocity_ready.fill_(True)
        if self._acp_step_inputs is not None:
            actual = measured_contacts(self)
            count = len(self._acp_step_inputs["root"])
            arrays = {"active": actual["active"][:count].cpu().numpy(),
                      "force": actual["force_w"][:count].cpu().numpy(),
                      "position": actual["position_w"][:count].cpu().numpy(),
                      "known": actual["known"][:count].cpu().numpy()}
            cmd = obs.command(self)
            for index in range(count):
                start = int(cmd.motion.motion_start_idx[cmd.motion_ids[index]])
                end = int(cmd.motion.motion_end_idx[cmd.motion_ids[index]])
                phase = (int(cmd.time_steps[index])-start)/max(1, end-start-1)
                frames = self.acp_recordings.setdefault(index, [])
                self.acp_recording_starts.setdefault(index, phase)
                if len(frames) < 500:
                    frames.append({**{k: v[index].copy() for k, v in self._acp_step_inputs.items()},
                                   **{k: v[index].copy() for k, v in arrays.items()}})
                if self.reset_buf[index]:
                    success = not bool(self.termination_manager.terminated[index]) and phase >= 0.9
                    admitted = self.acp_replay.admit(frames, success=success,
                        initial_phase=self.acp_recording_starts[index], final_phase=phase, tile=int(cmd.motion_ids[index]))
                    self.acp_admitted += int(admitted)
                    self.acp_completed += 1
                    self.acp_recordings.pop(index, None)
                    self.acp_recording_starts.pop(index, None)
        self.acp_second_previous_actions.copy_(self.acp_previous_actions)
        self.acp_previous_actions.copy_(self.action_manager.action)
        cmd = obs.command(self)
        self.log_dict.update({"ACP/csi_phase": cmd.csi_phase.mean(),
            "ACP/replay_samples": torch.tensor(float(self.acp_replay.size), device=self.device),
            "ACP/admitted_episodes": torch.tensor(float(self.acp_admitted), device=self.device),
            "ACP/completed_episodes": torch.tensor(float(self.acp_completed), device=self.device)})

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        if getattr(self, "acp_perception", None) is not None:
            self.acp_perception.reset(env_ids)
        if hasattr(self, "acp_history"):
            self.acp_history[env_ids] = 0
            self.acp_previous_actions[env_ids] = 0
            self.acp_second_previous_actions[env_ids] = 0
            self.acp_previous_body_velocity[env_ids] = 0
            self.acp_body_velocity_ready[env_ids] = False
            self.acp_pose_history.ready[env_ids] = False
        for index in env_ids.tolist():
            self.acp_recordings.pop(index, None)
            self.acp_recording_starts.pop(index, None)


def reference_complete(env):
    cmd = obs.command(env)
    return cmd.time_steps >= cmd.motion.motion_end_idx[cmd.motion_ids]-2
