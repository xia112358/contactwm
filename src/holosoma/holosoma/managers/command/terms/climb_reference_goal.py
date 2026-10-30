"""Reference-and-goal command for the trimmed climb00 contact experiment.

The actor receives only a task goal.  The demonstration trajectory remains
available to the asymmetric critic and imitation reward, following the
multi-task construction in arXiv:2602.20375v1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul, quat_rotate_inverse

CONTACT_ENDPOINT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "right_knee_link",
)
CONTACT_PART_ORDER = ("LF", "RF", "LH", "RH", "LK", "RK")
SOURCE_CONTACT_PART_ORDER = ("LHEE", "LTOE", "RHEE", "RTOE", "LH", "RH", "LK", "RK")


def collapse_contact_mask(value: np.ndarray) -> np.ndarray:
    """Collapse the canonical eight force parts to six task endpoints."""
    value = np.asarray(value, dtype=bool)
    if value.ndim != 2 or value.shape[1] != 8:
        raise ValueError(f"Expected contact mask [T, 8], got {value.shape}")
    return np.stack(
        (
            value[:, 0] | value[:, 1],
            value[:, 2] | value[:, 3],
            value[:, 4],
            value[:, 5],
            value[:, 6],
            value[:, 7],
        ),
        axis=-1,
    )


def debounce_contact_mask(value: np.ndarray, stable_frames: int = 3) -> np.ndarray:
    """Accept a contact transition only after ``stable_frames`` observations."""
    value = np.asarray(value, dtype=bool)
    output = np.empty_like(value)
    for channel in range(value.shape[1]):
        current = bool(value[0, channel])
        output[0, channel] = current
        candidate: int | None = None
        for frame in range(1, len(value)):
            state = bool(value[frame, channel])
            if state == current:
                if candidate is not None:
                    output[candidate:frame, channel] = current
                    candidate = None
                output[frame, channel] = current
                continue
            candidate = frame if candidate is None else candidate
            output[frame, channel] = current
            if frame - candidate + 1 >= stable_frames:
                output[candidate : frame + 1, channel] = state
                current = state
                candidate = None
        if candidate is not None:
            output[candidate:, channel] = current
    return output


def grouped_touchdowns(contact: np.ndarray, merge_gap: int = 4) -> list[tuple[int, np.ndarray]]:
    """Return grouped touchdown frame and six-bit touchdown mask."""
    contact = np.asarray(contact, dtype=bool)
    rising = contact[1:] & ~contact[:-1]
    frames = np.flatnonzero(rising.any(axis=1)) + 1
    groups: list[list[int]] = []
    for frame_value in frames:
        frame = int(frame_value)
        if groups and frame - groups[-1][-1] <= merge_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return [
        (group[-1], np.any(rising[np.asarray(group, dtype=np.int64) - 1], axis=0))
        for group in groups
    ]


def load_trimmed_contact_events(
    manifest_path: str,
    *,
    expected_event_count: int = 24,
    stable_frames: int = 3,
    merge_gap: int = 4,
    minimum_event_frames: int = 6,
) -> dict[str, np.ndarray]:
    """Load and validate the contact-event table used by the experiment.

    This is intentionally a pure CPU helper so the exact 207x24 contract can
    be unit-tested without initializing Isaac Sim or CUDA.
    """
    manifest = load_motion_terrain_manifest(manifest_path)
    entries = list(manifest["motion_files"])
    event_frames: list[np.ndarray] = []
    touchdown_masks: list[np.ndarray] = []
    end_contact_masks: list[np.ndarray] = []
    frame_counts: list[int] = []

    for entry in entries:
        plan_path = Path(str(entry["edit_plan_file"]))
        with plan_path.open("r", encoding="utf-8") as stream:
            plan = json.load(stream)
        source_path = Path(str(plan["source_motion_path"]))
        with np.load(source_path, allow_pickle=False) as source:
            raw = np.asarray(source["contact_force_part_mask"], dtype=bool)
            order = tuple(str(value) for value in source["contact_force_part_order"].tolist())
        if order != SOURCE_CONTACT_PART_ORDER:
            raise ValueError(f"Unexpected contact order in {source_path}: {order}")

        contact = debounce_contact_mask(collapse_contact_mask(raw), stable_frames=stable_frames)
        raw_events = grouped_touchdowns(contact, merge_gap=merge_gap)
        events: list[tuple[int, np.ndarray]] = []
        current_frame = 0
        for target_frame, touchdown_mask in raw_events:
            if target_frame - current_frame < int(minimum_event_frames):
                continue
            events.append((target_frame, touchdown_mask))
            current_frame = target_frame
        if len(events) != expected_event_count:
            raise ValueError(
                f"Expected {expected_event_count} touchdown groups for {entry['motion_file']}, got {len(events)}"
            )
        frames = np.asarray([frame for frame, _ in events], dtype=np.int64)
        if np.any(frames <= 0) or np.any(np.diff(frames) <= 0):
            raise ValueError(f"Touchdown frames must be strictly increasing: {entry['motion_file']}")
        event_frames.append(frames)
        touchdown_masks.append(np.stack([mask for _, mask in events], axis=0))
        end_contact_masks.append(contact[frames])
        frame_counts.append(int(contact.shape[0]))

    return {
        "event_frames": np.stack(event_frames, axis=0),
        "touchdown_masks": np.stack(touchdown_masks, axis=0),
        "end_contact_masks": np.stack(end_contact_masks, axis=0),
        "frame_counts": np.asarray(frame_counts, dtype=np.int64),
    }


class ClimbReferenceGoalCommand(MotionCommand):
    """Shared command for imitation and sparse-goal climb contact tasks."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.goal_mode = str(params.get("goal_mode", "contact_events"))
        if self.goal_mode not in {"contact_events", "motion_end"}:
            raise ValueError(
                "goal_mode must be either 'contact_events' or 'motion_end', "
                f"got {self.goal_mode!r}"
            )
        self.expected_event_count = int(params.get("expected_event_count", 24))
        if self.goal_mode == "motion_end":
            self.expected_event_count = 1
        self.imitation_probability_start = float(params.get("imitation_probability_start", 1.0))
        self.imitation_probability_final = float(params.get("imitation_probability_final", 0.5))
        self.imitation_probability = self.imitation_probability_start
        self.difficulty = float(params.get("initial_difficulty", 0.0))
        self.generalization_root_xy = float(params.get("generalization_root_xy", 0.4))
        self.generalization_root_roll_pitch = float(params.get("generalization_root_roll_pitch", 0.15))
        self.generalization_root_yaw = float(params.get("generalization_root_yaw", 0.8))
        self.generalization_goal_xy = float(params.get("generalization_goal_xy", 0.20))
        self.generalization_endpoint_jitter = float(params.get("generalization_endpoint_jitter", 0.08))
        self.goal_stability_tail_steps = max(int(params.get("goal_stability_tail_steps", 3)), 0)
        self.assistive_force_max = float(params.get("assistive_force_max", 350.0))
        self.assistive_beta_max = float(params.get("assistive_beta_max", 0.75))
        self.assistive_kp = float(params.get("assistive_kp", 600.0))
        self.assistive_kd = float(params.get("assistive_kd", 80.0))

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Persist the coupled task curriculum alongside motion-sampler state."""
        state = super().get_checkpoint_state()
        state["climb_reference_goal"] = {
            "difficulty": float(self.difficulty),
            "performance_ema": float(getattr(self, "curriculum_performance_ema", 0.0)),
            "has_performance": bool(getattr(self, "curriculum_has_performance", False)),
        }
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore the curriculum without treating old checkpoints as compatible policies."""
        super().load_checkpoint_state(state)
        payload = state.get("climb_reference_goal") if isinstance(state, dict) else None
        if not isinstance(payload, dict):
            return
        self.difficulty = min(max(float(payload.get("difficulty", 0.0)), 0.0), 1.0)
        self.imitation_probability = self.imitation_probability_start + self.difficulty * (
            self.imitation_probability_final - self.imitation_probability_start
        )
        self.curriculum_performance_ema = float(payload.get("performance_ema", 0.0))
        self.curriculum_has_performance = bool(payload.get("has_performance", False))
        curriculum_manager = getattr(self._env, "curriculum_manager", None)
        curriculum = (
            curriculum_manager.get_term("reference_goal_curriculum")
            if curriculum_manager is not None
            else None
        )
        if curriculum is not None:
            curriculum.performance_ema = self.curriculum_performance_ema
            curriculum._has_performance = self.curriculum_has_performance

    def setup(self) -> None:
        super().setup()
        if not self.motion_cfg.motion_manifest:
            raise ValueError("ClimbReferenceGoalCommand requires a motion_manifest")
        loaded_counts = self.motion.motion_end_idx - self.motion.motion_start_idx
        if self.goal_mode == "contact_events":
            table = load_trimmed_contact_events(
                self.motion_cfg.motion_manifest,
                expected_event_count=self.expected_event_count,
            )
            frame_counts = torch.as_tensor(table["frame_counts"], dtype=torch.long, device=self.device)
            if not torch.equal(frame_counts, loaded_counts):
                raise ValueError(
                    "Contact-source frame counts do not match loaded trimmed motions: "
                    f"contact={frame_counts.tolist()[:4]}, motion={loaded_counts.tolist()[:4]}"
                )
            self.event_frames = torch.as_tensor(
                table["event_frames"], dtype=torch.long, device=self.device
            )
            self.event_touchdown_masks = torch.as_tensor(
                table["touchdown_masks"], dtype=torch.bool, device=self.device
            )
            self.event_end_contact_masks = torch.as_tensor(
                table["end_contact_masks"], dtype=torch.bool, device=self.device
            )
        else:
            # A single behavior is conditioned only on its final root goal.
            # RSI and dense imitation still use every frame of the full motion.
            self.event_frames = (loaded_counts - 2).clamp_min(1)[:, None]
            mask_shape = (self.motion.num_motions, 1, len(CONTACT_PART_ORDER))
            self.event_touchdown_masks = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            self.event_end_contact_masks = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
        body_names = list(self._env.simulator._body_list)  # type: ignore[attr-defined]
        self.goal_endpoint_body_indexes = torch.as_tensor(
            [body_names.index(name) for name in CONTACT_ENDPOINT_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        tracked_names = list(self.motion_cfg.body_names_to_track)
        self.goal_endpoint_track_indexes = torch.as_tensor(
            [tracked_names.index(name) for name in CONTACT_ENDPOINT_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )

        self.is_imitation = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.event_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.deadline_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_touchdown_mask = torch.zeros(
            self.num_envs, len(CONTACT_PART_ORDER), dtype=torch.bool, device=self.device
        )
        self.goal_contact_mask = torch.zeros_like(self.goal_touchdown_mask)
        self.goal_ref_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.goal_ref_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.goal_ref_quat_w[:, 3] = 1.0
        self.goal_endpoint_pos_w = torch.zeros(
            self.num_envs, len(CONTACT_PART_ORDER), 3, dtype=torch.float32, device=self.device
        )
        self.assistive_wrench_w = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        body_id = int(self._env.simulator.body_ids[self.ref_body_index])
        self._assistive_body_ids = torch.tensor([body_id], dtype=torch.long, device=self.device)
        self._assistive_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._assistive_zero_torques = torch.zeros(
            self.num_envs, 1, 3, dtype=torch.float32, device=self.device
        )
        logger.info(
            "ClimbReferenceGoalCommand: loaded "
            f"{self.motion.num_motions} motions x {self.expected_event_count} goals "
            f"(goal_mode={self.goal_mode})"
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        super().reset(env_ids)

        motion_ids = self.motion_ids[env_ids]
        starts = self.motion.motion_start_idx[motion_ids]
        local_steps = self.time_steps[env_ids] - starts
        frames = self.event_frames[motion_ids]
        # Select the next touchdown strictly after the RSI frame.  The tail is
        # assigned to the last task and moved just before its touchdown.
        event_index = torch.searchsorted(frames, local_steps[:, None], right=True).squeeze(-1)
        after_last = event_index >= self.expected_event_count
        event_index = event_index.clamp(max=self.expected_event_count - 1)
        if torch.any(after_last):
            tail_env_ids = env_ids[after_last]
            tail_motion_ids = motion_ids[after_last]
            tail_local = self.event_frames[tail_motion_ids, -1] - 1
            self.time_steps[tail_env_ids] = self.motion.motion_start_idx[tail_motion_ids] + tail_local
            self._write_exact_reference_state(tail_env_ids)

        self.event_index[env_ids] = event_index
        self.is_imitation[env_ids] = (
            torch.rand(env_ids.numel(), device=self.device) < float(self.imitation_probability)
        )
        self._assign_goal(env_ids)
        self._apply_generalization_reset_noise(env_ids)
        self.assistive_wrench_w[env_ids] = 0.0

    def step(self) -> None:
        super().step()
        self._apply_assistive_force()

    def _write_exact_reference_state(self, env_ids: torch.Tensor) -> None:
        self._env.simulator.dof_pos[env_ids] = self.joint_pos[env_ids]
        self._env.simulator.dof_vel[env_ids] = self.reset_joint_vel[env_ids]
        self._env.simulator.robot_root_states[env_ids, :3] = self.root_pos_w[env_ids]
        self._env.simulator.robot_root_states[env_ids, 3:7] = self.root_quat_w[env_ids]
        self._env.simulator.robot_root_states[env_ids, 7:10] = self.reset_root_lin_vel_w[env_ids]
        self._env.simulator.robot_root_states[env_ids, 10:13] = self.reset_root_ang_vel_w[env_ids]

    def _assign_goal(self, env_ids: torch.Tensor) -> None:
        motion_ids = self.motion_ids[env_ids]
        event_ids = self.event_index[env_ids]
        local_goal = self.event_frames[motion_ids, event_ids]
        global_goal = self.motion.motion_start_idx[motion_ids] + local_goal
        self.goal_steps[env_ids] = global_goal
        final_event = event_ids == self.expected_event_count - 1
        motion_deadline = self.motion.motion_end_idx[motion_ids] - 2
        contact_goal_deadline = torch.minimum(
            global_goal + self.goal_stability_tail_steps,
            motion_deadline,
        )
        self.deadline_steps[env_ids] = torch.where(final_event, motion_deadline, contact_goal_deadline)
        self.goal_touchdown_mask[env_ids] = self.event_touchdown_masks[motion_ids, event_ids]
        self.goal_contact_mask[env_ids] = self.event_end_contact_masks[motion_ids, event_ids]

        raw_goal_pos = self.motion.body_pos_w[global_goal, self.ref_body_index][:, None, :]
        raw_goal_quat = self.motion.body_quat_w[global_goal, self.ref_body_index][:, None, :]
        goal_pos, goal_quat = self._localize_motion_pos_quat(
            raw_goal_pos,
            raw_goal_quat,
            env_ids=env_ids,
        )
        self.goal_ref_pos_w[env_ids] = goal_pos[:, 0]
        self.goal_ref_quat_w[env_ids] = goal_quat[:, 0]

        endpoint_pos = self.motion.body_pos_w[global_goal][:, self.goal_endpoint_body_indexes]
        endpoint_quat = self.motion.body_quat_w[global_goal][:, self.goal_endpoint_body_indexes]
        localized_endpoint_pos, _ = self._localize_motion_pos_quat(
            endpoint_pos,
            endpoint_quat,
            env_ids=env_ids,
        )
        self.goal_endpoint_pos_w[env_ids] = localized_endpoint_pos

        generalization = ~self.is_imitation[env_ids]
        if not torch.any(generalization) or self.difficulty <= 0.0:
            return
        gen_env_ids = env_ids[generalization]
        scale = float(self.difficulty)
        shared_xy = (
            torch.rand(gen_env_ids.numel(), 2, device=self.device) * 2.0 - 1.0
        ) * self.generalization_goal_xy * scale
        self.goal_ref_pos_w[gen_env_ids, :2] += shared_xy
        self.goal_endpoint_pos_w[gen_env_ids, :, :2] += shared_xy[:, None, :]
        jitter = (
            torch.rand(gen_env_ids.numel(), len(CONTACT_PART_ORDER), 2, device=self.device) * 2.0 - 1.0
        ) * self.generalization_endpoint_jitter * scale
        active = self.goal_touchdown_mask[gen_env_ids, :, None].to(jitter.dtype)
        self.goal_endpoint_pos_w[gen_env_ids, :, :2] += jitter * active

    def _apply_generalization_reset_noise(self, env_ids: torch.Tensor) -> None:
        generalization = ~self.is_imitation[env_ids]
        if not torch.any(generalization) or self.difficulty <= 0.0:
            return
        gen_env_ids = env_ids[generalization]
        scale = float(self.difficulty)
        root_state = self._env.simulator.robot_root_states
        root_state[gen_env_ids, :2] += (
            torch.rand(gen_env_ids.numel(), 2, device=self.device) * 2.0 - 1.0
        ) * self.generalization_root_xy * scale
        rpy = torch.rand(gen_env_ids.numel(), 3, device=self.device) * 2.0 - 1.0
        rpy[:, :2] *= self.generalization_root_roll_pitch * scale
        rpy[:, 2] *= self.generalization_root_yaw * scale
        delta = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        root_state[gen_env_ids, 3:7] = quat_mul(delta, root_state[gen_env_ids, 3:7], w_last=True)

    def _apply_assistive_force(self) -> None:
        """Apply the paper's annealed base assistance through IsaacLab buffers."""
        simulator = self._env.simulator
        robot = getattr(simulator, "_robot", None)
        if robot is None or not hasattr(robot, "set_external_force_and_torque"):
            raise RuntimeError("The climb reference-goal experiment requires IsaacLab external-force support")
        beta = self.assistive_beta_max * max(1.0 - float(self.difficulty), 0.0)
        force_w = self.assistive_kp * (self.ref_pos_w - self.robot_ref_pos_w)
        force_w -= self.assistive_kd * self.robot_ref_lin_vel_w
        norm = torch.linalg.vector_norm(force_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
        force_w = force_w * torch.clamp(self.assistive_force_max / norm, max=1.0) * beta
        self.assistive_wrench_w[:, :3] = force_w
        self.assistive_wrench_w[:, 3:] = 0.0

        local_force = quat_rotate_inverse(self.robot_ref_quat_w, force_w, w_last=True)
        robot.set_external_force_and_torque(
            forces=local_force[:, None, :],
            torques=self._assistive_zero_torques,
            env_ids=self._assistive_env_ids,
            body_ids=self._assistive_body_ids,
        )
