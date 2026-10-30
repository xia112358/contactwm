"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

import re
from typing import Any, List, Sequence

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import get_projected_gravity, gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    end_idx = motion_command.motion.motion_end_idx[motion_command.motion_ids]
    return motion_command.time_steps >= end_idx - 2


def start_probe_aware_timeout_exceeded(env, probe_margin_steps: int = 2, **_) -> torch.Tensor:
    """Use full motion length as the timeout horizon for start-probe environments."""
    motion_command = env.command_manager.get_state("motion_command")
    is_probe = getattr(motion_command, "_probe_env_mask", None)
    normal_timeout = env.episode_length_buf > env.max_episode_length
    if is_probe is None:
        return normal_timeout

    start_idx = motion_command.motion.motion_start_idx[motion_command.motion_ids]
    end_idx = motion_command.motion.motion_end_idx[motion_command.motion_ids]
    motion_len = (end_idx - start_idx).clamp(min=1)
    elapsed = (motion_command.time_steps - start_idx).clamp(min=0)
    probe_horizon = motion_len + int(probe_margin_steps)
    probe_timeout = elapsed > probe_horizon
    return torch.where(is_probe, probe_timeout, normal_timeout)


def base_height_below_threshold(env: WholeBodyTrackingManager, min_height: float) -> torch.Tensor:
    """Terminate only when the robot body is physically unrecoverable, not when it misses the reference."""
    return env.simulator.robot_root_states[:, 2] < min_height


def base_tilt_exceeded(env: WholeBodyTrackingManager, threshold_x: float, threshold_y: float) -> torch.Tensor:
    """Terminate when roll/pitch tilt is too large to recover."""
    projected_gravity = get_projected_gravity(env)
    return (torch.abs(projected_gravity[:, 0]) > threshold_x) | (torch.abs(projected_gravity[:, 1]) > threshold_y)


def severe_state_invalid(
    env: WholeBodyTrackingManager,
    min_height: float = 0.05,
    max_height: float = 5.0,
    max_ref_pos_error: float = 10.0,
    max_root_lin_vel: float = 50.0,
    max_root_ang_vel: float = 100.0,
    max_body_lin_vel: float = 100.0,
    max_joint_abs_vel: float = 200.0,
) -> torch.Tensor:
    """Terminate only physically invalid states, not ordinary reference-tracking drift."""
    root_states = env.simulator.robot_root_states[:, :13]
    root_nonfinite = ~torch.isfinite(root_states).all(dim=1)
    height_low = root_states[:, 2] < min_height
    height_high = root_states[:, 2] > max_height
    root_lin_norm = torch.norm(root_states[:, 7:10], dim=1)
    root_ang_norm = torch.norm(root_states[:, 10:13], dim=1)
    root_lin_fast = root_lin_norm > max_root_lin_vel
    root_ang_fast = root_ang_norm > max_root_ang_vel
    invalid = root_nonfinite | height_low | height_high | root_lin_fast | root_ang_fast

    body_nonfinite = torch.zeros_like(invalid)
    body_fast = torch.zeros_like(invalid)
    dof_nonfinite = torch.zeros_like(invalid)
    dof_fast = torch.zeros_like(invalid)
    ref_nonfinite = torch.zeros_like(invalid)
    ref_far = torch.zeros_like(invalid)

    if hasattr(env.simulator, "_rigid_body_vel"):
        body_vel = env.simulator._rigid_body_vel
        body_vel_norm = torch.norm(body_vel, dim=-1).amax(dim=1)
        body_nonfinite = ~torch.isfinite(body_vel).flatten(start_dim=1).all(dim=1)
        body_fast = body_vel_norm > max_body_lin_vel
        invalid |= body_nonfinite | body_fast
    else:
        body_vel_norm = torch.zeros_like(root_lin_norm)

    if hasattr(env.simulator, "dof_vel"):
        dof_vel = env.simulator.dof_vel
        dof_vel_abs = torch.abs(dof_vel).amax(dim=1)
        dof_nonfinite = ~torch.isfinite(dof_vel).all(dim=1)
        dof_fast = dof_vel_abs > max_joint_abs_vel
        invalid |= dof_nonfinite | dof_fast
    else:
        dof_vel_abs = torch.zeros_like(root_lin_norm)

    if env.command_manager is not None:
        motion_command = env.command_manager.get_state("motion_command")
        ref_error = torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1)
        ref_nonfinite = ~torch.isfinite(ref_error)
        ref_far = ref_error > max_ref_pos_error
        invalid |= ref_nonfinite | ref_far

    if hasattr(env, "log_dict"):
        for metric_name, metric_value in {
            "root_nonfinite": root_nonfinite,
            "height_low": height_low,
            "height_high": height_high,
            "root_lin_fast": root_lin_fast,
            "root_ang_fast": root_ang_fast,
            "body_nonfinite": body_nonfinite,
            "body_fast": body_fast,
            "dof_nonfinite": dof_nonfinite,
            "dof_fast": dof_fast,
            "ref_nonfinite": ref_nonfinite,
            "ref_far": ref_far,
            "total": invalid,
        }.items():
            env.log_dict[f"termination/severe_state_invalid/{metric_name}_rate"] = (
                metric_value.to(torch.float32).mean().detach().cpu()
            )
        env.log_dict["termination/severe_state_invalid/root_lin_vel_abs_max"] = torch.nan_to_num(
            root_lin_norm.detach(), nan=0.0, posinf=1.0e6, neginf=1.0e6
        ).max().detach().cpu()
        env.log_dict["termination/severe_state_invalid/root_ang_vel_abs_max"] = torch.nan_to_num(
            root_ang_norm.detach(), nan=0.0, posinf=1.0e6, neginf=1.0e6
        ).max().detach().cpu()
        env.log_dict["termination/severe_state_invalid/body_vel_abs_max"] = torch.nan_to_num(
            body_vel_norm.detach(), nan=0.0, posinf=1.0e6, neginf=1.0e6
        ).max().detach().cpu()
        env.log_dict["termination/severe_state_invalid/dof_vel_abs_max"] = torch.nan_to_num(
            dof_vel_abs.detach(), nan=0.0, posinf=1.0e6, neginf=1.0e6
        ).max().detach().cpu()

    return invalid


class UndesiredContacts(TerminationTermBase):
    """Terminate when disallowed robot bodies make contact."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        pattern = cfg.params.get("undesired_contacts_body_names", "")
        body_names = list(self.env.simulator.body_names)  # type: ignore[attr-defined]
        undesired_body_names = [body_name for body_name in body_names if re.match(pattern, body_name)]
        self.undesired_body_indices = torch.tensor(
            [body_names.index(name) for name in undesired_body_names],
            dtype=torch.long,
            device=self.env.device,
        )
        self.threshold = float(cfg.params.get("threshold", 1.0))

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        if self.undesired_body_indices.numel() == 0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        contact_forces = env.simulator.contact_forces_history
        force_norm = torch.norm(contact_forces[:, :, self.undesired_body_indices], dim=-1)
        return torch.max(force_norm, dim=1)[0].gt(self.threshold).any(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


DEFAULT_A2A_PART_BODY_NAMES = (
    ("left_ankle_roll_link",),
    ("right_ankle_roll_link",),
    ("left_wrist_yaw_link",),
    ("right_wrist_yaw_link",),
    ("left_knee_link",),
    ("right_knee_link",),
)


def _a2a_body_groups(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
) -> list[torch.Tensor]:
    groups = []
    names_by_part = part_body_names if part_body_names is not None else DEFAULT_A2A_PART_BODY_NAMES
    body_names = list(env.simulator.body_names)  # type: ignore[attr-defined]
    for names in names_by_part:
        ids = [body_names.index(name) for name in names if name in body_names]
        groups.append(torch.tensor(ids, dtype=torch.long, device=env.device))
    return groups


def _a2a_part_contact(env: WholeBodyTrackingManager, groups: list[torch.Tensor], threshold: float) -> torch.Tensor:
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    body_contact = body_force > threshold
    part_contact = []
    for ids in groups:
        if ids.numel() == 0:
            part_contact.append(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        else:
            part_contact.append(body_contact.index_select(1, ids).any(dim=1))
    return torch.stack(part_contact, dim=1)


class ProtoBoundaryContactMismatch(TerminationTermBase):
    """Terminate when a rollout reaches a proto boundary without matching the proto contact state."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.part_indices = tuple(cfg.params.get("part_indices", (0, 1, 2, 3)))
        self.part_body_names = cfg.params.get("part_body_names", None)
        self.threshold = float(cfg.params.get("threshold", 10.0))
        self.require_support_contact = bool(cfg.params.get("require_support_contact", True))
        self.groups = _a2a_body_groups(env, self.part_body_names)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        proto_end = motion_command.motion.proto_end_idx
        proto_motion_ids = motion_command.motion.proto_motion_ids
        at_boundary = (
            (motion_command.time_steps[:, None] == proto_end[None, :])
            & (motion_command.motion_ids[:, None] == proto_motion_ids[None, :])
        ).any(dim=1)
        if not at_boundary.any():
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        part_contact = _a2a_part_contact(self.env, self.groups, self.threshold)[:, list(self.part_indices)]
        expected_support = motion_command.support_part_mask[:, list(self.part_indices)].to(torch.bool)
        mismatch = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if self.require_support_contact:
            mismatch |= (expected_support & ~part_contact).any(dim=1)
        return at_boundary & mismatch

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class A2ASupportSlip(TerminationTermBase):
    """Terminate when a support part slides too fast while in contact."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.part_indices = tuple(cfg.params.get("part_indices", (0, 1, 2, 3)))
        self.part_body_names = cfg.params.get("part_body_names", None)
        self.contact_threshold = float(cfg.params.get("contact_threshold", 10.0))
        self.slip_threshold = float(cfg.params.get("slip_threshold", 0.5))
        self.use_xy_only = bool(cfg.params.get("use_xy_only", True))
        self.groups = _a2a_body_groups(env, self.part_body_names)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        support_mask = motion_command.support_part_mask[:, list(self.part_indices)].to(torch.bool)
        if not support_mask.any():
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        body_force = torch.norm(self.env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
        body_contact = body_force > self.contact_threshold
        vel_dims = 2 if self.use_xy_only else 3
        body_speed = torch.norm(self.env.simulator._rigid_body_vel[:, :, :vel_dims], dim=-1)
        body_slip = body_speed * body_contact.to(body_speed.dtype)

        part_slip = []
        for ids in self.groups:
            if ids.numel() == 0:
                part_slip.append(torch.zeros(self.env.num_envs, dtype=body_slip.dtype, device=self.env.device))
            else:
                part_slip.append(body_slip.index_select(1, ids).max(dim=1)[0])
        part_slip = torch.stack(part_slip, dim=1)[:, list(self.part_indices)]
        return ((part_slip > self.slip_threshold) & support_mask).any(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class A2ASupportContactMismatch(TerminationTermBase):
    """Terminate when any expected support part misses contact for several consecutive steps."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.part_indices = tuple(cfg.params.get("part_indices", (0, 1, 2, 3)))
        self.part_body_names = cfg.params.get("part_body_names", None)
        self.contact_threshold = float(cfg.params.get("contact_threshold", 10.0))
        self.grace_steps = int(cfg.params.get("grace_steps", 5))
        self.groups = _a2a_body_groups(env, self.part_body_names)
        self.mismatch_steps = torch.zeros(env.num_envs, len(self.part_indices), dtype=torch.long, device=env.device)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        support_mask = motion_command.support_part_mask[:, list(self.part_indices)].to(torch.bool)
        part_contact = _a2a_part_contact(self.env, self.groups, self.contact_threshold)[:, list(self.part_indices)]
        missing = support_mask & ~part_contact

        self.mismatch_steps = torch.where(
            missing,
            self.mismatch_steps + 1,
            torch.zeros_like(self.mismatch_steps),
        )
        return (self.mismatch_steps >= self.grace_steps).any(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.mismatch_steps.zero_()
        else:
            self.mismatch_steps[env_ids] = 0


class A2AActiveTrackingMismatch(TerminationTermBase):
    """Terminate when any active part's z tracking error is too large."""

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.part_indices = tuple(cfg.params.get("part_indices", (0, 1, 2, 3)))
        self.tracking_body_names = cfg.params.get(
            "tracking_body_names",
            ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
        )
        self.threshold = float(cfg.params.get("threshold", 0.25))
        self.body_indexes_in_track: torch.Tensor | None = None

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        if self.body_indexes_in_track is None:
            self.body_indexes_in_track = self._get_index_of_a_in_b(
                list(self.tracking_body_names),
                motion_command.motion_cfg.body_names_to_track,
                self.env.device,
            )
        active_mask = motion_command.active_part_mask[:, list(self.part_indices)].to(torch.bool)
        z_error = torch.abs(
            motion_command.body_pos_relative_w[:, self.body_indexes_in_track, -1]
            - motion_command.robot_body_pos_w[:, self.body_indexes_in_track, -1]
        )
        mismatch = active_mask & (z_error > self.threshold)
        return mismatch.any(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.exclude_probe_envs = bool(cfg.params.get("exclude_probe_envs", False))
        self.probe_fixed_qualification_boundary = bool(
            cfg.params.get("probe_fixed_qualification_boundary", False)
        )
        if self.exclude_probe_envs and self.probe_fixed_qualification_boundary:
            raise ValueError(
                "exclude_probe_envs and probe_fixed_qualification_boundary are mutually exclusive"
            )
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )
        self.check_motion_body_pos = self.bad_motion_body_pos_body_indexes.numel() > 0

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        ref_pos_error = torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1)
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        ref_ori_error = torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2])
        if self.check_motion_body_pos:
            body_idx = self.bad_motion_body_pos_body_indexes
            motion_body_pos_error = torch.norm(
                motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
            )
            max_motion_body_pos_error = motion_body_pos_error.max(dim=-1).values
        else:
            max_motion_body_pos_error = torch.zeros_like(ref_pos_error)

        bad_ref_pos = ref_pos_error > self.bad_ref_pos_threshold
        bad_ref_ori = ref_ori_error > self.bad_ref_ori_threshold
        bad_motion_body_pos = (
            max_motion_body_pos_error > self.bad_motion_body_pos_threshold
            if self.check_motion_body_pos
            else torch.zeros_like(bad_ref_pos)
        )

        self._last_ref_pos_error = ref_pos_error.detach().clone()
        self._last_ref_ori_error = ref_ori_error.detach().clone()
        self._last_motion_body_pos_error = max_motion_body_pos_error.detach().clone()
        if self.check_motion_body_pos:
            self._last_motion_body_pos_body_index = motion_body_pos_error.argmax(dim=-1).detach().clone()
        else:
            self._last_motion_body_pos_body_index = torch.zeros_like(ref_pos_error, dtype=torch.long)

        self.metrics["bad_ref_pos_error_max"] = ref_pos_error.max()
        self.metrics["bad_ref_ori_error_max"] = ref_ori_error.max()
        self.metrics["bad_motion_body_pos_error_max"] = max_motion_body_pos_error.max()

        curriculum_manager = getattr(self.env, "curriculum_manager", None)
        tracking_precision = (
            curriculum_manager.get_term("tracking_precision") if curriculum_manager is not None else None
        )
        observe_probe_error = getattr(tracking_precision, "observe_probe_tracking_error", None)
        if callable(observe_probe_error):
            observe_probe_error(
                max_motion_body_pos_error,
                ref_position_error=ref_pos_error,
                ref_orientation_error=ref_ori_error,
            )

        if self.probe_fixed_qualification_boundary:
            probe_mask = getattr(motion_command, "_probe_env_mask", None)
            if probe_mask is None or tracking_precision is None:
                raise RuntimeError(
                    "probe_fixed_qualification_boundary requires tracking_precision and probe environments"
                )
            fixed_bad_ref_pos = ref_pos_error > tracking_precision.base_root_pos_threshold
            fixed_bad_ref_ori = ref_ori_error > tracking_precision.base_root_ori_threshold
            fixed_bad_motion_body_pos = (
                max_motion_body_pos_error > tracking_precision.base_body_threshold
                if self.check_motion_body_pos
                else torch.zeros_like(bad_ref_pos)
            )
            bad_ref_pos = torch.where(probe_mask, fixed_bad_ref_pos, bad_ref_pos)
            bad_ref_ori = torch.where(probe_mask, fixed_bad_ref_ori, bad_ref_ori)
            bad_motion_body_pos = torch.where(
                probe_mask,
                fixed_bad_motion_body_pos,
                bad_motion_body_pos,
            )

        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        self.metrics["bad_ref_pos_rate"] = bad_ref_pos.to(torch.float32).mean()
        self.metrics["bad_ref_ori_rate"] = bad_ref_ori.to(torch.float32).mean()
        self.metrics["bad_motion_body_pos_rate"] = bad_motion_body_pos.to(torch.float32).mean()

        if motion_command.motion.has_object:
            object_pos_error = torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            object_ori_error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            bad_object_pos = object_pos_error > self.bad_object_pos_threshold
            bad_object_ori = object_ori_error > self.bad_object_ori_threshold
            bad_tracking |= bad_object_pos | bad_object_ori
            self.metrics["bad_object_pos_rate"] = bad_object_pos.to(torch.float32).mean()
            self.metrics["bad_object_ori_rate"] = bad_object_ori.to(torch.float32).mean()
            self.metrics["bad_object_pos_error_mean"] = object_pos_error.mean()
            self.metrics["bad_object_pos_error_max"] = object_pos_error.max()
            self.metrics["bad_object_ori_error_mean"] = object_ori_error.mean()
            self.metrics["bad_object_ori_error_max"] = object_ori_error.max()

        if getattr(env, "is_evaluating", False):
            start_idx = motion_command.motion.motion_start_idx[motion_command.motion_ids]
            first_eval_step = (env.episode_length_buf <= 1) & (motion_command.time_steps == start_idx)
            bad_tracking = bad_tracking & ~first_eval_step

        if self.exclude_probe_envs:
            probe_mask = getattr(motion_command, "_probe_env_mask", None)
            if probe_mask is not None:
                bad_tracking = bad_tracking & ~probe_mask

        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        if not self.check_motion_body_pos:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)
