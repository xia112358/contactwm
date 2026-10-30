"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Sequence

import torch
from somaforge_core.contact_schema import CONTACT_FORCE_PART_BODY_NAMES

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import CONTACT_FORCE_PART_ORDER, MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.contact_forces import contact_force_view
from holosoma.utils.rotations import get_euler_xyz_in_tensor, quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = torch.nan_to_num(env.action_manager.action, nan=0.0, posinf=0.0, neginf=0.0)
    prev_actions = torch.nan_to_num(env.action_manager.prev_action, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(torch.sum(torch.square(prev_actions - actions), dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def penalty_torque(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize actuator torque magnitude."""
    term = env.action_manager.get_term("joint_control")
    torques = getattr(term, "torques", None)
    if torques is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    torques = torch.nan_to_num(torques, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(torch.sum(torch.square(torques), dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    hard_limits = torch.nan_to_num(env.simulator.hard_dof_pos_limits, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[attr-defined]
    dof_pos = torch.nan_to_num(env.simulator.dof_pos, nan=0.0, posinf=0.0, neginf=0.0)
    m = (hard_limits[:, 0] + hard_limits[:, 1]) / 2
    r = hard_limits[:, 1] - hard_limits[:, 0]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.nan_to_num(torch.sum(out_of_limits, dim=1), nan=0.0, posinf=0.0, neginf=0.0)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    worst_k: int = 0,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    if worst_k > 0:
        count = min(int(worst_k), int(error.shape[-1]))
        error = torch.topk(error, k=count, dim=-1).values
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_selected_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: Sequence[str],
) -> torch.Tensor:
    """Track selected reference bodies without dilution by the full body set."""
    motion_command = _get_motion_command_and_assert_type(env)
    tracked_names = list(motion_command.motion_cfg.body_names_to_track)
    missing = [name for name in body_names if name not in tracked_names]
    if missing:
        raise RuntimeError(f"Selected position reward bodies are not tracked: {missing}")
    indices = torch.as_tensor([tracked_names.index(name) for name in body_names], device=env.device)
    error = torch.sum(
        torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1
    ).index_select(1, indices)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_selected_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    body_names: Sequence[str],
) -> torch.Tensor:
    """Track selected body orientations without dilution by unrelated links."""
    motion_command = _get_motion_command_and_assert_type(env)
    tracked_names = list(motion_command.motion_cfg.body_names_to_track)
    missing = [name for name in body_names if name not in tracked_names]
    if missing:
        raise RuntimeError(f"Selected orientation reward bodies are not tracked: {missing}")
    indices = torch.as_tensor([tracked_names.index(name) for name in body_names], device=env.device)
    error = torch.square(
        quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w)
    ).index_select(1, indices)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# A2A Active/Support Contact Rewards
# ================================================================================================

DEFAULT_A2A_TRACKING_BODY_NAMES = (
    ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link"),
    ("right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link"),
    ("right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"),
)

DEFAULT_A2A_PART_BODY_NAMES = (
    (
        "left_ankle_roll_link",
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "left_ankle_roll_sphere_5_link",
    ),
    (
        "right_ankle_roll_link",
        "right_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_2_link",
        "right_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_4_link",
        "right_ankle_roll_sphere_5_link",
    ),
    ("left_wrist_yaw_link",),
    ("right_wrist_yaw_link",),
    ("left_knee_link",),
    ("right_knee_link",),
)
CONTACT_FORCE_PART_INDICES = tuple(range(len(CONTACT_FORCE_PART_ORDER)))

DEFAULT_CONTACT_FORCE_PART_BODY_NAMES = tuple(
    CONTACT_FORCE_PART_BODY_NAMES[part] for part in CONTACT_FORCE_PART_ORDER
)

SPARSE_CLIMB_CONTACT_BODY_NAMES = (
    (*CONTACT_FORCE_PART_BODY_NAMES["LHEE"], *CONTACT_FORCE_PART_BODY_NAMES["LTOE"]),
    (*CONTACT_FORCE_PART_BODY_NAMES["RHEE"], *CONTACT_FORCE_PART_BODY_NAMES["RTOE"]),
    CONTACT_FORCE_PART_BODY_NAMES["LH"],
    CONTACT_FORCE_PART_BODY_NAMES["RH"],
    CONTACT_FORCE_PART_BODY_NAMES["LK"],
    CONTACT_FORCE_PART_BODY_NAMES["RK"],
)


PROTO_PART_ALIASES = {
    "left_heel": ("left_ankle_roll_sphere_1_link", "left_ankle_roll_sphere_2_link"),
    "left_toe": ("left_ankle_roll_sphere_3_link", "left_ankle_roll_sphere_4_link", "left_ankle_roll_sphere_5_link"),
    "right_heel": ("right_ankle_roll_sphere_1_link", "right_ankle_roll_sphere_2_link"),
    "right_toe": ("right_ankle_roll_sphere_3_link", "right_ankle_roll_sphere_4_link", "right_ankle_roll_sphere_5_link"),
    "left_foot": ("left_ankle_roll_link", "left_ankle_roll_sphere_1_link", "left_foot_contact_point"),
    "right_foot": ("right_ankle_roll_link", "right_ankle_roll_sphere_1_link", "right_foot_contact_point"),
    "left_hand": CONTACT_FORCE_PART_BODY_NAMES["LH"],
    "right_hand": CONTACT_FORCE_PART_BODY_NAMES["RH"],
    "left_knee": ("left_knee_link",),
    "right_knee": ("right_knee_link",),
    "left_hip": ("left_hip_roll_link",),
    "right_hip": ("right_hip_roll_link",),
}


def _finite_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_value(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=value, posinf=value, neginf=value)


def _finite_clamp(tensor: torch.Tensor, min_value: float, max_value: float, bad_value: float = 0.0) -> torch.Tensor:
    return _finite_value(tensor, bad_value).clamp(min=min_value, max=max_value)


def _proto_body_groups(env: WholeBodyTrackingManager, part_names: Sequence[str]) -> list[torch.Tensor]:
    body_names = list(env.simulator.body_names)  # type: ignore[attr-defined]
    groups = []
    for part_name in part_names:
        candidates = PROTO_PART_ALIASES.get(part_name, (part_name,))
        ids = [body_names.index(name) for name in candidates if name in body_names]
        groups.append(torch.tensor(ids, dtype=torch.long, device=env.device))
    return groups


def _proto_part_values(values: torch.Tensor, groups: list[torch.Tensor], reduce: str) -> torch.Tensor:
    if values.dtype != torch.bool:
        values = _finite_zero(values)
    out = []
    for ids in groups:
        if ids.numel() == 0:
            fallback = torch.zeros(
                (values.shape[0], *values.shape[2:]),
                dtype=values.dtype,
                device=values.device,
            )
            out.append(fallback.bool() if values.dtype == torch.bool else fallback)
            continue
        selected = values.index_select(1, ids)
        if reduce == "any":
            out.append(selected.any(dim=1))
        elif reduce == "max":
            out.append(selected.max(dim=1)[0])
        elif reduce == "mean":
            out.append(selected.mean(dim=1))
        elif reduce == "min":
            out.append(selected.min(dim=1)[0])
        else:
            raise ValueError(f"Unsupported proto reduction: {reduce}")
    return torch.stack(out, dim=1) if out else torch.zeros(values.shape[0], 0, device=values.device)


def _distance_to_aabb(points: torch.Tensor, center: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    points = _finite_zero(points)
    center = _finite_zero(center)
    half_extents = torch.nan_to_num(half_extents, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    delta = torch.abs(points - center[:, None, :]) - half_extents[:, None, :]
    outside = torch.clamp(delta, min=0.0)
    return _finite_value(torch.norm(outside, dim=-1), 1.0)


def _as_env_tensor(value: Sequence[float], env: WholeBodyTrackingManager, dims: int = 3) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float32, device=env.device)
    if tensor.numel() != dims:
        raise ValueError(f"Expected {dims} values, got {tensor.numel()}: {value}")
    return _finite_zero(tensor.view(1, dims).repeat(env.num_envs, 1))


class ProtoFunctionalReward(RewardTermBase):
    """Functional proto reward for a single climbing primitive.

    The term intentionally avoids demo pose, root trajectory, end-effector
    trajectory, force, phase, and fixed contact timing tracking. The proto is
    expressed as root progress toward an anchor, support validity, and active
    limb reach/contact inside a configured region.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.prev_progress_potential = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_active_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.active_contact_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_contact_established = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stable_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_root_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.root_goal_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.active_region_center_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __call__(
        self,
        env: WholeBodyTrackingManager,
        *,
        root_goal_offset: Sequence[float] = (0.8, 0.0, 0.45),
        root_goal: Sequence[float] | None = None,
        active_region_center_offset: Sequence[float] = (0.65, 0.0, 0.75),
        active_region_center: Sequence[float] | None = None,
        active_region_half_extents: Sequence[float] = (0.35, 0.35, 0.18),
        support_set: Sequence[str] = ("left_foot", "right_foot"),
        active_set: Sequence[str] = ("left_hand", "right_hand"),
        w_root: float = 1.0,
        w_support: float = 0.5,
        w_active: float = 0.5,
        alpha_forward: float = 1.0,
        alpha_height: float = 1.0,
        forward_axis: Sequence[float] = (1.0, 0.0),
        progress_clip: float = 0.25,
        goal_radius_xy: float = 0.20,
        goal_radius_z: float = 0.15,
        goal_bonus: float = 5.0,
        stable_frames_required: int = 10,
        root_vel_threshold: float = 0.35,
        torso_rp_threshold: float = 0.5,
        contact_threshold: float = 10.0,
        support_contact_bonus: float = 1.0,
        support_lost_penalty: float = 1.0,
        support_slip_threshold: float = 0.20,
        support_slip_weight: float = 1.0,
        active_reach_weight: float = 1.0,
        active_contact_bonus: float = 1.0,
        active_slip_threshold: float = 0.20,
        active_slip_weight: float = 1.0,
        active_valid_contact_frames: int = 3,
        use_motion_part_masks: bool = False,
        active_region_from_terrain: bool = False,
        active_region_grid_size: int = 5,
        active_region_surface_mode: str = "max_height",
    ) -> torch.Tensor:
        motion_command = _get_motion_command_and_assert_type(env)
        self.prev_progress_potential[:] = _finite_zero(self.prev_progress_potential)
        self.prev_active_distance[:] = _finite_value(self.prev_active_distance, 1.0)
        self.start_root_pos_w[:] = _finite_zero(self.start_root_pos_w)
        self.root_goal_w[:] = _finite_zero(self.root_goal_w)
        self.active_region_center_w[:] = _finite_zero(self.active_region_center_w)
        self._maybe_initialize(
            env,
            motion_command,
            root_goal,
            root_goal_offset,
            active_region_center,
            active_region_center_offset,
            active_region_half_extents,
            active_region_from_terrain,
            active_region_grid_size,
            active_region_surface_mode,
        )

        root_pos = _finite_zero(motion_command.robot_root_pos_w)
        root_vel = _finite_zero(motion_command.robot_root_lin_vel_w)
        goal = _finite_zero(self.root_goal_w)
        start = _finite_zero(self.start_root_pos_w)

        forward_axis_tensor = _finite_zero(torch.tensor(forward_axis, dtype=torch.float32, device=env.device))
        forward_axis_tensor = forward_axis_tensor / torch.clamp(torch.norm(forward_axis_tensor), min=1e-6)
        forward_total = torch.sum((goal[:, :2] - start[:, :2]) * forward_axis_tensor, dim=1)
        forward_now = torch.sum((root_pos[:, :2] - start[:, :2]) * forward_axis_tensor, dim=1)
        forward_progress = torch.minimum(torch.clamp(forward_now, min=0.0), torch.clamp(forward_total, min=0.0))
        height_total = torch.clamp(goal[:, 2] - start[:, 2], min=0.0)
        height_progress = torch.minimum(torch.clamp(root_pos[:, 2] - start[:, 2], min=0.0), height_total)
        progress_potential = _finite_zero(alpha_forward * forward_progress + alpha_height * height_progress)
        r_root_progress = _finite_clamp(
            progress_potential - self.prev_progress_potential,
            -abs(progress_clip),
            abs(progress_clip),
        )
        self.prev_progress_potential[:] = progress_potential

        root_xy_close = torch.norm(root_pos[:, :2] - goal[:, :2], dim=1) <= goal_radius_xy
        root_z_close = torch.abs(root_pos[:, 2] - goal[:, 2]) <= goal_radius_z
        root_slow = torch.norm(root_vel, dim=1) <= root_vel_threshold
        torso_quat = _finite_zero(motion_command.robot_ref_quat_w)
        torso_rpy = _finite_zero(get_euler_xyz_in_tensor(torso_quat))
        torso_roll_pitch = torch.atan2(torch.sin(torso_rpy[:, :2]), torch.cos(torso_rpy[:, :2]))
        torso_upright = torch.all(torch.abs(torso_roll_pitch) <= torso_rp_threshold, dim=1)
        stable = root_xy_close & root_z_close & root_slow & torso_upright
        self.stable_frames[:] = torch.where(stable, self.stable_frames + 1, torch.zeros_like(self.stable_frames))
        success = self.stable_frames >= max(int(stable_frames_required), 1)
        del goal_bonus
        r_root_goal = root_xy_close.to(torch.float32) * root_z_close.to(torch.float32)
        r_root_goal = torch.maximum(r_root_goal, success.to(torch.float32))
        r_root_goal = _finite_clamp(r_root_goal, 0.0, 1.0)
        r_root = _finite_clamp(r_root_progress + r_root_goal, -abs(progress_clip), 1.0)

        support_part_names, active_part_names, support_mask, active_mask = self._proto_part_sets(
            env,
            motion_command,
            support_set,
            active_set,
            use_motion_part_masks,
        )
        support_groups = _proto_body_groups(env, support_part_names)
        active_groups = _proto_body_groups(env, active_part_names)
        body_force = _finite_zero(torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0])
        body_contact = body_force > contact_threshold
        body_slip = _finite_zero(torch.norm(env.simulator._rigid_body_vel[:, :, :2], dim=-1))

        support_contact = _proto_part_values(body_contact, support_groups, reduce="any")
        support_slip = _proto_part_values(body_slip, support_groups, reduce="mean")
        support_valid = support_contact & (support_slip <= support_slip_threshold)
        support_weight = support_mask.to(torch.float32)
        support_count = support_weight.sum(dim=1).clamp(min=1.0)
        support_valid_score = _finite_clamp((support_valid.to(torch.float32) * support_weight).sum(dim=1) / support_count, 0.0, 1.0)
        support_lost = _finite_clamp(((~support_contact).to(torch.float32) * support_weight).sum(dim=1) / support_count, 0.0, 1.0)
        support_slip_penalty = (torch.clamp(support_slip - support_slip_threshold, min=0.0) * support_weight).sum(dim=1)
        support_slip_penalty = _finite_clamp(support_slip_penalty / support_count, 0.0, 1.0)
        pre_active = ~self.active_contact_established
        r_support = _finite_clamp(
            support_contact_bonus * support_valid_score
            - support_slip_weight * support_slip_penalty
            - support_lost_penalty * support_lost * pre_active.to(torch.float32),
            -abs(support_slip_weight) - abs(support_lost_penalty),
            abs(support_contact_bonus),
        )

        active_pos = _finite_zero(_proto_part_values(env.simulator._rigid_body_pos, active_groups, reduce="mean"))
        active_half_extents = _as_env_tensor(active_region_half_extents, env).clamp_min(0.0)
        active_dist_by_part = torch.nan_to_num(
            _distance_to_aabb(active_pos, _finite_zero(self.active_region_center_w), active_half_extents),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        active_weight = active_mask.to(torch.float32)
        active_count = active_weight.sum(dim=1).clamp(min=1.0)
        if active_dist_by_part.numel() > 0:
            active_distance = _finite_value((active_dist_by_part * active_weight).sum(dim=1) / active_count, 1.0)
        else:
            active_distance = torch.zeros(env.num_envs, device=env.device)
        r_active_reach = _finite_clamp(
            self.prev_active_distance - active_distance,
            -abs(progress_clip),
            abs(progress_clip),
        )
        r_active_reach = torch.where(env.episode_length_buf <= 1, torch.zeros_like(r_active_reach), r_active_reach)
        self.prev_active_distance[:] = active_distance

        active_contact = _proto_part_values(body_contact, active_groups, reduce="any")
        active_in_region = active_dist_by_part <= 1e-4
        active_valid_contact = active_contact & active_in_region
        required_active = active_count
        all_active_valid = (active_valid_contact.to(torch.float32) * active_weight).sum(dim=1) >= required_active
        self.active_contact_frames[:] = torch.where(
            all_active_valid,
            self.active_contact_frames + 1,
            torch.zeros_like(self.active_contact_frames),
        )
        self.active_contact_established[:] |= self.active_contact_frames >= max(int(active_valid_contact_frames), 1)

        active_slip = _proto_part_values(body_slip, active_groups, reduce="mean")
        active_slip_penalty = (torch.clamp(active_slip - active_slip_threshold, min=0.0) * active_weight).sum(dim=1)
        active_slip_penalty = _finite_clamp(active_slip_penalty / required_active, 0.0, 1.0)
        active_contact_score = _finite_clamp((active_valid_contact.to(torch.float32) * active_weight).sum(dim=1) / required_active, 0.0, 1.0)
        r_active_contact = _finite_clamp(
            active_contact_bonus * active_contact_score - active_slip_weight * active_slip_penalty,
            -abs(active_slip_weight),
            abs(active_contact_bonus),
        )
        r_active = torch.where(
            self.active_contact_established,
            r_active_contact,
            active_reach_weight * r_active_reach,
        )

        r_active = _finite_clamp(
            r_active,
            -max(abs(active_reach_weight) * abs(progress_clip), abs(active_slip_weight)),
            max(abs(active_reach_weight) * abs(progress_clip), abs(active_contact_bonus)),
        )
        r_root = _finite_clamp(r_root, -abs(progress_clip), 1.0)
        r_root_progress = _finite_clamp(r_root_progress, -abs(progress_clip), abs(progress_clip))
        r_root_goal = _finite_clamp(r_root_goal, 0.0, 1.0)
        r_support = _finite_clamp(
            r_support,
            -abs(support_slip_weight) - abs(support_lost_penalty),
            abs(support_contact_bonus),
        )
        forward_progress = _finite_zero(forward_progress)
        height_progress = _finite_zero(height_progress)
        reward = _finite_zero(w_root * r_root + w_support * r_support + w_active * r_active)
        env.log_dict["proto/r_root"] = r_root.detach()
        env.log_dict["proto/r_root_progress"] = r_root_progress.detach()
        env.log_dict["proto/r_root_goal"] = r_root_goal.detach()
        env.log_dict["proto/r_support"] = r_support.detach()
        env.log_dict["proto/r_active"] = r_active.detach()
        env.log_dict["proto/active_contact_established_rate"] = _finite_zero(
            self.active_contact_established.to(torch.float32).mean()
        ).detach()
        env.log_dict["proto/success_rate"] = _finite_zero(success.to(torch.float32).mean()).detach()
        env.log_dict["proto/mean_forward_progress"] = _finite_zero(forward_progress.mean()).detach()
        env.log_dict["proto/mean_height_progress"] = _finite_zero(height_progress.mean()).detach()
        return reward

    def _proto_part_sets(
        self,
        env: WholeBodyTrackingManager,
        motion_command: MotionCommand,
        support_set: Sequence[str],
        active_set: Sequence[str],
        use_motion_part_masks: bool,
    ) -> tuple[Sequence[str], Sequence[str], torch.Tensor, torch.Tensor]:
        if use_motion_part_masks and motion_command.motion.part_order:
            part_names = tuple(motion_command.motion.part_order)
            support_mask = motion_command.support_part_mask[:, : len(part_names)].to(torch.bool)
            active_mask = motion_command.active_part_mask[:, : len(part_names)].to(torch.bool)
            if support_mask.any() or active_mask.any():
                return part_names, part_names, support_mask, active_mask

        support_names = tuple(support_set)
        active_names = tuple(active_set)
        support_mask = torch.ones(env.num_envs, len(support_names), dtype=torch.bool, device=env.device)
        active_mask = torch.ones(env.num_envs, len(active_names), dtype=torch.bool, device=env.device)
        return support_names, active_names, support_mask, active_mask

    def _maybe_initialize(
        self,
        env: WholeBodyTrackingManager,
        motion_command: MotionCommand,
        root_goal: Sequence[float] | None,
        root_goal_offset: Sequence[float],
        active_region_center: Sequence[float] | None,
        active_region_center_offset: Sequence[float],
        active_region_half_extents: Sequence[float],
        active_region_from_terrain: bool,
        active_region_grid_size: int,
        active_region_surface_mode: str,
    ) -> None:
        env_ids = torch.where((~self._initialized) | (env.episode_length_buf <= 1))[0]
        if env_ids.numel() == 0:
            return
        root_pos = _finite_zero(motion_command.robot_root_pos_w.detach())
        self.start_root_pos_w[env_ids] = root_pos[env_ids]
        if root_goal is None:
            offset = _as_env_tensor(root_goal_offset, env)
            self.root_goal_w[env_ids] = _finite_zero(root_pos[env_ids] + offset[env_ids])
        else:
            self.root_goal_w[env_ids] = _as_env_tensor(root_goal, env)[env_ids]
        if active_region_center is None:
            offset = _as_env_tensor(active_region_center_offset, env)
            region_center = _finite_zero(root_pos + offset)
        else:
            region_center = _as_env_tensor(active_region_center, env)
        if active_region_from_terrain:
            half_extents = _as_env_tensor(active_region_half_extents, env).clamp_min(0.0)
            region_center = self._terrain_surface_region_center(
                env,
                region_center,
                half_extents,
                active_region_grid_size,
                active_region_surface_mode,
            )
        self.active_region_center_w[env_ids] = _finite_zero(region_center[env_ids])
        self.prev_progress_potential[env_ids] = 0.0
        self.prev_active_distance[env_ids] = 0.0
        self.active_contact_frames[env_ids] = 0
        self.active_contact_established[env_ids] = False
        self.stable_frames[env_ids] = 0
        self._initialized[env_ids] = True

    def _terrain_surface_region_center(
        self,
        env: WholeBodyTrackingManager,
        region_center: torch.Tensor,
        half_extents: torch.Tensor,
        grid_size: int,
        surface_mode: str,
    ) -> torch.Tensor:
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        query_terrain_heights = getattr(terrain_state, "query_terrain_heights", None)
        if not callable(query_terrain_heights):
            return _finite_zero(region_center)

        grid_size = max(int(grid_size), 1)
        if grid_size == 1:
            xy_flat = _finite_zero(region_center[:, :2])
        else:
            x_offsets = torch.linspace(-1.0, 1.0, grid_size, device=env.device) * half_extents[:, 0].mean()
            y_offsets = torch.linspace(-1.0, 1.0, grid_size, device=env.device) * half_extents[:, 1].mean()
            grid_x, grid_y = torch.meshgrid(x_offsets, y_offsets, indexing="ij")
            offsets = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
            xy = region_center[:, None, :2] + offsets[None, :, :]
            xy_flat = _finite_zero(xy.reshape(-1, 2))

        heights = torch.nan_to_num(
            query_terrain_heights(xy_flat).reshape(env.num_envs, -1),
            nan=-torch.inf,
            posinf=-torch.inf,
            neginf=-torch.inf,
        )
        xy_grid = xy_flat.reshape(env.num_envs, -1, 2)
        finite = torch.isfinite(heights)
        safe_heights = torch.where(finite, heights, torch.full_like(heights, -torch.inf))
        if surface_mode == "closest_to_hint":
            target_z = region_center[:, 2:3]
            score = torch.where(finite, -torch.abs(heights - target_z), torch.full_like(heights, -torch.inf))
            best_idx = score.argmax(dim=1)
        elif surface_mode == "center":
            center_idx = heights.shape[1] // 2
            best_idx = torch.full((env.num_envs,), center_idx, dtype=torch.long, device=env.device)
        else:
            best_idx = safe_heights.argmax(dim=1)

        selected_heights = heights[torch.arange(env.num_envs, device=env.device), best_idx]
        selected_xy = xy_grid[torch.arange(env.num_envs, device=env.device), best_idx]
        valid = torch.isfinite(selected_heights)
        surface_center = region_center.clone()
        surface_center[:, :2] = torch.where(valid[:, None], selected_xy, region_center[:, :2])
        surface_center[:, 2] = torch.where(valid, selected_heights, region_center[:, 2])
        surface_center = _finite_zero(surface_center)
        env.log_dict["proto/active_region_surface_height"] = surface_center[:, 2].detach()
        return surface_center

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_progress_potential[env_ids] = 0.0
        self.prev_active_distance[env_ids] = 0.0
        self.active_contact_frames[env_ids] = 0
        self.active_contact_established[env_ids] = False
        self.stable_frames[env_ids] = 0
        self._initialized[env_ids] = False


def _a2a_tracked_body_groups(
    motion_command: MotionCommand,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
) -> list[torch.Tensor]:
    groups = []
    names_by_part = tracking_body_names if tracking_body_names is not None else DEFAULT_A2A_TRACKING_BODY_NAMES
    body_names = list(motion_command.motion_cfg.body_names_to_track)
    for names in names_by_part:
        ids = [body_names.index(name) for name in names if name in body_names]
        groups.append(torch.tensor(ids, dtype=torch.long, device=motion_command.device))
    return groups


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


def _a2a_part_body_values(values: torch.Tensor, groups: list[torch.Tensor], reduce: str = "any") -> torch.Tensor:
    part_values = []
    for ids in groups:
        if ids.numel() == 0:
            fallback = torch.zeros(values.shape[0], device=values.device)
            part_values.append(fallback.bool() if reduce == "any" else fallback)
            continue
        value = values.index_select(1, ids)
        if reduce == "any":
            part_values.append(value.any(dim=1))
        elif reduce == "sum":
            part_values.append(value.sum(dim=1))
        elif reduce == "max":
            part_values.append(value.max(dim=1)[0])
        elif reduce == "mean":
            part_values.append(value.mean(dim=1))
        else:
            raise ValueError(f"Unsupported A2A part reduction: {reduce}")
    return torch.stack(part_values, dim=1)


def _part_contact_force_magnitude(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    force_reduce: str = "sum",
    history_reduce: str = "latest",
) -> torch.Tensor:
    part_history = _part_contact_force_history(env, part_body_names, force_reduce)
    magnitude_history = torch.norm(part_history, dim=-1)
    if history_reduce == "latest":
        return magnitude_history[:, 0]
    if history_reduce == "max":
        return magnitude_history.max(dim=1).values
    raise ValueError(f"Unsupported contact force history reduction: {history_reduce}")


def _part_contact_force_history(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    force_reduce: str = "sum",
) -> torch.Tensor:
    """Aggregate the full-resolution contact sensor into semantic parts.

    Body reduction is performed independently at every physics-history sample,
    matching ``extract_rollout_ref_contact_force_demo.py``.  Required collision
    links must all be present: silently treating an unresolved hand or partial
    foot as contact-free corrupts the supervision signal.
    """
    names_by_part = part_body_names or DEFAULT_CONTACT_FORCE_PART_BODY_NAMES
    group_key = tuple(tuple(names) for names in names_by_part)
    step_cache = getattr(env, "_holosoma_step_cache", None)
    if step_cache is None:
        step_cache = {}
        setattr(env, "_holosoma_step_cache", step_cache)
    cache_key = ("part_contact_force_history", group_key, force_reduce)
    cached = step_cache.get(cache_key)
    if cached is not None:
        return cached

    body_names, body_force_history = contact_force_view(env.simulator)
    index_cache = getattr(env, "_holosoma_contact_group_index_cache", None)
    if index_cache is None:
        index_cache = {}
        setattr(env, "_holosoma_contact_group_index_cache", index_cache)
    index_key = (
        tuple(body_names),
        group_key,
        body_force_history.device.type,
        body_force_history.device.index,
    )
    groups = index_cache.get(index_key)
    if groups is None:
        name_to_index = {name: index for index, name in enumerate(body_names)}
        resolved_groups = []
        for part_index, names in enumerate(names_by_part):
            missing = [name for name in names if name not in name_to_index]
            if missing:
                raise RuntimeError(
                    "Contact sensor is missing required collision links for "
                    f"part {part_index}: {missing}. Available body count={len(body_names)}."
                )
            resolved_groups.append(
                torch.as_tensor(
                    [name_to_index[name] for name in names],
                    dtype=torch.long,
                    device=body_force_history.device,
                )
            )
        groups = tuple(resolved_groups)
        index_cache[index_key] = groups

    part_history = []
    for ids in groups:
        selected = body_force_history.index_select(2, ids)
        if force_reduce == "sum":
            reduced = selected.sum(dim=2)
        elif force_reduce == "mean":
            reduced = selected.mean(dim=2)
        elif force_reduce == "max":
            magnitude = torch.norm(selected, dim=-1)
            max_index = magnitude.argmax(dim=2)
            reduced = torch.gather(
                selected,
                2,
                max_index[..., None, None].expand(-1, -1, 1, 3),
            ).squeeze(2)
        else:
            raise ValueError(f"Unsupported contact force vector reduction: {force_reduce}")
        part_history.append(reduced)
    result = torch.stack(part_history, dim=2)
    step_cache[cache_key] = result
    return result


def _part_contact_force_vector(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    force_reduce: str = "sum",
    history_reduce: str = "latest",
) -> torch.Tensor:
    part_history = _part_contact_force_history(env, part_body_names, force_reduce)
    if history_reduce == "latest":
        return part_history[:, 0]
    if history_reduce == "max":
        magnitude = torch.norm(part_history, dim=-1)
        max_index = magnitude.argmax(dim=1)
        return torch.gather(
            part_history,
            1,
            max_index[:, None, :, None].expand(-1, 1, -1, 3),
        ).squeeze(1)
    raise ValueError(f"Unsupported contact force history reduction: {history_reduce}")


def _a2a_support_mask(
    env: WholeBodyTrackingManager,
    part_indices: Sequence[int],
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.support_part_mask[:, list(part_indices)].to(torch.bool)


def _a2a_contact_mask(
    env: WholeBodyTrackingManager,
    part_indices: Sequence[int],
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.contact_part_mask[:, list(part_indices)].to(torch.bool)


def _a2a_active_mask(
    motion_command: MotionCommand,
    part_indices: Sequence[int],
) -> torch.Tensor:
    return motion_command.active_part_mask[:, list(part_indices)].to(torch.bool)


def _a2a_grouped_body_error(error: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    part_errors = []
    for ids in groups:
        if ids.numel() == 0:
            part_errors.append(torch.zeros(error.shape[0], device=error.device, dtype=error.dtype))
        else:
            part_errors.append(error.index_select(1, ids).mean(dim=1))
    return torch.stack(part_errors, dim=1)


def _a2a_active_tracking_reward(
    part_error: torch.Tensor,
    active_mask: torch.Tensor,
    sigma: float,
    normalize: bool,
) -> torch.Tensor:
    active = active_mask.to(part_error.dtype)
    denom = active.sum(dim=1).clamp(min=1.0)
    error = (part_error * active).sum(dim=1)
    if normalize:
        error = error / denom
    return torch.exp(-error / sigma**2)


def _a2a_single_active_tracking_reward(
    part_error: torch.Tensor,
    active_mask: torch.Tensor,
    sigma: float,
    part_index: int,
) -> torch.Tensor:
    active = active_mask[:, part_index].to(part_error.dtype)
    return torch.exp(-part_error[:, part_index] / sigma**2) * active


def motion_relative_body_position_error_exp_active(
    env: WholeBodyTrackingManager,
    sigma: float,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    normalize: bool = True,
) -> torch.Tensor:
    """Track only limbs marked active by the A2A frame mask."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_active_tracking_reward(part_error, active_mask, sigma, normalize)


def motion_relative_body_position_error_exp_active_part(
    env: WholeBodyTrackingManager,
    sigma: float,
    part_index: int,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Track one active limb's body positions."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_single_active_tracking_reward(part_error, active_mask, sigma, part_index)


def motion_relative_body_orientation_error_exp_active(
    env: WholeBodyTrackingManager,
    sigma: float,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    normalize: bool = True,
) -> torch.Tensor:
    """Track orientation only for limbs marked active by the A2A frame mask."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_active_tracking_reward(part_error, active_mask, sigma, normalize)


def motion_relative_body_orientation_error_exp_active_part(
    env: WholeBodyTrackingManager,
    sigma: float,
    part_index: int,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Track one active limb's body orientations."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_single_active_tracking_reward(part_error, active_mask, sigma, part_index)


def motion_global_body_lin_vel_active(
    env: WholeBodyTrackingManager,
    sigma: float,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    normalize: bool = True,
) -> torch.Tensor:
    """Track linear velocity only for limbs marked active by the A2A frame mask."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_active_tracking_reward(part_error, active_mask, sigma, normalize)


def motion_global_body_lin_vel_active_part(
    env: WholeBodyTrackingManager,
    sigma: float,
    part_index: int,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Track one active limb's linear velocity."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_single_active_tracking_reward(part_error, active_mask, sigma, part_index)


def motion_global_body_ang_vel_active(
    env: WholeBodyTrackingManager,
    sigma: float,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    normalize: bool = True,
) -> torch.Tensor:
    """Track angular velocity only for limbs marked active by the A2A frame mask."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_active_tracking_reward(part_error, active_mask, sigma, normalize)


def motion_global_body_ang_vel_active_part(
    env: WholeBodyTrackingManager,
    sigma: float,
    part_index: int,
    tracking_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
) -> torch.Tensor:
    """Track one active limb's angular velocity."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    part_error = _a2a_grouped_body_error(error, _a2a_tracked_body_groups(motion_command, tracking_body_names))
    active_mask = _a2a_active_mask(motion_command, part_indices)
    return _a2a_single_active_tracking_reward(part_error, active_mask, sigma, part_index)


def a2a_support_contact_match(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    threshold: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Reward actual contact on parts marked as A2A support."""
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_contact = _a2a_part_body_values(body_force > threshold, groups, reduce="any")[:, list(part_indices)]
    support_mask = _a2a_support_mask(env, part_indices)
    match = support_mask & part_contact
    reward = match.to(torch.float32).sum(dim=1)
    if normalize:
        reward = reward / support_mask.to(torch.float32).sum(dim=1).clamp(min=1.0)
    return reward


def a2a_support_force_progress(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    target_force: float = 80.0,
    force_reduce: str = "max",
    normalize: bool = True,
) -> torch.Tensor:
    """Reward support contact force continuously up to a target force."""
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_force = _a2a_part_body_values(body_force, groups, reduce=force_reduce)
    support_mask = _a2a_support_mask(env, part_indices).to(part_force.dtype)
    force_score = torch.clamp(part_force[:, list(part_indices)] / max(target_force, 1e-6), min=0.0, max=1.0)
    reward = (force_score * support_mask).sum(dim=1)
    if normalize:
        reward = reward / support_mask.sum(dim=1).clamp(min=1.0)
    return reward


def a2a_support_contact_match_part(
    env: WholeBodyTrackingManager,
    part_index: int,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    threshold: float = 10.0,
) -> torch.Tensor:
    """Reward actual contact for one limb when that limb is marked as support."""
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_contact = _a2a_part_body_values(body_force > threshold, groups, reduce="any")[:, list(part_indices)]
    support_mask = _a2a_support_mask(env, part_indices)
    return (support_mask[:, part_index] & part_contact[:, part_index]).to(torch.float32)


def a2a_support_slip(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    contact_threshold: float = 10.0,
    use_xy_only: bool = False,
    include_ang_vel: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Historical reward proxy for desired keep roles, preserving policy behavior.

    Net-force threshold and body-origin speed are reward shaping only. They do
    not establish native contact activation, actual load-bearing, or patch slip;
    collection and acceptance use somaforge_core.newton_support instead.
    """
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    body_contact = body_force > contact_threshold
    lin_vel = torch.norm(env.simulator._rigid_body_vel[:, :, :2 if use_xy_only else 3], dim=-1)
    body_slip = lin_vel * body_contact.to(lin_vel.dtype)
    if include_ang_vel:
        ang_vel = torch.norm(env.simulator._rigid_body_ang_vel, dim=-1)
        body_slip = body_slip + ang_vel * body_contact.to(ang_vel.dtype)
    part_slip = _a2a_part_body_values(body_slip, groups, reduce="mean")[:, list(part_indices)]
    support_mask = _a2a_support_mask(env, part_indices).to(part_slip.dtype)
    reward = (part_slip * support_mask).sum(dim=1)
    if normalize:
        reward = reward / support_mask.sum(dim=1).clamp(min=1.0)
    return reward


def a2a_support_force_validity(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    min_force: float = 10.0,
    max_force: float = 1500.0,
    force_reduce: str = "max",
    normalize: bool = True,
) -> torch.Tensor:
    """Penalize too-small or too-large support contact force magnitudes."""
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_force = _a2a_part_body_values(body_force, groups, reduce=force_reduce)
    support_mask = _a2a_support_mask(env, part_indices).to(part_force.dtype)
    part_force = part_force[:, list(part_indices)]
    force_error = torch.clamp(min_force - part_force, min=0.0) / max(min_force, 1e-6)
    force_error = force_error + torch.clamp(part_force - max_force, min=0.0) / max(max_force, 1e-6)
    reward = (force_error * support_mask).sum(dim=1)
    if normalize:
        reward = reward / support_mask.sum(dim=1).clamp(min=1.0)
    return reward


def a2a_unexpected_contact(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    threshold: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Penalize contact on A2A parts when the reference contact mask says the part should be free."""
    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_contact = _a2a_part_body_values(body_force > threshold, groups, reduce="any")[:, list(part_indices)]
    expected_contact = _a2a_contact_mask(env, part_indices)
    unexpected = part_contact & ~expected_contact
    reward = unexpected.to(torch.float32).sum(dim=1)
    if normalize:
        reward = reward / max(len(part_indices), 1)
    return reward


def motion_contact_force_magnitude_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float = 0.5,
    force_scale: float = 300.0,
    contact_threshold: float = 10.0,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = CONTACT_FORCE_PART_INDICES,
    force_reduce: str = "sum",
) -> torch.Tensor:
    """Track rollout-derived true contact-force magnitudes with an exponential kernel."""
    motion_command = _get_motion_command_and_assert_type(env)
    actual_force = _part_contact_force_magnitude(env, part_body_names, force_reduce)[:, list(part_indices)]
    ref_force = torch.norm(motion_command.contact_force_part_w[:, list(part_indices)], dim=-1)
    ref_contact = motion_command.contact_force_part_mask[:, list(part_indices)] | (ref_force > contact_threshold)
    expected = ref_contact.to(actual_force.dtype)
    error = torch.square((actual_force - ref_force) / max(force_scale, 1e-6)) * expected
    denom = expected.sum(dim=1).clamp(min=1.0)
    return torch.exp(-(error.sum(dim=1) / denom) / max(sigma, 1e-6) ** 2)


def _sparse_contact_balanced_score(
    actual_contact: torch.Tensor,
    expected_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return score, expected-contact recall, and expected-free specificity."""
    expected_count = expected_contact.sum(dim=-1)
    expected_free_count = (~expected_contact).sum(dim=-1)
    true_positive = (actual_contact & expected_contact).sum(dim=-1)
    true_negative = ((~actual_contact) & (~expected_contact)).sum(dim=-1)

    recall = torch.where(
        expected_count > 0,
        true_positive.to(torch.float32) / expected_count.clamp(min=1).to(torch.float32),
        torch.ones_like(expected_count, dtype=torch.float32),
    )
    specificity = torch.where(
        expected_free_count > 0,
        true_negative.to(torch.float32) / expected_free_count.clamp(min=1).to(torch.float32),
        torch.ones_like(expected_free_count, dtype=torch.float32),
    )
    return recall * specificity, recall, specificity


def motion_sparse_contact_match(
    env: WholeBodyTrackingManager,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Balanced six-part contact score for LF/RF/LH/RH/LK/RK.

    Expected-contact recall is multiplied by expected-free specificity. This
    prevents the class-imbalance shortcut where predicting no contacts earns a
    high mean accuracy. Canonical sphere-hand collision links are used rather
    than wrist links, matching the rollout contact schema and robot asset.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    # Reproduce the demonstration-side contract exactly: reduce each of the
    # eight canonical parts independently, take the maximum across the saved
    # physics history, then merge heel/toe masks into LF/RF.
    actual_part_force = _part_contact_force_magnitude(
        env,
        DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
        force_reduce="sum",
        history_reduce="max",
    )
    actual_part_contact = actual_part_force > contact_threshold
    actual_contact = torch.stack(
        (
            actual_part_contact[..., 0] | actual_part_contact[..., 1],
            actual_part_contact[..., 2] | actual_part_contact[..., 3],
            actual_part_contact[..., 4],
            actual_part_contact[..., 5],
            actual_part_contact[..., 6],
            actual_part_contact[..., 7],
        ),
        dim=-1,
    )
    force_mask = motion_command.contact_force_part_mask.to(torch.bool)
    expected_contact = torch.stack(
        (
            force_mask[..., 0] | force_mask[..., 1],
            force_mask[..., 2] | force_mask[..., 3],
            force_mask[..., 4],
            force_mask[..., 5],
            force_mask[..., 6],
            force_mask[..., 7],
        ),
        dim=-1,
    )
    score, recall, specificity = _sparse_contact_balanced_score(actual_contact, expected_contact)
    expected_count = expected_contact.sum(dim=-1)

    # Direct step metrics make failures attributable without adding zero-weight
    # reward terms (which RewardManager intentionally skips).
    env.log_dict["sparse_climb/contact_recall"] = recall.detach()
    env.log_dict["sparse_climb/contact_specificity"] = specificity.detach()
    env.log_dict["sparse_climb/contact_expected_count"] = expected_count.to(torch.float32).detach()
    for part_index, part_name in enumerate(("lf", "rf", "lh", "rh", "lk", "rk")):
        env.log_dict[f"sparse_climb/contact_match_{part_name}"] = (
            actual_contact[:, part_index] == expected_contact[:, part_index]
        ).to(torch.float32).detach()
        env.log_dict[f"sparse_climb/contact_actual_{part_name}"] = actual_contact[:, part_index].to(
            torch.float32
        ).detach()
        env.log_dict[f"sparse_climb/contact_expected_{part_name}"] = expected_contact[:, part_index].to(
            torch.float32
        ).detach()
    return score


def motion_sparse_contact_match_exp(
    env: WholeBodyTrackingManager,
    sigma: float = 1.0,
    contact_threshold: float = 10.0,
) -> torch.Tensor:
    """Legacy sparse-contact reward retained for saved-checkpoint replay.

    New experiments must use :func:`motion_sparse_contact_match`. Keeping the
    historical implementation here lets old configs load without silently
    changing the behavior being diagnosed.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    actual_force = _part_contact_force_magnitude(
        env,
        DEFAULT_A2A_PART_BODY_NAMES,
        force_reduce="sum",
        history_reduce="max",
    )
    actual_contact = actual_force > contact_threshold
    force_mask = motion_command.contact_force_part_mask.to(torch.bool)
    expected_contact = torch.stack(
        (
            force_mask[..., 0] | force_mask[..., 1],
            force_mask[..., 2] | force_mask[..., 3],
            force_mask[..., 4],
            force_mask[..., 5],
            force_mask[..., 6],
            force_mask[..., 7],
        ),
        dim=-1,
    )
    mismatch = torch.ne(actual_contact, expected_contact).to(torch.float32).mean(dim=-1)
    return torch.exp(-mismatch / max(sigma, 1e-6) ** 2)


def motion_contact_force_relative_magnitude_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float = 0.5,
    force_floor: float = 50.0,
    contact_threshold: float = 10.0,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = CONTACT_FORCE_PART_INDICES,
    force_reduce: str = "sum",
) -> torch.Tensor:
    """Track contact-force magnitudes by relative error with an exponential kernel."""
    motion_command = _get_motion_command_and_assert_type(env)
    actual_force = _part_contact_force_magnitude(env, part_body_names, force_reduce)[:, list(part_indices)]
    ref_force = torch.norm(motion_command.contact_force_part_w[:, list(part_indices)], dim=-1)
    ref_contact = motion_command.contact_force_part_mask[:, list(part_indices)] | (ref_force > contact_threshold)
    expected = ref_contact.to(actual_force.dtype)
    denom_force = torch.clamp(ref_force, min=max(force_floor, 1e-6))
    relative_error = torch.abs(actual_force - ref_force) / denom_force
    error = torch.square(relative_error) * expected
    denom = expected.sum(dim=1).clamp(min=1.0)
    return torch.exp(-(error.sum(dim=1) / denom) / max(sigma, 1e-6) ** 2)


def motion_contact_force_relative_vector_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float = 0.5,
    force_floor: float = 50.0,
    contact_threshold: float = 10.0,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = CONTACT_FORCE_PART_INDICES,
    force_reduce: str = "sum",
) -> torch.Tensor:
    """Track rollout-derived true contact-force vectors by relative 3D error."""
    motion_command = _get_motion_command_and_assert_type(env)
    part_ids = list(part_indices)
    actual_force = _part_contact_force_vector(env, part_body_names, force_reduce)[:, part_ids]
    ref_force = motion_command.contact_force_part_w[:, part_ids]
    ref_force_magnitude = torch.norm(ref_force, dim=-1)
    ref_contact = motion_command.contact_force_part_mask[:, part_ids] | (ref_force_magnitude > contact_threshold)
    expected = ref_contact.to(actual_force.dtype)
    denom_force = torch.clamp(ref_force_magnitude, min=max(force_floor, 1e-6))
    relative_error = torch.norm(actual_force - ref_force, dim=-1) / denom_force
    error = torch.square(relative_error) * expected
    denom = expected.sum(dim=1).clamp(min=1.0)
    return torch.exp(-(error.sum(dim=1) / denom) / max(sigma, 1e-6) ** 2)


def motion_contact_force_unexpected_contact(
    env: WholeBodyTrackingManager,
    contact_threshold: float = 10.0,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = CONTACT_FORCE_PART_INDICES,
    force_reduce: str = "sum",
) -> torch.Tensor:
    """Penalize true part contacts when the rollout demo has no contact for that part."""
    motion_command = _get_motion_command_and_assert_type(env)
    actual_force = _part_contact_force_magnitude(env, part_body_names, force_reduce)[:, list(part_indices)]
    ref_force = torch.norm(motion_command.contact_force_part_w[:, list(part_indices)], dim=-1)
    ref_contact = motion_command.contact_force_part_mask[:, list(part_indices)] | (ref_force > contact_threshold)
    unexpected = (actual_force > contact_threshold) & ~ref_contact
    return unexpected.to(torch.float32).sum(dim=1) / max(len(part_indices), 1)


def chain_transition_outcome(
    env: WholeBodyTrackingManager,
    part_body_names: Sequence[Sequence[str]] | None = None,
    part_indices: Sequence[int] = (0, 1, 2, 3),
    contact_threshold: float = 10.0,
    tracking_sigma: float = 0.25,
    contact_weight: float = 1.0,
    tracking_weight: float = 1.0,
) -> torch.Tensor:
    """Reward post-chain transition results during the pending grace window."""
    motion_command = _get_motion_command_and_assert_type(env)
    pending = getattr(motion_command, "_pending_chain_check", None)
    if pending is None or not torch.any(pending):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    groups = _a2a_body_groups(env, part_body_names)
    body_force = torch.norm(env.simulator.contact_forces_history, dim=-1).max(dim=1)[0]
    part_contact = _a2a_part_body_values(body_force > contact_threshold, groups, reduce="any")[:, list(part_indices)]
    support_mask = motion_command.support_part_mask[:, list(part_indices)].to(torch.bool)
    support_count = support_mask.to(torch.float32).sum(dim=1).clamp(min=1.0)
    contact_score = (support_mask & part_contact).to(torch.float32).sum(dim=1) / support_count
    contact_score = torch.where(support_mask.any(dim=1), contact_score, torch.ones_like(contact_score))

    limb_idx = motion_command.a2a_limb_body_indexes_in_track
    z_error = torch.abs(
        motion_command.body_pos_relative_w[:, limb_idx, -1] - motion_command.robot_body_pos_w[:, limb_idx, -1]
    )
    tracking_score = torch.exp(-(z_error.mean(dim=1) ** 2) / max(tracking_sigma, 1e-6) ** 2)

    reward = contact_weight * contact_score + tracking_weight * tracking_score
    return reward * pending.to(torch.float32)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
