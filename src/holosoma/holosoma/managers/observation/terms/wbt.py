"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.utils.rotations import (
    quat_apply_yaw,
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
    yaw_quat,
)
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


def php_velocity_command(
    env: WholeBodyTrackingManager, command: tuple[float, float] = (1.0, 0.0)
) -> torch.Tensor:
    """Planar velocity command paired with the climb trajectory."""
    return torch.tensor(command, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)


class PHPDepthImage(ObservationTermBase):
    """Noisy 58x87 ray depth with the paper's 60--80 ms observation delay."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.max_distance = float(params.get("max_distance", 5.0))
        self.offset_range = tuple(params.get("offset_range", (-0.03, 0.03)))
        self.gaussian_std = float(params.get("gaussian_std", 0.03))
        self.delay_steps = tuple(params.get("delay_steps", (3, 4)))
        self._buffer = None
        self._cursor = 0
        self._delay = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        low, high = int(self.delay_steps[0]), int(self.delay_steps[1])
        self._delay[env_ids] = torch.randint(low, high + 1, (env_ids.numel(),), device=self.env.device)
        if self._buffer is not None:
            self._buffer[:, env_ids] = self._buffer[self._cursor, env_ids]

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        sensor = env.simulator.scene.sensors["php_depth_camera"]
        ray_hits_w = sensor.data.ray_hits_w.torch
        sensor_pos_w = sensor.data.pos_w.torch
        depth = torch.linalg.vector_norm(ray_hits_w - sensor_pos_w[:, None, :], dim=-1)
        depth = torch.nan_to_num(depth, nan=self.max_distance, posinf=self.max_distance, neginf=0.0)
        depth = depth.clamp_(0.0, self.max_distance)
        offset = torch.empty((env.num_envs, 1), device=env.device).uniform_(*self.offset_range)
        depth = (depth + offset + torch.randn_like(depth) * self.gaussian_std).clamp_(0.0, self.max_distance)
        depth = depth.flatten(1)

        buffer_len = int(self.delay_steps[1]) + 1
        if self._buffer is None:
            self._buffer = depth.unsqueeze(0).repeat(buffer_len, 1, 1)
            self.reset()
        self._cursor = (self._cursor + 1) % buffer_len
        self._buffer[self._cursor] = depth
        read_idx = (self._cursor - self._delay) % buffer_len
        env_idx = torch.arange(env.num_envs, device=env.device)
        return self._buffer[read_idx, env_idx]


def proto_command(
    env: WholeBodyTrackingManager,
    root_goal_offset: tuple[float, float, float] = (0.8, 0.0, 0.45),
    active_region_center_offset: tuple[float, float, float] = (0.65, 0.0, 0.75),
    forward_axis: tuple[float, float] = (1.0, 0.0),
) -> torch.Tensor:
    """Task-level proto command without per-frame demo references."""
    root_goal = torch.tensor(root_goal_offset, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)
    active_region = torch.tensor(active_region_center_offset, dtype=torch.float32, device=env.device).expand(
        env.num_envs, -1
    )
    forward = torch.tensor(forward_axis, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)
    return torch.cat((root_goal, active_region, forward), dim=-1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_ref_joint_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.joint_pos - env.default_dof_pos


def motion_ref_joint_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.joint_vel


_FUTURE_REFERENCE_OFFSETS = (1, 2, 4, 8)
_SPARSE_CLIMB_REFERENCE_OFFSETS = (0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50)
_POLICY_REFERENCE_OFFSETS = (0, 1, 2, 4, 8)

_SPARSE_CLIMB_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "right_knee_link",
)


def policy_reference_window(
    env: WholeBodyTrackingManager,
    include_current: bool,
    add_noise: bool,
    include_joint_velocity: bool = True,
) -> torch.Tensor:
    """Return a fused, time-major policy reference window.

    Each full frame is ``[q(29), qd(29), torso_pos_b(3), torso_rot6d_b(6)]``.
    When ``include_joint_velocity`` is false, the qd block is omitted and each
    frame becomes ``[q(29), torso_pos_b(3), torso_rot6d_b(6)]``.
    The clean five-frame tensor is shared by actor and critic within a command step.
    """

    motion_command = _get_motion_command_and_assert_type(env)
    frames = motion_command._policy_reference_window_cache
    if frames is None:
        joint_pos, joint_vel = motion_command.future_joint_pos_vel(_POLICY_REFERENCE_OFFSETS)
        body_pos_w, body_quat_w = motion_command.future_ref_body_pos_quat_offsets(_POLICY_REFERENCE_OFFSETS)

        num_frames = len(_POLICY_REFERENCE_OFFSETS)
        robot_pos_w = motion_command.robot_ref_pos_w[:, None, :].expand(-1, num_frames, -1)
        robot_quat_w = motion_command.robot_ref_quat_w[:, None, :].expand(-1, num_frames, -1)
        pos_b, ori_b = subtract_frame_transforms(
            robot_pos_w.reshape(-1, 3),
            robot_quat_w.reshape(-1, 4),
            body_pos_w.reshape(-1, 3),
            body_quat_w.reshape(-1, 4),
        )
        pos_b = pos_b.reshape(env.num_envs, num_frames, 3)
        ori_rot6d_b = quaternion_to_matrix(ori_b, w_last=True)[..., :2].reshape(env.num_envs, num_frames, 6)
        frames = torch.cat([joint_pos, joint_vel, pos_b, ori_rot6d_b], dim=-1)
        motion_command._policy_reference_window_cache = frames

    frames = frames if include_current else frames[:, 1:]
    if add_noise:
        frames = frames.clone()
        frames[..., 58:61] += (torch.rand_like(frames[..., 58:61]) * 2.0 - 1.0) * 0.25
        frames[..., 61:67] += (torch.rand_like(frames[..., 61:67]) * 2.0 - 1.0) * 0.05
    if not include_joint_velocity:
        frames = torch.cat([frames[..., :29], frames[..., 58:67]], dim=-1)
    return frames.reshape(env.num_envs, -1)


def sparse_climb_reference_window(
    env: WholeBodyTrackingManager,
    add_noise: bool,
) -> torch.Tensor:
    """Future climb intent without joint targets.

    Per frame layout is torso pose(9), LF/RF/LH/RH/LK/RK poses(54), and
    matching contact bits(6). All spatial targets use the current executed
    torso-yaw frame, preserving gravity-aligned vertical directions while
    removing global translation and heading.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    body_names = list(motion_command.motion_cfg.body_names_to_track)
    missing = [name for name in _SPARSE_CLIMB_BODY_NAMES if name not in body_names]
    if missing:
        raise RuntimeError(f"Sparse climb reference bodies are not tracked: {missing}")
    selected = torch.as_tensor(
        [body_names.index(name) for name in _SPARSE_CLIMB_BODY_NAMES],
        dtype=torch.long,
        device=env.device,
    )
    body_pos_w, body_quat_w = motion_command.future_body_pos_quat_offsets(
        _SPARSE_CLIMB_REFERENCE_OFFSETS
    )
    body_pos_w = body_pos_w.index_select(2, selected)
    body_quat_w = body_quat_w.index_select(2, selected)
    frame_count = len(_SPARSE_CLIMB_REFERENCE_OFFSETS)
    body_count = len(_SPARSE_CLIMB_BODY_NAMES)
    robot_pos_w = motion_command.robot_ref_pos_w[:, None, None, :].expand(
        -1, frame_count, body_count, -1
    )
    robot_yaw_quat_w = yaw_quat(motion_command.robot_ref_quat_w, w_last=True)
    robot_quat_w = robot_yaw_quat_w[:, None, None, :].expand(
        -1, frame_count, body_count, -1
    )
    pos_b, quat_b = subtract_frame_transforms(
        robot_pos_w.reshape(-1, 3),
        robot_quat_w.reshape(-1, 4),
        body_pos_w.reshape(-1, 3),
        body_quat_w.reshape(-1, 4),
    )
    pos_b = pos_b.reshape(env.num_envs, frame_count, body_count, 3)
    rot6d_b = quaternion_to_matrix(quat_b, w_last=True)[..., :2].reshape(
        env.num_envs, frame_count, body_count, 6
    )

    # CONTACT_FORCE_PART_ORDER is LHEEL/LTOE/RHEEL/RTOE/LH/RH/LK/RK.
    force_mask = motion_command.future_contact_force_part_mask_offsets(
        _SPARSE_CLIMB_REFERENCE_OFFSETS
    ).to(torch.bool)
    contact = torch.stack(
        (
            force_mask[..., 0] | force_mask[..., 1],
            force_mask[..., 2] | force_mask[..., 3],
            force_mask[..., 4],
            force_mask[..., 5],
            force_mask[..., 6],
            force_mask[..., 7],
        ),
        dim=-1,
    ).to(pos_b.dtype)
    # Selected body order is pelvis, torso, LF, RF, LH, RH, LK, RK. Pelvis is
    # intentionally omitted: torso supplies the root target, while all six
    # contact-capable endpoints carry the same position/orientation contract.
    endpoint_pos = pos_b[..., 2:8, :].reshape(env.num_envs, frame_count, -1)
    endpoint_rot6d = rot6d_b[..., 2:8, :].reshape(env.num_envs, frame_count, -1)
    frames = torch.cat(
        (pos_b[..., 1, :], rot6d_b[..., 1, :], endpoint_pos, endpoint_rot6d, contact),
        dim=-1,
    )
    if add_noise:
        frames = frames.clone()
        frames[..., :3] += (torch.rand_like(frames[..., :3]) * 2.0 - 1.0) * 0.02
        frames[..., 3:9] += (torch.rand_like(frames[..., 3:9]) * 2.0 - 1.0) * 0.03
        frames[..., 9:27] += (torch.rand_like(frames[..., 9:27]) * 2.0 - 1.0) * 0.02
        frames[..., 27:63] += (torch.rand_like(frames[..., 27:63]) * 2.0 - 1.0) * 0.03
    return frames.reshape(env.num_envs, -1)


def future_motion_command(env: WholeBodyTrackingManager, offset: int) -> torch.Tensor:
    """Return one future frame with the same q/qd layout as ``motion_command``."""

    motion_command = _get_motion_command_and_assert_type(env)
    joint_pos, joint_vel = motion_command.future_joint_pos_vel(_FUTURE_REFERENCE_OFFSETS)
    offset_index = _FUTURE_REFERENCE_OFFSETS.index(offset)
    return torch.cat([joint_pos[:, offset_index], joint_vel[:, offset_index]], dim=1)


def future_motion_ref_pos_b(
    env: WholeBodyTrackingManager,
    offset: int,
) -> torch.Tensor:
    """Return the future torso target position relative to the current robot torso."""

    motion_command = _get_motion_command_and_assert_type(env)
    body_pos_w, body_quat_w = motion_command.future_ref_body_pos_quat_offsets(_FUTURE_REFERENCE_OFFSETS)
    offset_index = _FUTURE_REFERENCE_OFFSETS.index(offset)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        body_pos_w[:, offset_index],
        body_quat_w[:, offset_index],
    )
    return pos_b.view(env.num_envs, -1)


def future_motion_ref_ori_b(
    env: WholeBodyTrackingManager,
    offset: int,
) -> torch.Tensor:
    """Return the future torso target orientation relative to the current robot torso."""

    motion_command = _get_motion_command_and_assert_type(env)
    body_pos_w, body_quat_w = motion_command.future_ref_body_pos_quat_offsets(_FUTURE_REFERENCE_OFFSETS)
    offset_index = _FUTURE_REFERENCE_OFFSETS.index(offset)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        body_pos_w[:, offset_index],
        body_quat_w[:, offset_index],
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def pelvis_global_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.simulator.robot_root_states[:, :3]


def pelvis_global_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.simulator.robot_root_states[:, 7:10]


def terrain_height_scan(
    env: WholeBodyTrackingManager,
    scan_size: float = 0.7,
    num_points_per_axis: int = 7,
    offset: float = 0.5,
) -> torch.Tensor:
    scene = getattr(env.simulator, "scene", None)
    height_scanner = getattr(scene, "sensors", {}).get("height_scanner") if scene is not None else None
    if height_scanner is not None:
        ray_hits_w = height_scanner.data.ray_hits_w.torch
        sensor_pos_w = height_scanner.data.pos_w.torch
        if ray_hits_w.shape[0] != env.num_envs:
            raise RuntimeError(
                f"Height scanner returned {ray_hits_w.shape[0]} scanner frames for {env.num_envs} envs."
            )
        return sensor_pos_w[:, 2].unsqueeze(1) - ray_hits_w[..., 2] - offset

    terrain_state = env.terrain_manager.get_state("locomotion_terrain")
    query_terrain_heights = getattr(terrain_state, "query_terrain_heights", None)
    if not callable(query_terrain_heights):
        return torch.zeros(env.num_envs, num_points_per_axis * num_points_per_axis, device=env.device)

    half_extent = scan_size / 2.0
    points_1d = torch.linspace(-half_extent, half_extent, num_points_per_axis, device=env.device)
    grid_x, grid_y = torch.meshgrid(points_1d, points_1d, indexing="ij")
    scan_points_b = torch.zeros(env.num_envs, num_points_per_axis * num_points_per_axis, 3, device=env.device)
    scan_points_b[:, :, 0] = grid_x.flatten()
    scan_points_b[:, :, 1] = grid_y.flatten()

    scan_points_w = quat_apply_yaw(env.base_quat.repeat(1, scan_points_b.shape[1]), scan_points_b, True)
    scan_points_w = scan_points_w + env.simulator.robot_root_states[:, None, :3]
    terrain_heights = query_terrain_heights(scan_points_w[:, :, :2].reshape(-1, 2))
    terrain_heights = terrain_heights.reshape(env.num_envs, -1)
    return terrain_heights - env.simulator.robot_root_states[:, 2:3]


def active_part_mask(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.active_part_mask.to(torch.float32)


def support_part_mask(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.support_part_mask.to(torch.float32)


def contact_part_mask(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.contact_part_mask.to(torch.float32)


def contact_force_ref_b(env: WholeBodyTrackingManager, force_scale: float = 300.0) -> torch.Tensor:
    """Reference part contact forces from rollout demos in the current reference frame.

    Layout follows the motion's contact-force part order, each with force xyz plus a true-contact mask.
    The mask is generated from measured rollout contact forces, not from proto/support labels.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    force_w = motion_command.contact_force_part_w
    num_parts = force_w.shape[1]
    quat = motion_command.robot_ref_quat_w[:, None, :].expand(-1, num_parts, -1).reshape(-1, 4)
    force_b = quat_rotate_inverse(quat, force_w.reshape(-1, 3), w_last=True).reshape(env.num_envs, num_parts, 3)
    force_b = force_b / max(float(force_scale), 1e-6)
    mask = motion_command.contact_force_part_mask[:, :num_parts].to(force_b.dtype).unsqueeze(-1)
    return torch.cat([force_b, mask], dim=-1).reshape(env.num_envs, -1)


def limb_state(env: WholeBodyTrackingManager) -> torch.Tensor:
    """A2A limb state for [left_foot, right_foot, left_hand, right_hand].

    Encoding: active=1, free=0, support=-1.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    active = motion_command.active_part_mask[:, :4]
    support = motion_command.support_part_mask[:, :4]
    return torch.where(
        support,
        torch.full_like(active, -1.0, dtype=torch.float32),
        active.to(torch.float32),
    )


def a2a_root_ref(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Root/reference-body target in the current robot reference frame.

    Layout: position(3), orientation 6D(6), linear velocity(3), angular velocity(3).
    """
    motion_command = _get_motion_command_and_assert_type(env)
    pos, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    ori_6d = quaternion_to_matrix(ori, w_last=True)[..., :2].reshape(env.num_envs, -1)
    lin_vel = quat_rotate_inverse(
        motion_command.robot_ref_quat_w,
        motion_command.ref_lin_vel_w - motion_command.robot_ref_lin_vel_w,
        w_last=True,
    )
    ang_vel = quat_rotate_inverse(
        motion_command.robot_ref_quat_w,
        motion_command.ref_ang_vel_w - motion_command.robot_ref_ang_vel_w,
        w_last=True,
    )
    return torch.cat([pos, ori_6d, lin_vel, ang_vel], dim=-1)


_A2A_LIMB_REF_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def a2a_active_limb_ref(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Fixed-slot active limb references for LF/RF/LH/RH.

    Each slot contains position(3), orientation 6D(6), linear velocity(3), angular velocity(3).
    Non-active slots are zeroed, so the actor input shape is fixed while active count can vary.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    body_names = list(motion_command.motion_cfg.body_names_to_track)
    num_slots = len(_A2A_LIMB_REF_BODY_NAMES)
    slot_dim = 15
    refs = torch.zeros(env.num_envs, num_slots, slot_dim, device=env.device)

    for slot, body_name in enumerate(_A2A_LIMB_REF_BODY_NAMES):
        if body_name not in body_names:
            continue
        body_idx = body_names.index(body_name)
        pos, ori = subtract_frame_transforms(
            motion_command.robot_ref_pos_w,
            motion_command.robot_ref_quat_w,
            motion_command.body_pos_relative_w[:, body_idx],
            motion_command.body_quat_relative_w[:, body_idx],
        )
        ori_6d = quaternion_to_matrix(ori, w_last=True)[..., :2].reshape(env.num_envs, -1)
        lin_vel = quat_rotate_inverse(
            motion_command.robot_ref_quat_w,
            motion_command.body_lin_vel_w[:, body_idx] - motion_command.robot_ref_lin_vel_w,
            w_last=True,
        )
        ang_vel = quat_rotate_inverse(
            motion_command.robot_ref_quat_w,
            motion_command.body_ang_vel_w[:, body_idx] - motion_command.robot_ref_ang_vel_w,
            w_last=True,
        )
        refs[:, slot] = torch.cat([pos, ori_6d, lin_vel, ang_vel], dim=-1)

    active = motion_command.active_part_mask[:, :num_slots].to(refs.dtype).unsqueeze(-1)
    return (refs * active).reshape(env.num_envs, -1)


def a2a_contact_transition_tag(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Contact transition tags for LF/RF/LH/RH.

    Layout: touchdown(4), liftoff(4), computed from previous frame to current frame.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    current = motion_command.contact_part_mask[:, :4].to(torch.bool)
    start_idx = motion_command.motion.motion_start_idx[motion_command.motion_ids]
    prev_steps = torch.maximum(motion_command.time_steps - 1, start_idx)
    previous = motion_command.motion.contact_part_mask[prev_steps, :4].to(torch.bool)
    touchdown = current & ~previous
    liftoff = previous & ~current
    return torch.cat([touchdown, liftoff], dim=-1).to(torch.float32)


def a2a_phase_duration(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Normalized phase and duration for the currently sampled motion clip."""
    motion_command = _get_motion_command_and_assert_type(env)
    start_idx = motion_command.motion.motion_start_idx[motion_command.motion_ids]
    end_idx = motion_command.motion.motion_end_idx[motion_command.motion_ids]
    length = (end_idx - start_idx).clamp(min=1)
    phase = (motion_command.time_steps - start_idx).to(torch.float32) / (length - 1).clamp(min=1).to(torch.float32)
    duration = length.to(torch.float32) / float(motion_command.motion.fps)
    return torch.stack([phase, duration], dim=-1)


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)
