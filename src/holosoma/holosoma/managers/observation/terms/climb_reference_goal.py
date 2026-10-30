"""Observations for the climb00 reference-and-goal experiment."""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.climb_reference_goal import ClimbReferenceGoalCommand
from holosoma.managers.reward.terms.wbt import (
    DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
    _part_contact_force_magnitude,
)
from holosoma.utils.rotations import quat_inverse, quat_mul, quat_rotate_inverse, yaw_quat


def _command(env) -> ClimbReferenceGoalCommand:
    command = env.command_manager.get_state("motion_command")
    if not isinstance(command, ClimbReferenceGoalCommand):
        raise TypeError(f"Expected ClimbReferenceGoalCommand, got {type(command)}")
    return command


def climb_contact_goal(env) -> torch.Tensor:
    """Task goal in the measured torso-yaw frame, with no phase or event ID.

    Layout: torso xyz, six endpoint xyz, touchdown bits, end-contact bits.
    """
    command = _command(env)
    torso_pos = command.robot_ref_pos_w
    torso_yaw = yaw_quat(command.robot_ref_quat_w, w_last=True)
    torso_delta = quat_rotate_inverse(torso_yaw, command.goal_ref_pos_w - torso_pos, w_last=True)
    endpoint_delta_w = command.goal_endpoint_pos_w - torso_pos[:, None, :]
    expanded_yaw = torso_yaw[:, None, :].expand(-1, endpoint_delta_w.shape[1], -1)
    endpoint_delta = quat_rotate_inverse(
        expanded_yaw.reshape(-1, 4),
        endpoint_delta_w.reshape(-1, 3),
        w_last=True,
    ).reshape(env.num_envs, -1)
    return torch.cat(
        (
            torso_delta,
            endpoint_delta,
            command.goal_touchdown_mask.to(torch.float32),
            command.goal_contact_mask.to(torch.float32),
        ),
        dim=-1,
    )


def climb_root_pose_goal(env) -> torch.Tensor:
    """Paper-style low-dimensional root goal with no contact or phase label.

    Layout: target root ``xy`` in the measured torso-yaw frame, followed by
    the target orientation relative to that frame as an ``xyzw`` quaternion.
    Contact-event cuts select goal poses, but their IDs and contact masks are
    deliberately hidden from the actor.
    """
    command = _command(env)
    torso_yaw = yaw_quat(command.robot_ref_quat_w, w_last=True)
    delta_w = command.goal_ref_pos_w - command.robot_ref_pos_w
    delta_b = quat_rotate_inverse(torso_yaw, delta_w, w_last=True)
    relative_goal_quat = quat_mul(
        quat_inverse(torso_yaw, w_last=True),
        command.goal_ref_quat_w,
        w_last=True,
    )
    # q and -q encode the same rotation.  Canonicalizing the sign prevents an
    # observation discontinuity when the reference crosses that representation.
    sign = torch.where(relative_goal_quat[:, 3:4] < 0.0, -1.0, 1.0)
    return torch.cat((delta_b[:, :2], relative_goal_quat * sign), dim=-1)


def climb_task_indicator(env) -> torch.Tensor:
    """Critic-only one-hot indicator: imitation, generalization."""
    imitation = _command(env).is_imitation.to(torch.float32)
    return torch.stack((imitation, 1.0 - imitation), dim=-1)


def climb_assistive_wrench(env, force_scale: float = 350.0, torque_scale: float = 100.0) -> torch.Tensor:
    """Critic-only privileged assistive base wrench."""
    wrench = _command(env).assistive_wrench_w
    scale = torch.tensor(
        (force_scale, force_scale, force_scale, torque_scale, torque_scale, torque_scale),
        dtype=torch.float32,
        device=env.device,
    )
    return wrench / scale.clamp_min(1.0e-6)


def climb_actual_contact_force(env, force_scale: float = 300.0) -> torch.Tensor:
    """Critic-only measured force magnitudes for the canonical eight parts."""
    force = _part_contact_force_magnitude(
        env,
        DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
        force_reduce="sum",
        history_reduce="max",
    )
    return force / max(float(force_scale), 1.0e-6)
