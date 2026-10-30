"""Dual-task rewards for the trimmed climb00 contact-goal experiment."""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.climb_reference_goal import ClimbReferenceGoalCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.reward.terms.wbt import (
    DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
    _part_contact_force_history,
    motion_global_body_ang_vel,
    motion_global_body_lin_vel,
    motion_global_ref_orientation_error_exp,
    motion_global_ref_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_inverse


def _command(env) -> ClimbReferenceGoalCommand:
    command = env.command_manager.get_state("motion_command")
    if not isinstance(command, ClimbReferenceGoalCommand):
        raise TypeError(f"Expected ClimbReferenceGoalCommand, got {type(command)}")
    return command


def climb_survival(env) -> torch.Tensor:
    """Shared survival reward used by both task distributions."""
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def _sparse_climb_contact_force_magnitude(env) -> torch.Tensor:
    """Six endpoint forces derived from the cached canonical eight-part view."""
    canonical = _part_contact_force_history(
        env,
        DEFAULT_CONTACT_FORCE_PART_BODY_NAMES,
        force_reduce="sum",
    )
    endpoint_history = torch.stack(
        (
            canonical[:, :, 0] + canonical[:, :, 1],
            canonical[:, :, 2] + canonical[:, :, 3],
            canonical[:, :, 4],
            canonical[:, :, 5],
            canonical[:, :, 6],
            canonical[:, :, 7],
        ),
        dim=2,
    )
    return torch.linalg.vector_norm(endpoint_history, dim=-1).max(dim=1).values


def climb_imitation_tracking(
    env,
    root_position_sigma: float = 0.30,
    root_orientation_sigma: float = 0.40,
    body_position_sigma: float = 0.30,
    body_orientation_sigma: float = 0.40,
    linear_velocity_sigma: float = 1.0,
    angular_velocity_sigma: float = 3.14,
    joint_position_sigma: float | None = None,
    projected_gravity_sigma: float | None = None,
) -> torch.Tensor:
    """Dense reference tracking, enabled only for imitation episodes."""
    command = _command(env)
    score_terms = [
        motion_global_ref_position_error_exp(env, root_position_sigma),
        motion_global_ref_orientation_error_exp(env, root_orientation_sigma),
        motion_relative_body_position_error_exp(env, body_position_sigma),
        motion_relative_body_orientation_error_exp(env, body_orientation_sigma),
        motion_global_body_lin_vel(env, linear_velocity_sigma),
        motion_global_body_ang_vel(env, angular_velocity_sigma),
    ]
    if joint_position_sigma is not None:
        joint_error = torch.square(command.joint_pos - env.simulator.dof_pos).mean(dim=-1)
        score_terms.append(torch.exp(-joint_error / float(joint_position_sigma) ** 2))
    if projected_gravity_sigma is not None:
        gravity = gravity_vector(env)
        ref_gravity = quat_rotate_inverse(command.ref_quat_w, gravity, w_last=True)
        robot_gravity = quat_rotate_inverse(command.robot_ref_quat_w, gravity, w_last=True)
        gravity_error = torch.square(ref_gravity - robot_gravity).sum(dim=-1)
        score_terms.append(torch.exp(-gravity_error / float(projected_gravity_sigma) ** 2))
    scores = torch.stack(score_terms, dim=-1)
    tracking_score = scores.mean(dim=-1)
    env.log_dict["climb_reference_goal/tracking_score"] = tracking_score.detach()
    curriculum_manager = getattr(env, "curriculum_manager", None)
    curriculum = (
        curriculum_manager.get_term("imitation_tracking_curriculum")
        if curriculum_manager is not None
        else None
    )
    observe = getattr(curriculum, "observe_tracking_score", None)
    if callable(observe):
        observe(tracking_score)
    return tracking_score * command.is_imitation.to(scores.dtype)


def climb_imitation_base_height_error(env) -> torch.Tensor:
    """Absolute reference base-height error, restricted to imitation samples."""
    command = _command(env)
    error = torch.abs(command.robot_ref_pos_w[:, 2] - command.ref_pos_w[:, 2])
    return error * command.is_imitation.to(error.dtype)


def climb_generalization_position_error(env) -> torch.Tensor:
    """Horizontal root-goal distance for generalization samples."""
    command = _command(env)
    generalization = ~command.is_imitation
    error = torch.linalg.vector_norm(
        command.robot_ref_pos_w[:, :2] - command.goal_ref_pos_w[:, :2], dim=-1
    )
    env.log_dict["climb_reference_goal/goal_position_error"] = error.detach()
    return error * generalization.to(error.dtype)


def climb_generalization_orientation_error(env) -> torch.Tensor:
    """Root orientation error for generalization samples."""
    command = _command(env)
    generalization = ~command.is_imitation
    error = quat_error_magnitude(
        command.robot_ref_quat_w,
        command.goal_ref_quat_w,
        w_last=True,
    )
    env.log_dict["climb_reference_goal/goal_orientation_error"] = error.detach()
    return error * generalization.to(error.dtype)


def climb_generalization_reach(
    env,
    position_threshold: float = 0.20,
    orientation_threshold: float = 0.50,
) -> torch.Tensor:
    """Paper-style constant reach reward inside the target root-pose region."""
    command = _command(env)
    generalization = ~command.is_imitation
    position_error = torch.linalg.vector_norm(
        command.robot_ref_pos_w[:, :2] - command.goal_ref_pos_w[:, :2], dim=-1
    )
    orientation_error = quat_error_magnitude(
        command.robot_ref_quat_w,
        command.goal_ref_quat_w,
        w_last=True,
    )
    reached = (
        generalization
        & (position_error <= float(position_threshold))
        & (orientation_error <= float(orientation_threshold))
    )
    env.log_dict["climb_reference_goal/generalization_fraction"] = generalization.to(
        torch.float32
    ).detach()
    env.log_dict["climb_reference_goal/success"] = reached.to(torch.float32).detach()
    return reached.to(torch.float32)


class ClimbSparseGoalSuccess(RewardTermBase):
    """Sparse terminal goal reward for generalization episodes.

    No progress, reference pose, phase, or future-frame shaping is used.  A
    success requires torso arrival, the requested contact topology, active
    endpoint placement, and a short stability window.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stable_frames = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.stable_frames.zero_()
        else:
            self.stable_frames[env_ids] = 0

    def __call__(
        self,
        env,
        *,
        torso_radius: float = 0.20,
        orientation_threshold: float = 0.50,
        endpoint_radius: float = 0.12,
        root_speed_threshold: float = 0.35,
        contact_threshold: float = 10.0,
        stable_frames_required: int = 3,
    ) -> torch.Tensor:
        command = _command(env)
        generalization = ~command.is_imitation
        torso_error = torch.linalg.vector_norm(command.robot_ref_pos_w - command.goal_ref_pos_w, dim=-1)
        torso_ok = torso_error <= float(torso_radius)
        orientation_error = quat_error_magnitude(
            command.robot_ref_quat_w,
            command.goal_ref_quat_w,
            w_last=True,
        )
        orientation_ok = orientation_error <= float(orientation_threshold)

        endpoint_pos = command.robot_body_pos_w[:, command.goal_endpoint_track_indexes]
        endpoint_error = torch.linalg.vector_norm(endpoint_pos - command.goal_endpoint_pos_w, dim=-1)
        expected = command.goal_contact_mask
        expected_count = expected.sum(dim=-1).clamp(min=1)
        endpoint_ok = (
            ((endpoint_error <= float(endpoint_radius)) | ~expected).sum(dim=-1)
            == expected.shape[-1]
        )

        force = _sparse_climb_contact_force_magnitude(env)
        actual = force > float(contact_threshold)
        contact_ok = torch.all(actual == expected, dim=-1)
        slow = torch.linalg.vector_norm(command.robot_root_lin_vel_w, dim=-1) <= float(root_speed_threshold)
        stable = torso_ok & orientation_ok & endpoint_ok & contact_ok & slow
        self.stable_frames[:] = torch.where(stable, self.stable_frames + 1, torch.zeros_like(self.stable_frames))
        task_success = self.stable_frames >= max(int(stable_frames_required), 1)
        reward_success = task_success & generalization

        curriculum_manager = getattr(env, "curriculum_manager", None)
        curriculum = (
            curriculum_manager.get_term("reference_goal_curriculum")
            if curriculum_manager is not None
            else None
        )
        observe = getattr(curriculum, "observe_task_success", None)
        if callable(observe):
            observe(task_success)

        env.log_dict["climb_reference_goal/generalization_fraction"] = generalization.to(torch.float32).detach()
        env.log_dict["climb_reference_goal/torso_error"] = torso_error.detach()
        env.log_dict["climb_reference_goal/goal_orientation_error"] = orientation_error.detach()
        env.log_dict["climb_reference_goal/contact_match"] = contact_ok.to(torch.float32).detach()
        env.log_dict["climb_reference_goal/endpoint_match"] = endpoint_ok.to(torch.float32).detach()
        env.log_dict["climb_reference_goal/task_success"] = task_success.to(torch.float32).detach()
        env.log_dict["climb_reference_goal/success"] = reward_success.to(torch.float32).detach()
        env.log_dict["climb_reference_goal/expected_contact_count"] = expected_count.to(torch.float32).detach()
        return reward_success.to(torch.float32)
