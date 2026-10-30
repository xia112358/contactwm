"""Table 3 task rewards; contact activation stays native constraint truth."""
from __future__ import annotations

import torch

from holosoma.acp import observations as obs
from holosoma.acp.contacts import measured_contacts
from holosoma.acp.labels import BODY_NAMES
from holosoma.acp.models import contact_schedule
from holosoma.utils.rotations import quat_error_magnitude


def task(env):
    cmd = obs.command(env)
    root = env.simulator.robot_root_states[:]
    index = cmd.time_steps
    reference_pos = cmd.motion.body_pos_w[index, 0]+cmd._reference_env_origins
    reference_quat = cmd.motion.body_quat_w[index, 0]
    reference_v = cmd.motion.body_lin_vel_w[index, 0]
    reference_w = cmd.motion.body_ang_vel_w[index, 0]
    moving = (reference_v.norm(dim=-1) > 0.1).float()
    position_error = (root[:, :3]-reference_pos).square().sum(-1)
    orientation_error = quat_error_magnitude(root[:, 3:7], reference_quat, w_last=True).square()
    base_position = torch.exp(-moving*position_error/0.5**2-orientation_error)
    velocity = torch.exp(-(root[:, 7:10]-reference_v).square().sum(-1)
                         -(root[:, 10:13]-reference_w).square().sum(-1)/3**2)
    q_error = env.simulator.dof_pos-cmd.joint_pos
    q_error = torch.atan2(q_error.sin(), q_error.cos())
    joints = torch.exp(-q_error.square().sum(-1)/2**2)
    joint_velocity = torch.exp(-(env.simulator.dof_vel-cmd.joint_vel).square().sum(-1)/10**2)
    actual = measured_contacts(env)
    reference_active = cmd.acp_reference_active[index]
    loaded = actual["active"] & (actual["force_w"][..., 2].abs() > 1)
    contact_match = (loaded.float()*(2*reference_active.float()-1)*actual["known"]).mean(-1)
    ids = [env.simulator.body_names.index(name) for name in BODY_NAMES]
    positions = env.simulator._robot.data.body_pos_w.torch[:, ids]
    # Eq. Table 3 tracks all nine contact-capable body positions even when
    # inactive. Occupancy gates planner targets, not this kinematic prior.
    reference_contact = cmd.motion.body_pos_w[index][:, ids]+cmd._reference_env_origins[:, None]
    global_error = (positions-reference_contact).square().sum((1, 2))
    local_positions = obs.to_local(env, positions)
    reference_local = obs.rotate_yaw(reference_contact-reference_pos[:, None],
                                     -obs.yaw(reference_quat))
    base_error = (local_positions-reference_local).square().sum((1, 2))
    eta = 0.0
    if env.acp_settings["stage"] > 1:
        eta = contact_schedule(env.acp_iteration, env.acp_settings["reward_start"], env.acp_settings["anneal_steps"])
    planner_reward = torch.ones(env.num_envs, device=env.device)
    if hasattr(env, "acp_issued_plan_w"):
        mask = env.acp_issued_plan_valid
        error = ((positions-env.acp_issued_plan_w).square().sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        planner_reward = torch.exp(-error)
    result = 3*base_position+velocity+2.5*joints+0.5*joint_velocity
    result += (1-eta)*(5*contact_match+2*torch.exp(-base_error)+torch.exp(-moving*global_error))
    result += 1.5*eta*planner_reward
    env.log_dict["ACP/contact_match"] = contact_match.mean().detach()
    env.log_dict["ACP/reward_eta"] = torch.tensor(eta, device=env.device)
    return result


def action_jerk(env):
    return (env.action_manager.action-2*env.acp_previous_actions+env.acp_second_previous_actions).square().sum(-1)


def torque_limits(env):
    """Table 3 difference between the command before and after limiting."""
    term = env.action_manager.get_term("joint_control")
    if not hasattr(term, "computed_torques"):
        raise RuntimeError("ACP torque regularization requires the pre-limit PD command")
    return (term.computed_torques-term.torques).abs().sum(-1)


def joint_position_limits(env):
    from holosoma.managers.reward.terms.wbt import limits_dof_pos
    return limits_dof_pos(env, soft_dof_pos_limit=0.9).clamp_max(30)


def joint_velocity_limits(env):
    return (env.simulator.dof_vel.abs()-0.9*env.dof_vel_limits).clamp(0, 1).sum(-1)


def contact_body_velocities(env):
    ids = [env.simulator.body_names.index(name) for name in BODY_NAMES]
    return env.simulator._robot.data.body_lin_vel_w.torch[:, ids]


def contact_acceleration(env):
    acceleration = (contact_body_velocities(env)-env.acp_previous_body_velocity)/env.dt
    value = acceleration.norm(dim=-1).sum(-1).clamp_max(1e4)
    return value*env.acp_body_velocity_ready
