"""Measured actor observations and explicitly privileged reference streams."""
from __future__ import annotations

import torch

from holosoma.acp.models import contact_schedule
from holosoma.acp.randomization import observation_noise_enabled
from holosoma.managers.observation.terms.wbt import base_ang_vel, base_lin_vel, projected_gravity


def command(env):
    return env.command_manager.get_state("motion_command")


def yaw(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def rotate_yaw(points, angle):
    while angle.ndim < points.ndim - 1:
        angle = angle.unsqueeze(-1)
    x, y, z = points.unbind(-1)
    return torch.stack((angle.cos()*x-angle.sin()*y, angle.sin()*x+angle.cos()*y, z), -1)


def observation_pose(env, measured=False):
    root = env.simulator.robot_root_states
    if (measured and env.acp_settings.get('interface_version', 1) >= 2
            and observation_noise_enabled(env) and getattr(env, 'acp_pose_history', None) is not None):
        history = env.acp_pose_history
        history.update(root, env.acp_sensor_step)
        return history.measured(env.acp_odometry_delay_s, env.acp_odometry_drift)
    return root[:, :3], yaw(root[:, 3:7])


def to_local(env, points, measured=False):
    position, angle = observation_pose(env, measured)
    return rotate_yaw(points-position[:, None], -angle)


def proprio(env):
    actions = env.action_manager.action
    return torch.cat((base_ang_vel(env), projected_gravity(env),
                      env.simulator.dof_pos-env.default_dof_pos, env.simulator.dof_vel, actions), -1)


def terrain(env, rows=17, columns=11, measured=False):
    if getattr(env, "acp_settings", {}).get("terrain_observation", "ground_truth") != "ground_truth":
        from holosoma.acp.perception import fused_scan
        # Fused observations retain the mapping pipeline's explicit odometry
        # contract; never fill unknown cells with privileged terrain.
        return fused_scan(env, rows, columns)
    return ground_truth_terrain(env, rows, columns, measured=measured)


def ground_truth_terrain(env, rows=17, columns=11, measured=False):
    position, angle = observation_pose(env, measured)
    x = torch.linspace(-(rows-1)*0.05, (rows-1)*0.05, rows, device=env.device)
    y = torch.linspace(-(columns-1)*0.05, (columns-1)*0.05, columns, device=env.device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    local = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), -1)
    local = local[None].expand(env.num_envs, -1, -1)
    world = rotate_yaw(local, angle)+position[:, None]
    state = env.terrain_manager.get_state("locomotion_terrain")
    query = getattr(state, "query_terrain_heights", None)
    if not callable(query):
        raise RuntimeError("ACP requires actual initialized terrain height queries")
    heights = query(world[..., :2].reshape(-1, 2)).reshape(env.num_envs, -1)
    valid = torch.isfinite(heights)
    if not valid.any(-1).all():
        raise RuntimeError("ACP height scan has no observed terrain")
    return torch.cat((local[..., :2], torch.where(valid, heights-position[:, 2:3], 0)[..., None]), -1), valid


def goal(env, measured=False):
    cmd = command(env)
    steps = (cmd.time_steps + cmd.acp_goal_lookahead).minimum(cmd.motion.motion_end_idx[cmd.motion_ids]-1)
    target = cmd.motion.body_pos_w[steps, 0]+cmd._reference_env_origins
    target_quat = cmd.motion.body_quat_w[steps, 0]
    position, angle = observation_pose(env, measured)
    delta = rotate_yaw(target-position, -angle)[:, :2]
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(0.3)
    heading = yaw(target_quat)-angle
    return torch.cat((delta, torch.atan2(heading.sin(), heading.cos())[:, None]), -1)


def actor_goal(env):
    value = goal(env, measured=True)
    if observation_noise_enabled(env):
        noise = torch.empty(env.num_envs, 2, device=env.device).uniform_(-0.2, 0.2)
        xy = rotate_yaw(torch.cat((value[:, :2], torch.zeros_like(value[:, :1])), -1), noise[:, 0])[:, :2]
        heading = value[:, 2]+noise[:, 1]
        value = torch.cat((xy, torch.atan2(heading.sin(), heading.cos())[:, None]), -1)
    return value


def contact_targets(env):
    cache = env._holosoma_step_cache
    if "acp_command" not in cache:
        cmd = command(env)
        idx = cmd.time_steps
        target = to_local(env, cmd.acp_target_w[idx]+cmd._reference_env_origins[:, None], measured=True)
        occupancy = cmd.acp_target_occupancy[idx]
        inside = (target[..., 0].abs() <= 0.8) & (target[..., 1].abs() <= 0.5)
        reference = target * (occupancy & inside)[..., None]
        stage = env.acp_settings["stage"]
        predicted = torch.zeros_like(reference)
        if stage > 1 and env.acp_planner is not None:
            scan, valid = terrain(env, measured=True)
            if observation_noise_enabled(env):
                scan = scan.clone()
                scan[..., 2] += torch.randn_like(scan[..., 2])*0.03
            history = history_observation(env)
            with torch.no_grad():
                # An entirely unknown scan issues no contact targets. Do not let
                # attention normalize an all-masked row or invent ground points.
                observed = valid.any(-1)
                if observed.any():
                    predicted[observed] = env.acp_planner(history[observed], actor_goal(env)[observed],
                                                        scan[observed], valid[observed])["positions"]
            probability = contact_schedule(env.acp_iteration, env.acp_settings["input_start"],
                                            env.acp_settings["anneal_steps"])
            if env.is_evaluating:
                probability = 1.0
            use_planner = torch.rand(env.num_envs, 1, 1, device=env.device) < probability
            selected = torch.where(use_planner, predicted, reference)
        else:
            selected = reference
        cache["acp_command"] = (selected, predicted, reference)
    return cache["acp_command"][0].flatten(1)


def history_observation(env):
    """One history update per actual observation; no planner-driven extra samples."""
    return env.acp_history.flatten(1)


def actor_contacts(env):
    contacts = contact_targets(env).reshape(env.num_envs, 9, 3)
    if observation_noise_enabled(env):
        valid = contacts.abs().sum(-1) > 0
        contacts = contacts + (2*torch.rand_like(contacts)-1)*0.1*valid[..., None]
    return contacts.flatten(1)


def tracker_scan(env):
    points, valid = terrain(env, 9, 7, measured=True)
    scan = points[..., 2]
    if observation_noise_enabled(env):
        scan = scan + (2*torch.rand_like(scan)-1)*0.1*valid
    return scan


def critic_reference(env):
    cmd = command(env)
    from holosoma.acp.labels import BODY_NAMES
    ids = [env.simulator.body_names.index(name) for name in BODY_NAMES]
    return torch.cat((cmd.motion.body_lin_vel_w[cmd.time_steps, 0],
                      cmd.motion.body_ang_vel_w[cmd.time_steps, 0], cmd.joint_pos, cmd.joint_vel,
                      cmd.acp_reference_active[cmd.time_steps].float(),
                      to_local(env, cmd.motion.body_pos_w[cmd.time_steps][:, ids]
                               +cmd._reference_env_origins[:, None]).flatten(1)), -1)


def critic_observation(env):
    # The critic retains privileged terrain even when the actor uses a fused
    # scan; v2 removes actor goal/contact commands from its reference input.
    if env.acp_settings.get("terrain_observation", "ground_truth") != "ground_truth":
        scan = ground_truth_terrain(env)
    else:
        scan = terrain(env)
    state = (proprio(env), base_lin_vel(env), scan[0][..., 2])
    if env.acp_settings.get("interface_version", 1) < 2:
        return torch.cat((*state, goal(env), contact_targets(env), critic_reference(env)), -1)
    return torch.cat((*state, critic_reference(env)), -1)
