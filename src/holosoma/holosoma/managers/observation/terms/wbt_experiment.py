"""Separate measured student observations from privileged reference inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from holosoma.managers.observation.terms import wbt
from holosoma.utils.rotations import quat_rotate_inverse, subtract_frame_transforms, yaw_quat, quaternion_to_matrix
from holosoma.utils.wbt_experiment import load_contact_reference


def _command(env):
    return env.command_manager.get_state("motion_command")


def velocity_command(env, command: tuple[float, ...] = (1., 0.), source: str = "constant"):
    """High-level task command; reference poses, force targets and phase are hidden.

    PHP's released vel_cmd is a 15-way skill/direction command. Dataset mode
    replays that explicit command stream, not reference q/qdot or a pose goal.
    Constant mode permits a caller to hold a manually supplied command.
    """
    if source == "constant":
        return torch.tensor(command, device=env.device).expand(env.num_envs, -1)
    if source != "dataset":
        raise ValueError("Task command source must be constant or dataset")
    data = getattr(env, "_wbt_task_command_stream", None)
    if data is None:
        rows = []
        for filename in _command(env).motion.motion_files:
            with np.load(filename, allow_pickle=False) as bundle:
                if "vel_cmd" not in bundle:
                    raise ValueError(f"{filename}: no task command stream; choose constant mode explicitly")
                values = bundle["vel_cmd"].copy()
                if values.ndim != 2 or len(values) != len(bundle["joint_pos"]) or not np.isfinite(values).all():
                    raise ValueError(f"{filename}: invalid task command stream")
                rows.append(torch.as_tensor(values, device=env.device))
        if len({r.shape[1] for r in rows}) != 1:
            raise ValueError("Task command widths differ between motions")
        data = env._wbt_task_command_stream = torch.cat(rows)
    return data[_command(env).time_steps]


def force_in_torso_yaw_frame(force, quaternion):
    orientation = yaw_quat(quaternion, w_last=True)[:, None].expand(-1, force.shape[1], -1)
    return quat_rotate_inverse(orientation.reshape(-1, 4), force.reshape(-1, 3), w_last=True).reshape_as(force)


def actual_contacts(env, mode: str = "state_force", force_scale: float = 300.):
    from holosoma.utils.wbt_contacts import measured_contacts
    if mode not in {"state", "force", "state_force"} or force_scale <= 0:
        raise ValueError("Expected measured contact mode state/force/state_force and positive scale")
    data = measured_contacts(env)
    fields = []
    if mode in {"state", "state_force"}:
        fields.append(data["active"].float())
    if mode in {"force", "state_force"}:
        robot = _command(env)
        force_b = force_in_torso_yaw_frame(data["force_w"], robot.robot_ref_quat_w)
        fields.append((force_b / force_scale).flatten(1))
    fields.append(data["known"].float())
    return torch.cat(fields, dim=-1)


def reference_contacts(env, offsets=(0,)):
    command = _command(env)
    store = getattr(env, "_wbt_reference_contacts", None)
    if store is None:
        active, force = [], []
        for filename in command.motion.motion_files:
            a, f, _ = load_contact_reference(filename, collision_profile=getattr(env.robot_config.asset, "collision_profile", "legacy_v1"))
            with np.load(filename, allow_pickle=False) as z:
                if len(a) != len(z["joint_pos"]):
                    raise ValueError(f"{filename}: contact and trajectory frame counts differ")
            active.append(torch.as_tensor(a, device=env.device))
            force.append(torch.as_tensor(f, device=env.device))
        store = env._wbt_reference_contacts = (torch.cat(active), torch.cat(force))
    ids = command.motion_ids
    start, end = command.motion.motion_start_idx[ids], command.motion.motion_end_idx[ids]
    steps = command.time_steps[:, None] + torch.as_tensor(offsets, device=env.device)[None]
    steps = steps.maximum(start[:, None]).minimum(end[:, None]-1)
    return store[0][steps], store[1][steps]


def reference_force(env, force_scale: float = 300.):
    _, force = reference_contacts(env)
    return (force_in_torso_yaw_frame(force[:, 0], _command(env).robot_ref_quat_w) / force_scale).flatten(1)


def sparse_reference(env, offsets=(0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50), add_noise=False):
    """Torso + six endpoint poses and native contact bits; no reference q or qdot."""
    command = _command(env)
    names = ("torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link", "right_knee_link")
    selected = [command.motion_cfg.body_names_to_track.index(name) for name in names]
    positions, rotations = command.future_body_pos_quat_offsets(tuple(offsets))
    positions, rotations = positions[:, :, selected], rotations[:, :, selected]
    origin = command.robot_ref_pos_w[:, None, None].expand_as(positions)
    rotation = yaw_quat(command.robot_ref_quat_w, w_last=True)[:, None, None].expand_as(rotations)
    p, q = subtract_frame_transforms(origin.reshape(-1, 3), rotation.reshape(-1, 4),
        positions.reshape(-1, 3), rotations.reshape(-1, 4))
    p = p.reshape(env.num_envs, len(offsets), 7, 3)
    r = quaternion_to_matrix(q, w_last=True)[..., :2].reshape(env.num_envs, len(offsets), 7, 6)
    if add_noise:
        p = p + (torch.rand_like(p)*2-1)*.02
        r = r + (torch.rand_like(r)*2-1)*.03
    active, _ = reference_contacts(env, offsets)
    return torch.cat((p[:, :, 0], r[:, :, 0], p[:, :, 1:].flatten(2),
                      r[:, :, 1:].flatten(2), active.float()), dim=-1).flatten(1)


def route_table(env, manifests):
    cache = getattr(env, "_wbt_route_tables", None)
    if cache is None:
        cache = env._wbt_route_tables = {}
    key = tuple(manifests)
    if key not in cache:
        from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
        owners = {}
        for index, filename in enumerate(manifests):
            for row in load_motion_terrain_manifest(filename)["motion_files"]:
                path = str(Path(row["motion_file"]).resolve())
                if path in owners:
                    raise ValueError(f"Motion has multiple experts: {path}")
                owners[path] = index
        cache[key] = torch.tensor([owners[str(Path(p).resolve())] for p in _command(env).motion.motion_files],
            device=env.device, dtype=torch.long)
    return cache[key]


def teacher_route(env, manifests):
    return route_table(env, manifests)[_command(env).motion_ids, None].float()
