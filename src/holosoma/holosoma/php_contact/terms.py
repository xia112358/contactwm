"""PHP manager terms: future contact intentions, privileged tracking and native truth."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms import wbt
from holosoma.php_contact.dataset import load_prepared
from holosoma.utils.wbt_contacts import measured_contacts


class ContactCommand(MotionCommand):
    def setup(self):
        super().setup()
        manifest = load_prepared(self.motion_cfg.motion_manifest)
        by_motion = {str(Path(row['motion_file']).resolve()): row
                     for row in manifest['motion_files']}
        fields = {name: [] for name in ('reference_active', 'target_active', 'target_position_w', 'target_frame', 'segment_id')}
        for index, filename in enumerate(self.motion.motion_files):
            row = by_motion[str(Path(filename).resolve())]
            with np.load(row['contact_file'], allow_pickle=False) as z:
                count = int(self.motion.motion_end_idx[index]-self.motion.motion_start_idx[index])
                if len(z['target_frame']) != count:
                    raise ValueError('Reference interpolation changed the contact clock')
                for name in fields:
                    value = z[name].copy()
                    if name == 'target_frame':
                        value += int(self.motion.motion_start_idx[index])
                    fields[name].append(value)
        self.contact_data = {key: torch.as_tensor(np.concatenate(rows), device=self.device)
                             for key, rows in fields.items()}


def command(env):
    return env.command_manager.get_state('motion_command')


def yaw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def rotate(points, angle):
    while angle.ndim < points.ndim-1:
        angle = angle.unsqueeze(-1)
    x, y, z = points.unbind(-1)
    return torch.stack((angle.cos()*x-angle.sin()*y, angle.sin()*x+angle.cos()*y, z), -1)


def future_contacts(env):
    cmd = command(env)
    root = env.simulator.robot_root_states
    active = cmd.contact_data['target_active'][cmd.time_steps]
    points = cmd.contact_data['target_position_w'][cmd.time_steps]+cmd._reference_env_origins[:, None]
    local = rotate(points-root[:, None, :3], -yaw(root[:, 3:7]))
    return torch.cat((torch.where(active[..., None], local, 0).flatten(1), active.float()), -1)


def goal(env):
    cmd = command(env)
    frames = cmd.contact_data['target_frame'][cmd.time_steps]
    root = env.simulator.robot_root_states
    target = cmd.motion.body_pos_w[frames, 0]+cmd._reference_env_origins
    angle = yaw(root[:, 3:7])
    direction = rotate(target-root[:, :3], -angle)[:, :2]
    direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(0.3)
    heading = yaw(cmd.motion.body_quat_w[frames, 0])-angle
    return torch.cat((direction, torch.atan2(heading.sin(), heading.cos())[:, None]), -1)


def proprioception(env):
    return torch.cat((wbt.base_ang_vel(env), wbt.projected_gravity(env),
                      wbt.dof_pos(env), wbt.dof_vel(env), wbt.actions(env)), -1)


def height_scan(env):
    root = env.simulator.robot_root_states
    xx, yy = torch.meshgrid(torch.linspace(-.4, .4, 9, device=env.device),
                           torch.linspace(-.3, .3, 7, device=env.device), indexing='ij')
    local = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(63, device=env.device)), -1)
    world = rotate(local[None].expand(env.num_envs, -1, -1), yaw(root[:, 3:7]))+root[:, None, :3]
    state = env.terrain_manager.get_state('locomotion_terrain')
    query = getattr(state, 'query_terrain_heights', None)
    if not callable(query):
        raise RuntimeError('PHP contact tracker requires actual terrain height queries')
    height = query(world[..., :2].reshape(-1, 2)).reshape(env.num_envs, 63)
    if not torch.isfinite(height).all():
        raise RuntimeError('PHP contact tracker height scan contains unobserved terrain')
    return height-root[:, 2:3]


def contact_match(env):
    cmd = command(env)
    reference = cmd.contact_data['reference_active'][cmd.time_steps]
    actual = measured_contacts(env)
    known = actual['known']
    # Fixed-q reference activation is a contact intention, not a force target.
    score = (actual['active'] == reference).float()
    env.log_dict['PHPContact/active_fraction'] = actual['active'].float().mean()
    env.log_dict['PHPContact/known_fraction'] = known.float().mean()
    return (score*known).sum(-1)/known.sum(-1).clamp_min(1)


def reference_complete(env):
    cmd = command(env)
    return cmd.time_steps >= cmd.motion.motion_end_idx[cmd.motion_ids]-2
