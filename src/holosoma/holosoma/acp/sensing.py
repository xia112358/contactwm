"""Control-rate pose history for synthetic odometry delay and drift."""
from __future__ import annotations

import math

import torch


class PoseHistory:
    def __init__(self, num_envs, dt, device):
        self.dt = float(dt)
        self.pose = torch.zeros(num_envs, math.ceil(0.1/self.dt)+2, 4, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.step = None

    def update(self, root_state, step):
        qx, qy, qz, qw = root_state[:, 3:7].unbind(-1)
        yaw = torch.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        current = torch.cat((root_state[:, :3], yaw[:, None]), -1)
        if step != self.step:
            self.pose[:, 1:] = self.pose[:, :-1].clone()
            self.pose[:, 0] = current
            self.step = step
        fresh = ~self.ready
        self.pose[fresh] = current[fresh, None]
        self.ready[fresh] = True

    def measured(self, delay_s, drift):
        steps = delay_s/self.dt
        low = steps.floor().long().clamp(0, self.pose.shape[1]-2)
        blend = (steps-low).clamp(0, 1)[:, None]
        ids = torch.arange(len(low), device=self.pose.device)
        first, second = self.pose[ids, low], self.pose[ids, low+1]
        xyz = first[:, :3]+blend*(second[:, :3]-first[:, :3])+drift
        delta = second[:, 3]-first[:, 3]
        angle = first[:, 3]+blend[:, 0]*torch.atan2(delta.sin(), delta.cos())
        return xyz, angle
