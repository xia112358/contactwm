"""Raster terrain sensing. Network inputs contain heights, never box parameters."""
from __future__ import annotations

import torch


class OriginalContext:
    """Task pairing only; no terrain or contact features enter any network."""
    height_dim = dim = contact_dim = 0
    contact_slice = slice(33, 33)
    schema = "supermimic_original_proprioception_v1"

    def __init__(self, tasks):
        self.count = len(tasks)
        self.tasks = tasks


class HeightContactContext:
    height_dim = 121
    dim = 123
    task_column = 33
    contact_slice = slice(34, 36)
    contact_dim = 2
    schema = "local_heightmap_11x11_contact2_v1"

    def __init__(self, tasks):
        from .rotations import quat_to_mat
        self.count = len(tasks)
        self.tasks = tasks
        self.resolution = 0.05
        # A task-wide raster is sampled into a heading-aligned local patch.
        all_boxes = [box for task in tasks for box in task["boxes"]]
        points = torch.tensor([box[:2] for box in all_boxes], dtype=torch.float32)
        radii = torch.tensor([box[7:9] for box in all_boxes], dtype=torch.float32).norm(dim=-1).max()
        self.minimum = (points.amin(0) - radii - 3).floor()
        maximum = (points.amax(0) + radii + 3).ceil()
        size = ((maximum - self.minimum) / self.resolution).ceil().long() + 1
        self.width, self.height = int(size[0]), int(size[1])
        yy, xx = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing="ij")
        xy = torch.stack((xx, yy), -1) * self.resolution + self.minimum
        atlas = torch.zeros(self.count, self.height, self.width)
        for index, task in enumerate(tasks):
            for box in task["boxes"]:
                box = torch.tensor(box, dtype=torch.float32)
                rotation = quat_to_mat(box[3:7])
                if not torch.allclose(rotation[2], torch.tensor([0., 0., 1.]), atol=1e-5):
                    raise ValueError("Height rasterization requires the published upright low-step boxes")
                local = (xy - box[:2]) @ rotation[:2, :2]
                inside = (local.abs() <= box[7:9] * 0.5).all(-1)
                top = box[2] + box[9] * 0.5
                atlas[index] = torch.where(inside, torch.maximum(atlas[index], top), atlas[index])
        local_y, local_x = torch.meshgrid(torch.linspace(-0.8, 0.8, 11),
                                        torch.linspace(-0.5, 1.5, 11), indexing="ij")
        self.local = torch.stack((local_x, local_y, torch.zeros_like(local_x)), -1).reshape(-1, 3)
        self.atlas = atlas.flatten()
        self.cache = {}

    def heights(self, state, task_ids):
        from .rotations import yaw_mat
        position, rotation = state[:2]
        key = str(position.device)
        if key not in self.cache:
            self.cache[key] = (self.atlas.to(position.device), self.minimum.to(position.device),
                               self.local.to(position.device))
        atlas, minimum, local = self.cache[key]
        world = (yaw_mat(rotation).unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1)
        world = world + position.unsqueeze(-2)
        grid = (world[..., :2] - minimum) / self.resolution
        lower = grid.floor().long()
        fraction = grid - lower
        ix, iy = lower.unbind(-1)
        wx, wy = fraction.unbind(-1)
        base = task_ids.long().unsqueeze(-1) * self.width * self.height

        def gather(dx, dy):
            x, y = ix + dx, iy + dy
            valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            index = base + y.clamp(0, self.height - 1) * self.width + x.clamp(0, self.width - 1)
            return atlas[index] * valid

        height = ((1-wx)*(1-wy)*gather(0, 0) + wx*(1-wy)*gather(1, 0)
                  + (1-wx)*wy*gather(0, 1) + wx*wy*gather(1, 1))
        return (height - position[..., 2:3]).clamp(-2, 2)

    def features(self, state, xi):
        return torch.cat((self.heights(state, xi[..., self.task_column]), xi[..., self.contact_slice]), -1)


class HeightForceContext(HeightContactContext):
    """Six measured world-force vectors; normalized heading-frame network features."""
    dim = 139
    contact_dim = 18
    contact_slice = slice(34, 52)
    schema = "local_heightmap_11x11_force6x3_heading_v1"
    force_scale = 200.0  # Newtons; xi and replay store world forces divided by this scale.
    parts = ("left_foot", "right_foot", "left_hand", "right_hand", "left_knee", "right_knee")

    @staticmethod
    def local_forces(state, world_forces):
        from .rotations import yaw_mat
        inverse_heading = yaw_mat(state[1]).transpose(-1, -2).unsqueeze(-3)
        vectors = world_forces.reshape(*world_forces.shape[:-1], 6, 3)
        return (inverse_heading @ vectors.unsqueeze(-1)).squeeze(-1).flatten(-2)

    @staticmethod
    def world_forces(state, local_forces):
        from .rotations import yaw_mat
        heading = yaw_mat(state[1]).unsqueeze(-3)
        vectors = local_forces.reshape(*local_forces.shape[:-1], 6, 3)
        return (heading @ vectors.unsqueeze(-1)).squeeze(-1).flatten(-2)

    def features(self, state, xi):
        return torch.cat((self.heights(state, xi[..., self.task_column]),
                          self.local_forces(state, xi[..., self.contact_slice])), -1)


class HeightOnlyContext(HeightContactContext):
    dim = 121
    contact_dim = 0
    contact_slice = slice(34, 34)
    schema = "local_heightmap_11x11_without_contact_v1"

    def features(self, state, xi):
        return self.heights(state, xi[..., self.task_column])


class ForceOnlyContext(HeightForceContext):
    height_dim = 0
    dim = 18
    schema = "force6x3_heading_without_heightmap_v1"

    def __init__(self, tasks):
        # No terrain raster is constructed or queried in this ablation.
        self.count = len(tasks)
        self.tasks = tasks

    def features(self, state, xi):
        return self.local_forces(state, xi[..., self.contact_slice])


CONTEXT_TYPES = {"original": OriginalContext, "binary2": HeightContactContext,
                 "force6": HeightForceContext, "height_only": HeightOnlyContext,
                 "force_only": ForceOnlyContext}
