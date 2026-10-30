"""Native terrain-ray adapter for ACP mapping (synthetic LiDAR, exact odometry).

This is an instantaneous panoramic terrain scan, not a MID-360 driver or DLIO.
Terrain occlusion is ray traced; robot self-occlusion and motion deskew are not
simulated. Real scans must be deskewed before MappingPipeline.integrate.
"""
from __future__ import annotations

import math

import torch

from holosoma.acp.mapping import MappingConfig, MappingPipeline, compose_rotation, load_mapping_network, rotate


class ACPPerception:
    def __init__(self, env):
        self.env = env
        settings = env.acp_settings
        self.mode = settings.get("terrain_observation", "ground_truth")
        if self.mode not in ("fused_raw", "fused_neural"):
            raise ValueError(f"Unsupported fused observation mode: {self.mode}")
        self.period = 1.0/settings.get("lidar_hz", 10.0)
        self.ray_count = settings.get("lidar_points", 20_000)
        self.max_range = settings.get("lidar_max_range", 20.0)
        self.offset = torch.tensor(settings.get("lidar_offset", (0.0, 0.0, 0.35)), device=env.device)
        self.mount_rotation = torch.tensor(settings.get("lidar_rotation_xyzw", (1.0, 0.0, 0.0, 0.0)), device=env.device)
        self.body_name = settings.get("lidar_body", "torso_link")
        if self.body_name not in env.simulator.body_names:
            raise ValueError(f"LiDAR mount body {self.body_name!r} is absent from the canonical articulation")
        self.body_id = env.simulator.body_names.index(self.body_name)
        network = None
        if self.mode == "fused_neural":
            checkpoint = settings.get("mapping_checkpoint")
            if not checkpoint:
                raise ValueError("fused_neural requires --mapping-checkpoint with trained mapping weights")
            network = load_mapping_network(checkpoint, env.device)
        self.pipeline = MappingPipeline(env.num_envs, env.device,
            MappingConfig(max_age=settings.get("mapping_max_age", 10.0)), network)
        self.last_scan = torch.full((env.num_envs,), -torch.inf, device=env.device)
        self.latest_scan = None

    def reset(self, env_ids):
        self.pipeline.map.reset(env_ids)
        self.last_scan[env_ids] = -torch.inf

    @torch.no_grad()
    def update(self, timestamp):
        env = self.env
        due = timestamp-self.last_scan >= self.period-1e-6
        if not due.any():
            return
        from holosoma.utils.warp_utils import ray_cast

        terrain = env.terrain_manager.get_state("locomotion_terrain")
        if getattr(terrain, "warp_mesh", None) is None:
            raise RuntimeError("Fused ACP perception requires the initialized terrain ray mesh")
        position = env.simulator._rigid_body_pos[:, self.body_id]
        body_orientation = env.simulator._rigid_body_rot[:, self.body_id]  # Native Newton xyzw.
        sensor = position+rotate(self.offset.expand_as(position), body_orientation)
        orientation = compose_rotation(body_orientation, self.mount_rotation)
        # Quasi-uniform panoramic rays. The rotating pattern accumulates new hits
        # across frames; this is a configurable synthetic approximation to LiDAR.
        index = torch.arange(self.ray_count, device=env.device, dtype=torch.float32)
        azimuth = index*2.399963229728653 + float(timestamp)*0.37
        low, high = math.sin(math.radians(-7)), math.sin(math.radians(52))
        z = low+(high-low)*(index+0.5)/self.ray_count
        radius = (1-z.square()).sqrt()
        directions = torch.stack((radius*azimuth.cos(), radius*azimuth.sin(), z), -1)
        directions = directions[None].expand(env.num_envs, -1, -1)
        world_directions = rotate(directions, orientation)
        points_w = torch.full_like(world_directions, torch.inf)
        ids = due.nonzero(as_tuple=False).flatten()
        # Limit temporary ray buffers when many environments reset together.
        for chunk in ids.split(8):
            starts = sensor[chunk, None].expand(-1, self.ray_count, -1).contiguous()
            hits = ray_cast(starts, world_directions[chunk].contiguous(), terrain.warp_mesh)
            points_w[chunk] = hits.to(env.device)
        distance = (points_w-sensor[:, None]).norm(dim=-1)
        valid = torch.isfinite(points_w).all(-1) & (distance <= self.max_range) & (distance > 0.05)
        delta = (points_w-sensor[:, None]).masked_fill(~valid[..., None], 0)
        inverse = orientation.clone()
        inverse[:, :3] *= -1
        points_sensor = rotate(delta, inverse)
        # Undue environments retain their prior timestamp, so fusion ignores them.
        # On the first call all environments are due and timestamps are finite.
        time = torch.where(due, torch.full_like(self.last_scan, timestamp), self.last_scan)
        self.pipeline.integrate(points_sensor, valid, sensor, orientation, time)
        self.last_scan.copy_(time)
        root = env.simulator.robot_root_states
        self.latest_scan = {"points_sensor": points_sensor, "valid": valid, "position_w": sensor,
            "quaternion_xyzw": orientation, "timestamps": time.clone(), "sampled_envs": due,
            "root_position_w": root[:, :3].clone(), "root_quaternion_xyzw": root[:, 3:7].clone()}

    def scan(self, timestamp, rows=17, columns=11):
        self.update(timestamp)
        root = self.env.simulator.robot_root_states
        return self.pipeline.scan(root[:, :3], root[:, 3:7], timestamp, rows, columns)


def fused_scan(env, rows=17, columns=11):
    if getattr(env, "acp_perception", None) is None:
        env.acp_perception = ACPPerception(env)
    points, valid = env.acp_perception.scan(env.acp_sensor_time, rows, columns)
    if hasattr(env, "log_dict"):
        env.log_dict[f"ACP/map_scan_{rows}x{columns}_coverage"] = valid.float().mean()
    return points, valid
