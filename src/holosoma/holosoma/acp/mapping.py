"""History-aware elevation mapping, independent of Isaac/robot dependencies.

The three fill modes, grid geometry and overhang prior follow ACP Fig. 8.
CNN widths, uncertainty formula and precision fusion are explicit local choices:
the paper does not specify enough detail to reproduce these numerically.
All quaternions are xyzw, distances metres, timestamps seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MAPPING_SCHEMA = "somaforge_acp_mapping_v1"


@dataclass(frozen=True)
class MappingConfig:
    resolution: float = 0.03
    local_rows: int = 200
    local_columns: int = 100
    map_size: int = 256
    max_age: float = 10.0
    measurement_std: float = 0.03
    process_variance_per_second: float = 0.0025

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.resolution, self.max_age, self.measurement_std)):
            raise ValueError("Mapping resolution, age and measurement uncertainty must be positive")
        if min(self.local_rows, self.local_columns, self.map_size) < 3:
            raise ValueError("Mapping grids must have at least three cells per axis")
        if not math.isfinite(self.process_variance_per_second) or self.process_variance_per_second < 0:
            raise ValueError("Mapping process variance must be nonnegative")


def rotate(points, quaternion):
    """Apply a unit xyzw quaternion with broadcastable leading dimensions."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    while q.ndim < points.ndim:
        q = q.unsqueeze(-2)
    v = q[..., :3].expand_as(points)
    uv = torch.linalg.cross(v, points)
    return points + 2 * (q[..., 3:4] * uv + torch.linalg.cross(v, uv))


def compose_rotation(left, right):
    """Hamilton product for xyzw frame orientations."""
    left, right = torch.broadcast_tensors(left, right)
    xyz = left[..., 3:4]*right[..., :3]+right[..., 3:4]*left[..., :3]
    xyz = xyz+torch.linalg.cross(left[..., :3], right[..., :3])
    w = left[..., 3:4]*right[..., 3:4]-(left[..., :3]*right[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


def yaw(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def rotate_yaw(points, angle):
    while angle.ndim < points.ndim - 1:
        angle = angle.unsqueeze(-1)
    x, y, z = points.unbind(-1)
    return torch.stack((angle.cos()*x-angle.sin()*y, angle.sin()*x+angle.cos()*y, z), -1)


def local_centers(config, device):
    x = (torch.arange(config.local_rows, device=device) + 0.5 - config.local_rows/2) * config.resolution
    y = (torch.arange(config.local_columns, device=device) + 0.5 - config.local_columns/2) * config.resolution
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy, torch.zeros_like(xx)), -1).flatten(0, 1)


def rasterize(points, valid, config):
    """Highest hit per cell in the gravity-aligned sensor frame; unknown != ground."""
    batch, _, _ = points.shape
    shape = (config.local_rows, config.local_columns)
    limits = torch.tensor(shape, device=points.device)
    safe = torch.where(torch.isfinite(points), points, 0)
    cells = torch.floor(safe[..., :2] / config.resolution + limits / 2).long()
    inside = valid & torch.isfinite(points).all(-1) & ((cells >= 0) & (cells < limits)).all(-1)
    ids = (cells[..., 0] * shape[1] + cells[..., 1]).clamp(0, shape[0]*shape[1]-1)
    heights = torch.full((batch, shape[0]*shape[1]), -torch.inf, device=points.device)
    heights.scatter_reduce_(1, ids, safe[..., 2].masked_fill(~inside, -torch.inf), reduce="amax")
    known = torch.isfinite(heights).reshape(batch, *shape)
    return heights.nan_to_num(neginf=0).reshape(batch, *shape), known


def neighborhood(height, valid):
    values = F.unfold(height[:, None], 3, padding=1).reshape(height.shape[0], 9, *height.shape[1:])
    masks = F.unfold(valid[:, None].float(), 3, padding=1).reshape_as(values).bool()
    count = masks.sum(1)
    average = (values * masks).sum(1) / count.clamp_min(1)
    return values, masks, count, average


class MappingNetwork(nn.Module):
    """Local CNN implementation of measured / neighborhood average / empty modes."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ELU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ELU(), nn.Conv2d(32, 3, 1))

    def forward(self, height, valid):
        _, _, count, average = neighborhood(height, valid)
        logits = self.layers(torch.stack((height.masked_fill(~valid, 0), valid.float(), average), 1))
        available = torch.stack((valid, count > 0, torch.ones_like(valid)), 1)
        return logits.masked_fill(~available, -1e9)

    @torch.no_grad()
    def predict(self, height, valid, measurement_std=0.03):
        _, _, count, average = neighborhood(height, valid)
        probabilities = self(height, valid).softmax(1)
        mode = probabilities.argmax(1)
        prediction = torch.where(mode == 0, height, average)
        known = (mode != 2) & (count > 0)
        # Conditional variance over finite elevations, plus uncertainty for empty-mode mass.
        weights = probabilities[:, :2] / probabilities[:, :2].sum(1, keepdim=True).clamp_min(1e-8)
        choices = torch.stack((height, average), 1)
        mean = (weights * choices).sum(1)
        variance = (weights * (choices-mean[:, None]).square()).sum(1)
        std = (variance + measurement_std**2 + probabilities[:, 2]*0.15**2).sqrt()
        return prediction.masked_fill(~known, 0), known, std


def mode_labels(height, valid, target, target_valid, tolerance=0.03):
    """Choose the closest usable mode to known ground truth; unknown truth is ignored."""
    if tolerance <= 0:
        raise ValueError("Label tolerance must be positive")
    _, _, count, average = neighborhood(height, valid)
    errors = torch.stack(((height-target).abs().masked_fill(~valid, torch.inf),
                          (average-target).abs().masked_fill(count == 0, torch.inf)), 1)
    error, labels = errors.min(1)
    labels = torch.where(error <= tolerance, labels, 2)
    return labels.masked_fill(~target_valid, -100)


def load_mapping_network(path, device):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get("schema") != MAPPING_SCHEMA or bundle.get("trained_steps", 0) < 1:
        raise ValueError("Neural mapping requires a trained ACP mapping checkpoint")
    network = MappingNetwork().to(device)
    network.load_state_dict(bundle["model"])
    return network.eval().requires_grad_(False)


class ElevationMap:
    """Batched rolling world-axis map with bounded memory and per-environment reset."""
    def __init__(self, num_envs, device="cpu", config=None):
        self.config = config or MappingConfig()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.perception_mode = "fused_raw"
        n = self.config.map_size
        self.origin = torch.zeros(num_envs, 2, device=device, dtype=torch.long)
        self.height = torch.zeros(num_envs, n*n, device=device)
        self.variance = torch.full_like(self.height, torch.inf)
        self.stamp = torch.full_like(self.height, -torch.inf)
        self.clock = torch.full((num_envs,), -torch.inf, device=device)
        i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
        self.cells = torch.stack((i.flatten(), j.flatten()), -1)

    def reset(self, env_ids):
        self.height[env_ids] = 0
        self.variance[env_ids] = torch.inf
        self.stamp[env_ids] = -torch.inf
        self.clock[env_ids] = -torch.inf

    def recenter(self, position):
        n = self.config.map_size
        origin = torch.floor(position[:, :2]/self.config.resolution).long() - n//2
        old_cells = self.cells[None] + (origin-self.origin)[:, None]
        inside = ((old_cells >= 0) & (old_cells < n)).all(-1)
        ids = (old_cells[..., 0]*n + old_cells[..., 1]).clamp(0, n*n-1)
        for field, empty in (("height", 0), ("variance", torch.inf), ("stamp", -torch.inf)):
            setattr(self, field, getattr(self, field).gather(1, ids).masked_fill(~inside, empty))
        self.origin.copy_(origin)

    @torch.no_grad()
    def fuse(self, points_w, valid, std, sensor_position, timestamp):
        """Fuse a gravity-grid prediction, with only strictly newer frames accepted.

        Timestamp is scalar or one value per environment. Duplicate frames do not
        change uncertainty; backwards time requires an explicit reset.
        """
        time = torch.as_tensor(timestamp, device=self.device, dtype=torch.float32).expand(self.num_envs)
        if not torch.isfinite(time).all() or not torch.isfinite(sensor_position).all():
            raise ValueError("Mapping requires finite timestamps and sensor positions")
        if (time < self.clock).any():
            raise ValueError("Out-of-order mapping frame; reset the environment before rewinding time")
        fresh = time > self.clock
        if not fresh.any():
            return
        self.recenter(sensor_position)
        n, res = self.config.map_size, self.config.resolution
        safe = torch.where(torch.isfinite(points_w), points_w, 0)
        cells = torch.floor(safe[..., :2]/res).long()-self.origin[:, None]
        usable = valid & torch.isfinite(points_w).all(-1) & torch.isfinite(std) & (std > 0)
        usable &= ((cells >= 0) & (cells < n)).all(-1) & fresh[:, None]
        ids = (cells[..., 0]*n+cells[..., 1]).clamp(0, n*n-1)
        incoming = torch.full_like(self.height, -torch.inf)
        incoming.scatter_reduce_(1, ids, safe[..., 2].masked_fill(~usable, -torch.inf), reduce="amax")
        highest = usable & (safe[..., 2] == incoming.gather(1, ids))
        noise = torch.full_like(self.height, torch.inf)
        noise.scatter_reduce_(1, ids, std.square().masked_fill(~highest, torch.inf), reduce="amin")
        new_valid = torch.isfinite(incoming)
        incoming = incoming.masked_fill(~new_valid, 0)
        age = time[:, None]-self.stamp
        old_valid = torch.isfinite(self.variance) & (age <= self.config.max_age)
        xy = (self.origin[:, None]+self.cells[None]+0.5)*res
        distance = (xy-sensor_position[:, None, :2]).norm(dim=-1)
        angle = torch.atan2(incoming-sensor_position[:, 2:3], distance)
        protect = old_valid & (self.variance.sqrt() < 0.15) & (incoming < self.height-0.15)
        protect &= angle <= -40 * torch.pi/180
        accept = new_valid & ~protect
        prior_var = self.variance + age.clamp_min(0)*self.config.process_variance_per_second
        same_surface = old_valid & ((incoming-self.height).abs() <= 0.15)
        # Mask absent priors before arithmetic, avoiding inf / inf on unknown cells.
        prior_var = torch.where(same_surface, prior_var, torch.ones_like(prior_var))
        noise = torch.where(new_valid, noise, torch.ones_like(noise))
        weight = prior_var/(prior_var+noise)
        fused = self.height+weight*(incoming-self.height)
        self.height = torch.where(accept, torch.where(same_surface, fused, incoming), self.height)
        self.variance = torch.where(accept, torch.where(same_surface, (1-weight)*prior_var, noise), self.variance)
        self.stamp = torch.where(accept, time[:, None], self.stamp)
        self.clock = torch.where(fresh, time, self.clock)

    def query(self, xy, timestamp, *, fill=True):
        """Nearest-cell samples; optional local minimum/median repair of uncertain cells."""
        n = self.config.map_size
        time = torch.as_tensor(timestamp, device=self.device).expand(self.num_envs)
        if not torch.isfinite(time).all() or (time < self.clock).any():
            raise ValueError("Cannot query a map at a nonfinite or earlier timestamp")
        known = torch.isfinite(self.variance) & ((time[:, None]-self.stamp) <= self.config.max_age)
        cells = torch.floor(xy/self.config.resolution).long()-self.origin[:, None]
        inside = torch.isfinite(xy).all(-1) & ((cells >= 0) & (cells < n)).all(-1)

        def gather(field, coords):
            ids = (coords[..., 0]*n+coords[..., 1]).clamp(0, n*n-1)
            return field.gather(1, ids.flatten(1)).reshape(ids.shape)

        height = gather(self.height, cells)
        valid = inside & gather(known, cells)
        if fill:
            # Query only the neighborhoods of requested samples, not 9 copies of
            # the full map per actor observation. Each median neighbor has its
            # own 3x3 minimum-inpainting support (a 5x5 total footprint).
            axis = torch.arange(-1, 2, device=self.device)
            offsets = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(9, 2)
            neighbors = cells[..., None, :]+offsets
            support = neighbors[..., None, :]+offsets
            support_valid = ((support >= 0) & (support < n)).all(-1) & gather(known, support)
            minimum = gather(self.height, support).masked_fill(~support_valid, torch.inf).amin(-1)
            neighbor_known = ((neighbors >= 0) & (neighbors < n)).all(-1) & gather(known, neighbors)
            repair_neighbors = ~neighbor_known | (gather(self.variance, neighbors).sqrt() >= 0.15)
            repaired = torch.where(repair_neighbors, minimum, gather(self.height, neighbors))
            available = ((neighbors >= 0) & (neighbors < n)).all(-1) & (neighbor_known | support_valid.any(-1))
            ordered = repaired.masked_fill(~available, torch.inf).sort(-1).values
            median = ordered.gather(-1, ((available.sum(-1)-1).clamp_min(0)//2)[..., None]).squeeze(-1)
            repair = ~valid | (gather(self.variance, cells).sqrt() >= 0.15)
            valid = inside & (valid | support_valid[..., 4, :].any(-1))
            height = torch.where(repair, median, height)
        return height.masked_fill(~valid, 0), valid

    def export(self, path, env_id=0):
        """Export accumulated 2.5-D surface points and the raw map, without overwriting."""
        import numpy as np
        path = Path(path)
        known = torch.isfinite(self.variance[env_id]) & (self.clock[env_id]-self.stamp[env_id] <= self.config.max_age)
        xy = (self.origin[env_id]+self.cells+0.5)*self.config.resolution
        points = torch.cat((xy, self.height[env_id, :, None]), -1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, schema=MAPPING_SCHEMA, perception_mode=self.perception_mode,
                points_w=points[known].cpu().numpy(),
                height=self.height[env_id].reshape(self.config.map_size, -1).cpu().numpy(),
                valid=known.reshape(self.config.map_size, -1).cpu().numpy(),
                variance=self.variance[env_id].reshape(self.config.map_size, -1).cpu().numpy(),
                timestamps=self.stamp[env_id].reshape(self.config.map_size, -1).cpu().numpy(),
                origin_xy=(self.origin[env_id]*self.config.resolution).cpu().numpy(),
                resolution=self.config.resolution)


class MappingPipeline:
    def __init__(self, num_envs, device="cpu", config=None, network=None):
        self.map = ElevationMap(num_envs, device, config)
        self.network = network
        self.map.perception_mode = "fused_neural" if network is not None else "fused_raw"

    @torch.no_grad()
    def integrate(self, points_sensor, valid, position_w, quaternion_xyzw, timestamp):
        """Input: deskewed scan in the sensor frame and its odometry pose at scan time."""
        if not torch.isfinite(quaternion_xyzw).all() or (quaternion_xyzw.norm(dim=-1) < 1e-6).any():
            raise ValueError("Invalid sensor orientation")
        config = self.map.config
        angle = yaw(quaternion_xyzw / quaternion_xyzw.norm(dim=-1, keepdim=True))
        world_delta = rotate(points_sensor, quaternion_xyzw)
        local = rotate_yaw(world_delta, -angle)
        height, known = rasterize(local, valid, config)
        if self.network is None:
            std = torch.full_like(height, config.measurement_std)
        else:
            height, known, std = self.network.predict(height, known, config.measurement_std)
        centers = local_centers(config, height.device)[None].expand(len(height), -1, -1).clone()
        centers[..., 2] = height.flatten(1)
        points = rotate_yaw(centers, angle)+position_w[:, None]
        self.map.fuse(points, known.flatten(1), std.flatten(1), position_w, timestamp)

    def scan(self, root_position, root_quaternion, timestamp, rows=17, columns=11):
        x = (torch.arange(rows, device=self.map.device)-(rows-1)/2)*0.1
        y = (torch.arange(columns, device=self.map.device)-(columns-1)/2)*0.1
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack((xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())), -1)
        points = points[None].expand(self.map.num_envs, -1, -1).clone()
        world = rotate_yaw(points, yaw(root_quaternion))+root_position[:, None]
        height, valid = self.map.query(world[..., :2], timestamp)
        points[..., 2] = (height-root_position[:, 2:3]).masked_fill(~valid, 0)
        return points, valid
