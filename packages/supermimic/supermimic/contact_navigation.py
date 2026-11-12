"""Differentiable navigation distance for top contacts on one grounded box.

This is the retained single-box research metric, not the contact-training default.
The conservative whole-foot envelope uses material XY references and the mesh's
minimum Z; it does not fully constrain the final material-point heights.
Native solver evidence remains responsible for contact and termination.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.sparse.csgraph import floyd_warshall


def segment_visible(start, end, half):
    """Whether segments avoid the open interior of an axis-aligned box."""
    delta = end - start
    parallel = delta.abs() < 1e-10
    denominator = torch.where(parallel, torch.ones_like(delta), delta)
    first = (-half - start) / denominator
    last = (half - start) / denominator
    enter = torch.where(parallel, torch.full_like(first, -float('inf')), torch.minimum(first, last))
    leave = torch.where(parallel, torch.full_like(first, float('inf')), torch.maximum(first, last))
    outside_parallel = (parallel & (start.abs() >= half)).any(-1)
    crosses = (enter.amax(-1).clamp_min(0) < leave.amin(-1).clamp_max(1) - 1e-7) & ~outside_parallel
    return ~crosses


class BoxVisibilityGraph:
    """Cache free-space connectivity; source and goal coordinates stay online.

    Coordinates are world-space outside ``local_query``. Each reference has its
    own inflated half extents. The box must extend to the z=0 floor; underpasses
    are outside this representation. Graph routes approximate shortest paths.
    """

    def __init__(self, center, rotation, half, device):
        self.center = center.to(device)
        self.R = rotation.to(device)
        self.half = half.to(device)
        nodes, costs, predecessors = [], [], []
        for extent in half.cpu().double():
            hx, hy, hz = extent.tolist()
            margin = .005
            floor = -float(center[2]) + .01
            points = []
            for z in [floor, (floor + hz) / 2, hz + margin]:
                for t in np.linspace(-1, 1, 9):
                    points.extend([
                        [t * (hx + margin), -hy - margin, z],
                        [t * (hx + margin), hy + margin, z],
                        [-hx - margin, t * (hy + margin), z],
                        [hx + margin, t * (hy + margin), z],
                    ])
            points = torch.tensor(np.unique(np.asarray(points), axis=0), dtype=torch.float64)
            visible = segment_visible(points[:, None], points[None], extent)
            edges = (points[:, None] - points[None]).norm(dim=-1).masked_fill(~visible, float('inf'))
            distances, parent = floyd_warshall(edges.numpy(), directed=False, return_predecessors=True)
            assert np.isfinite(distances).all(), 'Disconnected free-space roadmap'
            nodes.append(points.float())
            costs.append(torch.tensor(distances, dtype=torch.float32))
            predecessors.append(parent)
        self.nodes = torch.stack(nodes).to(device)
        self.costs = torch.stack(costs).to(device)
        self.predecessors = np.stack(predecessors)
        self.node_count = self.nodes.shape[1]

    def local_query(self, position, goal, return_route=False):
        gap = position.abs() - self.half[None]
        inside = gap.amax(-1) < 0
        # A grounded box's bottom exit is below the floor and has no route.
        # Keep the nearest side/top exit instead of projecting into an underpass.
        exit_gap = gap.clone()
        exit_gap[..., 2] = torch.where(position[..., 2] < 0, -self.half[None, :, 2] + position[..., 2], gap[..., 2])
        axis = exit_gap.argmax(-1)
        face = torch.nn.functional.one_hot(axis, 3).bool()
        sign = torch.where(position >= 0, 1., -1.)
        sign = torch.where(face & (axis[..., None] == 2), torch.ones_like(sign), sign)
        # An overlapping envelope has no free-space path. Extend the metric by
        # exiting the nearest surface, rather than navigating through the solid.
        free = torch.where(
            inside[..., None],
            torch.where(face, sign * self.half[None], position) + face * sign * 1e-5,
            position,
        )
        nodes = self.nodes[None]
        half = self.half[None, :, None]
        source = (free[:, :, None] - nodes).norm(dim=-1).masked_fill(
            ~segment_visible(free[:, :, None], nodes, half), float('inf'))
        target = (goal[:, :, None] - nodes).norm(dim=-1).masked_fill(
            ~segment_visible(goal[:, :, None], nodes, half), float('inf'))
        tail, index_j = (self.costs[None] + target[:, :, None]).min(-1)
        indirect, index_i = (source + tail).min(-1)
        direct = (free - goal).norm(dim=-1).masked_fill(
            ~segment_visible(free, goal, self.half[None]), float('inf'))
        depth = torch.where(inside, -exit_gap.amax(-1), 0.)
        distance = torch.minimum(direct, indirect) + depth
        if return_route:
            return distance, (free, index_i, index_j, direct <= indirect)
        return distance

    def query(self, position, goal):
        return self.local_query((position - self.center) @ self.R, (goal - self.center) @ self.R)


class WholeFootBoxNavigation:
    """Four material XY references with an orientation-aware whole-foot base.

    ``boxes`` contains one grounded, upright box (xyz, wxyz, size).
    ``goals`` and ``material_local`` contain four corresponding sole anchors.
    Goals describe the intended pose on the box top. Geometry is an optimization
    metric only; this object never detects contact or changes solver margins.
    """

    def __init__(self, boxes, goals, vertices, material_local, device):
        from .rotations import quat_to_mat

        start = time.perf_counter()
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 10)
        assert len(boxes) == 1, 'Single-box navigation contract'
        self.center = boxes[0, :3].to(device)
        self.R = quat_to_mat(boxes[0, 3:7]).to(device)
        self.half = boxes[0, 7:10].to(device) * .5
        assert float((self.R[:, 2] - self.R.new_tensor([0, 0, 1])).abs().max()) < 1e-5, 'Upright box required'
        if float(self.center[2] - self.half[2]) > 0:
            raise ValueError('Whole-foot box navigation does not support floating-box underpasses')
        self.vertices = vertices.to(device)
        self.material_local = material_local.to(device)
        self.radius = (self.vertices[None] - self.material_local[:, None]).norm(dim=-1).amax(-1)
        self.nav_half = self.half[None].expand(4, -1).clone()
        self.nav_half[:, :2] += self.radius[:, None]
        self.road = BoxVisibilityGraph(self.center, self.R, self.nav_half, device)
        self.set_goal(goals.to(device))
        self.info = dict(
            representation='cached visibility roadmap', nodes_per_material=self.road.node_count,
            build_seconds=time.perf_counter() - start, envelope_radii_m=self.radius.tolist(),
            cached_tensor_bytes=self.road.nodes.numel() * 4 + self.road.costs.numel() * 4,
            goal_update_rebuilds_graph=False,
        )

    def set_goal(self, targets):
        """Update the top-contact pose without rebuilding graph connectivity."""
        local = self.material_local
        local_center, target_center = local.mean(0), targets.mean(0)
        u, _, vh = torch.linalg.svd((local - local_center).T @ (targets - target_center))
        v = vh.T
        fix = torch.ones(3, device=targets.device, dtype=targets.dtype)
        fix[-1] = torch.linalg.det(v @ u.T)
        rotation = (v * fix[None]) @ u.T
        position = target_center - rotation @ local_center
        mesh = position + torch.einsum('ij,kj->ki', rotation, self.vertices)
        anchors = torch.cat((targets[:, :2], mesh[:, 2].amin().expand(4, 1)), dim=-1)
        goal = (anchors - self.center) @ self.R
        self.goals = torch.cat((goal[:, :2], torch.maximum(goal[:, 2:], self.half[2] + 1e-6)), dim=-1)
        self.goal_world = self.goals @ self.R.T + self.center

    def distance(self, points, position, rotation):
        mesh = position[:, None] + torch.einsum('bij,kj->bki', rotation, self.vertices)
        anchors = torch.cat((points[:, :, :2], mesh[:, :, 2].amin(-1)[:, None, None].expand(-1, 4, 1)), dim=-1)
        return self.anchor_distance(anchors)

    def anchor_distance(self, points):
        return self.road.query(points, self.goal_world[None].expand(len(points), -1, -1))
