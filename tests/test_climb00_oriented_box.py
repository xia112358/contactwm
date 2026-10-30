from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from generator.training_data import box_obbs  # noqa: E402
from contact_solver.collision_geometry import full_geometry_box_ground_penetration, full_geometry_box_penetration
from generator.unified_interaction import rollout_ground_collision_loss  # noqa: E402


def test_box_obbs_follow_polygon_edges_in_arbitrary_surface_coordinates() -> None:
    axis_u = np.asarray((2.0**-0.5, 2.0**-0.5), dtype=np.float32)
    axis_v = np.asarray((-2.0**-0.5, 2.0**-0.5), dtype=np.float32)
    polygon = np.stack(
        (
            -axis_u - 0.5 * axis_v,
            axis_u - 0.5 * axis_v,
            axis_u + 0.5 * axis_v,
            -axis_u + 0.5 * axis_v,
        )
    )[None]
    data = {
        "box_edge_start": polygon,
        "box_origin": np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
        "box_basis": np.eye(3, dtype=np.float32)[None],
        "box_height": np.asarray((1.0,), dtype=np.float32),
    }

    center, rotation, half, ground = box_obbs(data)
    torch.testing.assert_close(torch.from_numpy(center), torch.tensor(((0.0, 0.0, 0.5),)))
    torch.testing.assert_close(torch.from_numpy(half), torch.tensor(((1.0, 0.5, 0.5),)))

    # This point lies inside the old UV min/max bounding rectangle but outside
    # the actual rotated rectangular prism.
    points = torch.tensor([[[[0.9, -0.9, 0.5], [0.0, 0.0, 0.5]]]])
    penetration = full_geometry_box_penetration(
        points,
        torch.tensor((-1, -1)),
        torch.zeros(1, 1, 6),
        box_center=torch.from_numpy(center),
        box_rotation=torch.from_numpy(rotation),
        box_half_extents=torch.from_numpy(half),
        ground_height=torch.from_numpy(ground),
        margin_m=0.0,
        planned_contact_tolerance_m=0.0,
    )
    torch.testing.assert_close(penetration[0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(penetration[0, 0, 1], torch.tensor(0.5), atol=1.0e-6, rtol=0.0)


def test_ground_penetration_is_separate_and_pushes_points_upward() -> None:
    points = torch.tensor([[[[0.0, 0.0, -0.05], [0.0, 0.0, 0.10]]]], requires_grad=True)
    box, ground = full_geometry_box_ground_penetration(
        points,
        torch.tensor((-1, -1)),
        torch.zeros(1, 1, 6),
        box_center=torch.tensor(((10.0, 10.0, 10.0),)),
        box_rotation=torch.eye(3)[None],
        box_half_extents=torch.ones(1, 3),
        ground_height=torch.zeros(1),
        margin_m=0.0,
        planned_contact_tolerance_m=0.0,
    )
    torch.testing.assert_close(box, torch.zeros_like(box))
    torch.testing.assert_close(ground, torch.tensor([[[0.05, 0.0]]]))

    rollout_ground_collision_loss(ground, tolerance_m=0.0).backward()
    assert points.grad is not None
    assert points.grad[0, 0, 0, 2] < 0.0
    torch.testing.assert_close(points.grad[0, 0, 1], torch.zeros(3))


def test_box_and_ground_penetration_metrics_do_not_mask_each_other() -> None:
    points = torch.tensor([[[[0.0, 0.0, 0.5], [2.0, 0.0, -0.2]]]])
    box, ground = full_geometry_box_ground_penetration(
        points,
        torch.tensor((-1, -1)),
        torch.zeros(1, 1, 6),
        box_center=torch.tensor(((0.0, 0.0, 0.5),)),
        box_rotation=torch.eye(3)[None],
        box_half_extents=torch.tensor(((0.5, 0.5, 0.5),)),
        ground_height=torch.zeros(1),
        margin_m=0.0,
        planned_contact_tolerance_m=0.0,
    )
    torch.testing.assert_close(box, torch.tensor([[[0.5, 0.0]]]))
    torch.testing.assert_close(ground, torch.tensor([[[0.0, 0.2]]]))
