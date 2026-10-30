import inspect

import numpy as np
import torch

from generator.next_interaction_heightmap import (
    HEIGHTMAP_COLS,
    HEIGHTMAP_RESOLUTION_M,
    HEIGHTMAP_ROWS,
    HeightmapInteractionPredictor,
    heightmap_grid,
    render_box_heightmaps,
)


def test_two_centimeter_grid_contract():
    grid = heightmap_grid()
    assert grid.shape == (71, 61, 2)
    assert HEIGHTMAP_ROWS * HEIGHTMAP_COLS == 4331
    assert HEIGHTMAP_RESOLUTION_M == 0.02
    np.testing.assert_allclose(np.diff(grid[:, 0, 0]), 0.02, atol=1e-7)
    np.testing.assert_allclose(np.diff(grid[0, :, 1]), 0.02, atol=1e-7)


def test_renderer_contains_ground_and_rotated_box_top():
    angle = np.deg2rad(30.0)
    rotation = np.asarray(
        [[[np.cos(angle), -np.sin(angle), 0.0],
          [np.sin(angle), np.cos(angle), 0.0],
          [0.0, 0.0, 1.0]]], np.float32,
    )
    height = render_box_heightmaps(
        np.asarray([[0.35, 0.0, 0.4]], np.float32),
        rotation,
        np.asarray([[0.25, 0.20, 0.4]], np.float32),
        np.asarray([0.0], np.float32),
    )
    assert height.shape == (1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS)
    assert np.isclose(height.min(), 0.0)
    assert np.isclose(height.max(), 0.8)
    assert 0 < np.count_nonzero(height[0] > 0.0) < HEIGHTMAP_ROWS * HEIGHTMAP_COLS


def test_predictor_uses_heightmap_and_has_observation_only_signature():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = HeightmapInteractionPredictor(48, 1, refinements=1)
    assert list(inspect.signature(model.forward).parameters) == [
        "current_q", "current_contact", "current_anchor", "current_surface", "heightmap"
    ]
    q = torch.zeros(2, 36)
    q[:, 2] = 0.8
    q[:, 3] = 1.0
    contact = torch.zeros(2, 6, dtype=torch.bool)
    anchor = torch.zeros(2, 6, 3)
    surface = torch.full((2, 6), -1, dtype=torch.long)
    height = torch.zeros(2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS)
    height[1, 35:, 20:40] = 0.8
    output = model(q, contact, anchor, surface, height)
    assert output.qpos.shape == (2, 36)
    assert output.role_logits.shape == (2, 6, 4)
    assert output.surface_logits.shape == (2, 6, 2)
    assert not torch.allclose(output.qpos[0], output.qpos[1])
    (output.qpos.square().sum() + output.role_logits.square().sum()).backward()
    assert model.height.conv[0].weight.grad is not None
    assert model.height.conv[0].weight.grad.abs().sum() > 0
