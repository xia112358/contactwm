import numpy as np
import torch

from contact_solver.heightmap_contact_projector import (
    HeightmapContactProjector,
    ObservedHeightmapSurfaces,
    observed_surface_indices,
    verify_observation_only_prediction,
)
from generator.next_interaction_heightmap import HEIGHTMAP_COLS, HEIGHTMAP_ROWS


def test_surface_indices_are_derived_from_heightmap_and_current_contact_points() -> None:
    heightmap = torch.zeros((HEIGHTMAP_ROWS, HEIGHTMAP_COLS))
    heightmap[20:40, 18:42] = 0.9
    q = torch.zeros(36)
    q[2] = 0.8
    q[3] = 1.0
    observed = ObservedHeightmapSurfaces.from_observation(heightmap, q)

    np.testing.assert_allclose(observed.level_height_local.tolist(), [0.0, 0.9])
    mask = np.asarray((True, True, False, False, False, False))
    points = np.zeros((6, 3), dtype=np.float32)
    points[0, 2] = 0.8
    points[1, 2] = 1.7
    assert observed.classify_contact_points(mask, points).tolist() == [0, 1, -1, -1, -1, -1]
    assert observed_surface_indices(
        heightmap.numpy(), q.numpy(), mask, points
    ).tolist() == [0, 1, -1, -1, -1, -1]


def test_collision_envelope_comes_only_from_height_samples() -> None:
    heightmap = torch.zeros((HEIGHTMAP_ROWS, HEIGHTMAP_COLS))
    heightmap[30, 30] = 0.9
    q = torch.zeros(36)
    q[3] = 1.0
    observed = ObservedHeightmapSurfaces.from_observation(heightmap, q)

    assert int((observed.collision_heightmap == 0.9).sum()) == 9


def test_one_pass_verifier_does_not_feed_geometry_back(monkeypatch) -> None:
    heightmap = torch.zeros((HEIGHTMAP_ROWS, HEIGHTMAP_COLS))
    q = torch.zeros(36)
    q[3] = 1.0
    surfaces = ObservedHeightmapSurfaces.from_observation(heightmap, q)
    calls = []

    def query(value, _geometry):
        calls.append(np.asarray(value).copy())
        return {
            "full_robot_separation": [{"worst_terrain": None, "worst_self": None}],
        }

    monkeypatch.setattr(
        HeightmapContactProjector,
        "_audit_plan",
        staticmethod(lambda _observed, _surfaces: (
            np.asarray((True, False, False, False, False, False)),
            np.asarray((0, -1, -1, -1, -1, -1)),
        )),
    )
    intended = torch.tensor((True, False, False, False, False, False))
    surface = torch.tensor((0, 0, 0, 0, 0, 0))
    result = verify_observation_only_prediction(
        query, q, intended, surface, surfaces, safe_qpos=q
    )

    assert result.converged
    assert result.iterations == 0
    assert len(calls) == 1
    torch.testing.assert_close(result.qpos, q)
