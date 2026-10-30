"""Geometry checks for importing PHP terrains into the WBT scene."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from holosoma.simulator.shared import terrain as terrain_module


def _importer():
    spec = importlib.util.spec_from_file_location(
        "prepare_php_wbt", Path(__file__).parents[1] / "scripts/prepare_php_wbt_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_php_quaternion_is_wxyz_and_box_size_is_full_extent():
    angle = np.pi / 2
    boxes = np.array([[[1, 2, 1, np.cos(angle / 2), 0, 0, np.sin(angle / 2), 2, 4, 2]]])
    mesh = _importer().terrain_mesh(boxes, 0, np.zeros((2, 1, 3)))
    vertices = mesh.vertices[mesh.vertices[:, 2] > 0]
    np.testing.assert_allclose(vertices.min(axis=0), [-1, 1, 2], atol=1e-6)
    np.testing.assert_allclose(vertices.max(axis=0), [3, 3, 2], atol=1e-6)


def test_flat_terrain_covers_full_motion_without_obstacles():
    positions = np.array([[[-12, 3, 0.8]], [[16, -8, 0.8]]])
    mesh = _importer().terrain_mesh(np.empty((0, 10, 10)), 0, positions)
    np.testing.assert_allclose(mesh.bounds, [[-14, -10, -0.04], [18, 5, 0]])


def test_selects_one_paired_variant_without_mixing_layouts():
    boxes = np.array([[[0, 0, 1, 1, 0, 0, 0, 1, 1, 2], [8, 0, 1, 1, 0, 0, 0, 1, 1, 2]]])
    mesh = _importer().terrain_mesh(boxes, 1, np.zeros((2, 1, 3)))
    top = mesh.vertices[mesh.vertices[:, 2] > 0]
    assert top[:, 0].min() == 7.5
    assert top[:, 0].max() == 8.5


@pytest.mark.parametrize(("includes_ground", "expected_faces"), [(True, 12), (False, 24)])
def test_terrain_loader_adds_ground_only_when_missing(tmp_path, monkeypatch, includes_ground, expected_faces):
    path = tmp_path / "floor.obj"
    trimesh.creation.box(extents=[10, 10, 0.04]).export(path)
    manifest = {
        "path": "test",
        "terrains": [{"terrain_id": 0, "terrain_file": str(path), "includes_ground": includes_ground}],
    }
    monkeypatch.setattr(terrain_module, "load_motion_terrain_manifest", lambda _: manifest)
    terrain = terrain_module.Terrain.__new__(terrain_module.Terrain)
    terrain._cfg = SimpleNamespace(motion_matched_manifest="test", terrain_length=12, terrain_width=12)
    mesh = terrain._initialize_motion_matched_config()
    assert len(mesh.faces) == expected_faces
