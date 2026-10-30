"""Real native-asset EPA regressions and invalid-result boundary checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from somaforge_core.solid_distance import SolidDistanceScene


@pytest.fixture
def recorded_pair():
    path = Path(__file__).parent/'fixtures'/'g1_cylinder_convex_epa_regression.json'
    data = json.loads(path.read_text())
    names = tuple(shape['body'] for shape in data['shapes'])
    return data, names, SolidDistanceScene(data, names)


def body_poses(data, case):
    """The log records shape poses; remove the exported local shape transform."""
    position, rotation = np.array(case['positions']), np.array(case['rotations'])
    local = np.array([s['transform'] for s in data['shapes']])
    body_rotation = rotation @ Rotation.from_quat(local[:, 3:]).as_matrix().swapaxes(-1, -2)
    return position-np.einsum('bij,bj->bi', body_rotation, local[:, :3]), body_rotation


def test_recorded_abort_and_wrong_finite_depth_use_supported_precision(recorded_pair):
    data, _, scene = recorded_pair
    positions, rotations = zip(*(body_poses(data, c) for c in data['cases']))
    rows = scene.query(np.stack(positions), np.stack(rotations))
    np.testing.assert_allclose(rows['dist'], [-.00999538, -.00999393], atol=1.e-6, rtol=0)
    assert rows['sample'].tolist() == [0, 1]
    # Opposite argument order must describe the same geometric violation.
    reverse = copy.deepcopy(data); reverse['allowed_pairs'] = [[1, 0]]
    other = SolidDistanceScene(reverse, scene.link_names).query(np.stack(positions), np.stack(rotations))
    np.testing.assert_allclose(rows['dist'], other['dist'], atol=1.e-6, rtol=0)
    assert (np.einsum('ij,ij->i', rows['normal_w'], -other['normal_w']) > .99999).all()


def test_recorded_neighborhood_has_consistent_depth_and_direction(recorded_pair):
    data, names, scene = recorded_pair
    base, rotation = body_poses(data, data['cases'][0])
    rng = np.random.default_rng(20261008)
    positions = np.concatenate([base+rng.normal(size=(200, 2, 3))*sigma
                                for sigma in (1.e-6, 1.e-4, 1.e-3)])
    rotations = np.broadcast_to(rotation, (len(positions), 2, 3, 3))
    rows = scene.query(positions, rotations)
    reverse = copy.deepcopy(data); reverse['allowed_pairs'] = [[1, 0]]
    other = SolidDistanceScene(reverse, names).query(positions, rotations)
    assert len(rows['dist']) == len(positions)
    np.testing.assert_allclose(rows['dist'], other['dist'], atol=2.e-6, rtol=0)
    assert (np.einsum('ij,ij->i', rows['normal_w'], -other['normal_w']) > .999).all()
    moved = positions.copy(); moved[:, 0] -= .001*rows['normal_w']
    after = scene.query(moved, rotations)
    assert (after['dist'] > rows['dist']+.00099).all()


@pytest.mark.parametrize('gap', [-np.finfo(float).max, np.finfo(float).max,
                                -.2157492372144776, float('nan')])
def test_invalid_distance_is_rejected_before_witness_operations(recorded_pair, monkeypatch, gap):
    data, _, scene = recorded_pair
    position, rotation = body_poses(data, data['cases'][0])
    monkeypatch.setattr(coal, 'distance', lambda *args: gap)
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(ValueError, match='invalid_distance_or_geometric_exit_bound'):
            scene.query(position[None], rotation[None])


def test_finite_sentinel_points_are_rejected_without_overflow(recorded_pair):
    data, _, scene = recorded_pair
    position, rotation = body_poses(data, data['cases'][0])
    scene.coal = SimpleNamespace(Transform3s=coal.Transform3s, distance=lambda *args: -.01,
        DistanceResult=lambda: SimpleNamespace(normal=np.array([1., 0., 0.]),
            getNearestPoint1=lambda: np.full(3, -1.e307),
            getNearestPoint2=lambda: np.full(3, 1.e307)))
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(ValueError, match='invalid_witness_coordinates_or_normal'):
            scene.query(position[None], rotation[None])


def test_finite_sentinel_normal_is_rejected_without_overflow(recorded_pair):
    data, _, scene = recorded_pair
    position, rotation = body_poses(data, data['cases'][0])
    scene.coal = SimpleNamespace(Transform3s=coal.Transform3s, distance=lambda *args: -.01,
        DistanceResult=lambda: SimpleNamespace(normal=np.full(3, 1.e307),
            getNearestPoint1=lambda: np.array(data['cases'][0]['positions'][0]),
            getNearestPoint2=lambda: np.array(data['cases'][0]['positions'][0])))
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(ValueError, match='invalid_witness_coordinates_or_normal'):
            scene.query(position[None], rotation[None])


@pytest.mark.parametrize('reverse', [False, True])
def test_native_batch_matches_reference_neighborhood(recorded_pair, reverse):
    data, names, _ = recorded_pair
    data = copy.deepcopy(data)
    if reverse:
        data['allowed_pairs'] = [[1, 0]]
    reference = SolidDistanceScene(data, names)
    native = SolidDistanceScene(data, names, backend='native_batch')
    base, rotation = body_poses(data, data['cases'][0])
    rng = np.random.default_rng(20261008)
    positions = np.concatenate([base+rng.normal(size=(200, 2, 3))*sigma
                                for sigma in (1.e-6, 1.e-4, 1.e-3)])
    rotations = np.broadcast_to(rotation, (len(positions), 2, 3, 3))
    for poses in (positions, positions+np.array([[[0., 0., 0.], [0., 0., 10.]]])):
        expected, actual = reference.query(poses, rotations), native.query(poses, rotations)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize('bad', ['distance', 'finite_distance', 'points', 'normal'])
def test_native_batch_rejects_invalid_results_before_arithmetic(recorded_pair, monkeypatch, bad):
    from somaforge_core import solid_distance_batch
    data, names, _ = recorded_pair
    scene = SolidDistanceScene(data, names, backend='native_batch')
    position, rotation = body_poses(data, data['cases'][0])
    raw = np.zeros((1, 10)); raw[0, 0] = -.01; raw[0, 1] = 1.
    if bad == 'distance':
        raw[0, 0] = -np.finfo(float).max
    elif bad == 'finite_distance':
        raw[0, 0] = -.2157492372144776
    elif bad == 'points':
        raw[0, 4:7] = -1.e307; raw[0, 7:] = 1.e307
    else:
        raw[0, 1:4] = 1.e307
    monkeypatch.setattr(solid_distance_batch, 'load_coal_batch',
                        lambda: SimpleNamespace(run=lambda *args: raw))
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(ValueError, match='invalid_distance' if 'distance' in bad else 'invalid_witness'):
            scene.query(position[None], rotation[None])


def test_native_batch_buffer_and_index_contract():
    from somaforge_core.coal_batch_loader import load_coal_batch
    native = load_coal_batch()
    geometry = [coal.Sphere(1.)]
    p = np.zeros((1, 1, 3)); r = np.eye(3)[None, None].copy()
    ids = np.zeros((1, 3), dtype=np.int64)
    request = coal.DistanceRequest()
    assert native.run(geometry, p, r, ids[:0], request).shape == (0, 10)
    for positions, candidates in ((p.astype(np.int64), ids), (p, ids.astype(float)),
                                  (p, ids+1), (p, ids-1)):
        with pytest.raises((ValueError, RuntimeError)):
            native.run(geometry, positions, r, candidates, request)


def test_triangle_soup_preserves_disconnected_solids_and_open_mesh_rejection(recorded_pair):
    import trimesh
    data, _, _ = recorded_pair
    data = copy.deepcopy(data)
    boxes = [trimesh.creation.box(), trimesh.creation.box()]
    boxes[1].apply_translation([3., 0., 0.])
    mesh = trimesh.util.concatenate(boxes)
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    shape = dict(shape=0, body=None, kind='mesh', scale=[1., 1., 1.],
                 transform=[0., 0., 0., 0., 0., 0., 1.],
                 vertices=vertices.tolist(), faces=np.arange(len(vertices)).reshape(-1, 3).tolist())
    data.update(shapes=[shape], allowed_pairs=[])
    scene = SolidDistanceScene(data, ())
    assert len(scene.records) == 2
    assert all(np.allclose(record['high']-record['low'], 1.) for record in scene.records)
    shape['faces'] = shape['faces'][1:]
    with pytest.raises(ValueError, match='not a verified closed convex solid'):
        SolidDistanceScene(data, ())
