import numpy as np
import pytest
import torch

from somaforge_core.contact_source_geometry import (
    incident_witness_faces, incident_witness_faces_tensor,
    project_source_triangle, project_source_triangles_tensor,
)


def test_numpy_tensor_source_feature_projection_agree():
    rng = np.random.default_rng(47)
    triangles = rng.normal(size=(100, 3, 3))
    points = rng.normal(size=(100, 3))
    expected = np.stack([project_source_triangle(p, t) for p, t in zip(points, triangles)])
    actual, valid = project_source_triangles_tensor(torch.from_numpy(points), torch.from_numpy(triangles))
    assert valid.all()
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-14)


@pytest.mark.parametrize('point, expected', [
    ([.2, .3, 1e-5], [.2, .3, 0]),
    ([.5, -.1, 0], [.5, 0, 0]),
    ([-.1, -.1, 0], [0, 0, 0]),
    ([1, 1, 0], [.5, .5, 0]),
])
def test_projection_identifies_source_feature_without_changing_witness(point, expected):
    point = np.asarray(point)
    before = point.copy()
    triangle = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(project_source_triangle(point, triangle), expected)
    np.testing.assert_array_equal(point, before)


def test_incident_planes_exclude_remote_face_and_numpy_tensor_agree():
    normals = np.asarray([[0., -1., 0.], [0., 0., 1.]])
    offsets, extents = np.asarray([0., 1.]), np.asarray([1., 1.])
    for point, expected in (([.2, 0., .3], [True, False]), ([.5, 0., 1.], [True, True])):
        p = np.asarray(point)
        actual = incident_witness_faces(p, normals, offsets, extents)
        tensor = incident_witness_faces_tensor(*map(torch.from_numpy, (p, normals, offsets, extents)))
        assert actual.tolist() == expected == tensor.tolist()


def test_degenerate_source_is_explicitly_invalid():
    triangle = np.zeros((3, 3))
    with pytest.raises(ValueError, match='Degenerate'):
        project_source_triangle(np.zeros(3), triangle)
    _, valid = project_source_triangles_tensor(torch.zeros(1, 3), torch.zeros(1, 3, 3))
    assert not valid.any()
