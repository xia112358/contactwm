import numpy as np
from generator.fullbody_dataset import _resample_rotation6d, inherit_active_contact_descriptors
from generator.mechanical_teacher import _match_segment_boundary_velocities


def test_active_contact_descriptor_is_inherited_at_shared_boundary() -> None:
    surface, uv, unresolved = inherit_active_contact_descriptors(
        active=np.asarray((1, 1, 0), dtype=bool),
        surface=np.asarray((-1, 1, -1), dtype=np.int64),
        uv=np.zeros((3, 2), dtype=np.float32),
        previous_active=np.asarray((1, 1, 1), dtype=bool),
        previous_surface=np.asarray((0, 0, 1), dtype=np.int64),
        previous_uv=np.asarray(((0.25, -0.5), (9.0, 9.0), (0.5, 0.5)), dtype=np.float32),
    )

    np.testing.assert_array_equal(surface, (0, 1, -1))
    np.testing.assert_allclose(uv[0], (0.25, -0.5))
    np.testing.assert_allclose(uv[1], (0.0, 0.0))
    np.testing.assert_array_equal(unresolved, (False, False, False))


def test_rotation_resampling_stays_on_so3_and_preserves_endpoints() -> None:
    angle = np.linspace(0.0, np.pi / 2.0, 3)
    rotation = np.zeros((3, 1, 3, 3), dtype=np.float32)
    rotation[:, 0, 0, 0] = np.cos(angle)
    rotation[:, 0, 0, 1] = -np.sin(angle)
    rotation[:, 0, 1, 0] = np.sin(angle)
    rotation[:, 0, 1, 1] = np.cos(angle)
    rotation[:, 0, 2, 2] = 1.0
    rotation6d = rotation[..., :2].reshape(3, 1, 6)
    output = _resample_rotation6d(rotation6d, 11).reshape(11, 1, 3, 2)
    first = output[..., 0]
    second = output[..., 1]
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    identity = np.broadcast_to(np.eye(3), matrix.shape)
    np.testing.assert_allclose(matrix.swapaxes(-1, -2) @ matrix, identity, atol=1.0e-6)
    np.testing.assert_allclose(output[0].reshape(1, 6), rotation6d[0], atol=0.0)
    np.testing.assert_allclose(output[-1].reshape(1, 6), rotation6d[-1], atol=0.0)


def test_segment_velocity_matching_preserves_keyframes_and_matches_velocity() -> None:
    first = np.zeros((8, 36), dtype=np.float32)
    second = np.zeros((9, 36), dtype=np.float32)
    first[:, 0] = np.linspace(0.0, 1.0, len(first))
    second[:, 0] = 1.0 + np.linspace(0.0, 0.2, len(second))
    first[:, 7] = np.linspace(0.0, 0.4, len(first))
    second[:, 7] = 0.4 + np.linspace(0.0, -0.6, len(second))
    output = _match_segment_boundary_velocities([first, second])
    np.testing.assert_allclose(output[0][[0, -1]], first[[0, -1]], atol=1.0e-7)
    np.testing.assert_allclose(output[1][[0, -1]], second[[0, -1]], atol=1.0e-7)
    np.testing.assert_allclose(
        output[0][-1, (0, 7)] - output[0][-2, (0, 7)],
        output[1][1, (0, 7)] - output[1][0, (0, 7)],
        atol=5.0e-3,
    )
