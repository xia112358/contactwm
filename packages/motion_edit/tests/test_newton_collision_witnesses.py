import numpy as np
import pytest

from somaforge_core.newton_collision import _effective_surface_witnesses, _quat_rotate_xyzw


def test_sphere_surface_penetrates_even_when_center_is_above_plane():
    # Recorded frame 853: center gap 0.346 mm, actual sphere radius 5 mm.
    a, b, distance = _effective_surface_witnesses(
        np.array([[0., 0., 0.]]), np.array([[0., 0., .000346]]),
        np.array([[0., 0., 1.]]), np.array([.01]), np.array([.015]),
        np.array([.01]), np.array([.01]))
    np.testing.assert_allclose(distance, [-.004654])
    np.testing.assert_allclose(b-a, [[0., 0., -.004654]])


def test_swapping_pair_does_not_change_depth_and_zero_thickness_mesh_is_unchanged():
    p0=np.array([[0., 0., 0.], [1., 2., 3.]])
    p1=np.array([[0., 0., .03], [1., 2., 3.04]])
    normal=np.array([[0., 0., 1.], [0., 0., 1.]])
    t0=np.array([.03, .01]); t1=np.array([.03, .01])
    m0=np.array([.01, .01]); m1=np.array([.01, .01])
    a,b,d=_effective_surface_witnesses(p0,p1,normal,t0,t1,m0,m1)
    rb,ra,rd=_effective_surface_witnesses(p1,p0,-normal,t1,t0,m1,m0)
    np.testing.assert_allclose(d, [-.01, .04])
    np.testing.assert_allclose(rd, d)
    np.testing.assert_allclose(ra,a); np.testing.assert_allclose(rb,b)
    np.testing.assert_allclose(a[1],p0[1]); np.testing.assert_allclose(b[1],p1[1])


def test_batched_witness_rotation_matches_independent_rigid_transforms():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(20261004)
    rotation = Rotation.random(40, random_state=rng)
    points = rng.normal(size=(40, 3))
    actual = _quat_rotate_xyzw(rotation.as_quat(), points)
    np.testing.assert_allclose(actual, rotation.apply(points), rtol=1.e-13, atol=1.e-13)
    np.testing.assert_array_equal(actual[0], _quat_rotate_xyzw(rotation.as_quat()[0], points[0]))
    assert _quat_rotate_xyzw(np.empty((0, 4)), np.empty((0, 3))).shape == (0, 3)


@pytest.mark.parametrize('include_fk', [False, True])
def test_captured_collision_requeries_changed_poses_with_identical_native_fields(include_fk):
    from dataclasses import fields
    import pytest
    pytest.importorskip('newton')
    from somaforge_core.newton_collision import DirectNewtonCollisionScene

    scene = DirectNewtonCollisionScene(None)
    poses = np.zeros((3, 36), dtype=np.float32)
    poses[:, 2] = .8
    poses[:, 3] = 1.
    poses[1, 7:] = np.linspace(-.8, .8, 29)
    poses[2, 7:] = np.linspace(.5, -.5, 29)
    expected, body_poses = [], []
    for pose in poses:
        expected.append(scene.query_qpos(pose))
        body_poses.append(scene.body_poses())
    scene.prepare_collision_graph(include_fk=include_fk)
    for index in (2, 0, 1, 0):
        actual = scene.query_qpos(poses[index])
        for item in fields(actual):
            before, after = getattr(expected[index], item.name), getattr(actual, item.name)
            if isinstance(before, np.ndarray):
                np.testing.assert_array_equal(before, after, err_msg=item.name)
            else:
                assert before == after, item.name
        for before, after in zip(body_poses[index], scene.body_poses()):
            np.testing.assert_array_equal(before, after)
    # A body-pose query must preserve the supplied transforms even after joint
    # FK was captured. Stale joint_q cannot overwrite this separate public API.
    positions, quaternions = body_poses[0]
    moved = positions+.25
    scene.query_body_poses(moved, quaternions, scene.body_names)
    np.testing.assert_array_equal(scene.body_poses()[0], moved)
    old_contacts = scene.contacts
    scene.contacts = scene.pipeline.contacts()
    with pytest.raises(RuntimeError, match='recapture'):
        scene.collide()
    scene.contacts = old_contacts
    if include_fk:
        old_joint_q = scene.state.joint_q
        scene.state.joint_q = scene.wp.zeros_like(old_joint_q)
        with pytest.raises(RuntimeError, match='recapture'):
            scene.query_qpos(poses[0])
        scene.state.joint_q = old_joint_q
