import pytest
import torch

from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def test_prediction_cache_survives_target_diagnostics_and_preserves_gradient():
    torch.manual_seed(17)
    fk = CanonicalG1ForwardKinematics().double()
    q = torch.randn(3, 36, dtype=torch.float64, requires_grad=True)
    target = torch.randn(3, 36, dtype=torch.float64)
    first = ('left_ankle_roll_link', 'right_wrist_yaw_link')
    second = ('torso_link', 'left_knee_link')

    def objective():
        a = fk.link_poses(q, first)
        with torch.no_grad():
            fk(target[:, None])
        b = fk.link_poses(q, second)
        return sum(x.square().sum() for x in (*a, *b))

    expected = objective()
    expected_gradient = torch.autograd.grad(expected, q)[0]
    fk.begin_link_pose_cache(q)
    try:
        fk.link_poses(q, first)
        cached = fk._link_pose_cache
        actual = objective()
        assert fk._link_pose_cache is cached
        actual_gradient = torch.autograd.grad(actual, q)[0]
    finally:
        fk.end_link_pose_cache()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert fk._link_pose_cache is None


def test_cache_rejects_nested_scope_and_does_not_reuse_previous_prediction():
    fk = CanonicalG1ForwardKinematics()
    q = torch.zeros(1, 36); q[:, 3] = 1
    fk.begin_link_pose_cache(q)
    try:
        a = fk(q)[0]
        with pytest.raises(RuntimeError, match='Nested'):
            fk.begin_link_pose_cache(q)
    finally:
        fk.end_link_pose_cache()
    shifted = q.clone(); shifted[:, 0] += 2
    fk.begin_link_pose_cache(shifted)
    try:
        b = fk(shifted)[0]
    finally:
        fk.end_link_pose_cache()
    torch.testing.assert_close(b[..., 0], a[..., 0]+2)
