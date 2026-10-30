import pytest
import torch

from contact_solver.keep_patch_loss import (KeepPatch, bind_keep_patch,
    keep_patch_loss, patch_minimum_motion, calibrated_keep_allowance, keep_allowance_for_observation)


def test_patch_allows_pivot_and_penalizes_translation_without_count_bias():
    # Rotating a patch about its center gives displacements enclosing zero.
    delta = torch.tensor([[.1, .1], [-.1, .1], [-.1, -.1], [.1, -.1]], requires_grad=True)
    groups = torch.zeros(4, dtype=torch.long)
    motion, present = patch_minimum_motion(delta, groups, 2)
    assert present.tolist() == [True, False]
    assert motion.tolist() == [0., 0.]
    motion.sum().backward()
    assert torch.isfinite(delta.grad).all()
    shift = torch.tensor([.3, 0.], requires_grad=True)
    a, _ = patch_minimum_motion(delta.detach()+shift, groups, 1)
    b, _ = patch_minimum_motion((delta.detach()+shift).repeat(3, 1), groups.repeat(3), 1)
    torch.testing.assert_close(a, torch.tensor([.2]))
    torch.testing.assert_close(a, b)
    a.sum().backward()
    torch.testing.assert_close(shift.grad, torch.tensor([1., 0.]))


def test_pivot_between_witnesses_not_at_a_chosen_single_point():
    d = torch.tensor([[0., .01], [0., -.01]], dtype=torch.float64, requires_grad=True)
    motion, _ = patch_minimum_motion(d, torch.zeros(2, dtype=torch.long), 1)
    assert motion.item() == 0
    motion.sum().backward()
    assert torch.isfinite(d.grad).all()


def test_single_witness_empty_patch_and_small_scale_have_finite_gradients():
    d = torch.tensor([[1e-7, 0.], [2e-7, 0.]], dtype=torch.float64, requires_grad=True)
    motion, _ = patch_minimum_motion(d, torch.zeros(2, dtype=torch.long), 1)
    torch.testing.assert_close(motion, torch.tensor([1e-7], dtype=torch.float64))
    motion.sum().backward()
    assert torch.isfinite(d.grad).all()
    empty, present = patch_minimum_motion(d[:0], torch.zeros(0, dtype=torch.long), 3)
    assert empty.tolist() == [0., 0., 0.] and not present.any()


class RigidFK:
    def link_poses(self, q, names):
        rotation = torch.eye(3, dtype=q.dtype).expand(len(q), len(names), 3, 3)
        return q[:, None, :3].expand(-1, len(names), -1), rotation


def fixture():
    return KeepPatch(torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]),
        torch.tensor([[-.1, 0., 0.], [.1, 0., 0.]]),
        torch.tensor([[-.1, 0., 0.], [.1, 0., 0.]]),
        torch.tensor([[0., 0., 1.], [0., 0., 1.]]), ('foot',), 1)


def test_only_predicted_keep_is_regularized_and_normal_lift_is_not_slip():
    q = torch.tensor([[.03, 0., .1]], requires_grad=True)
    allowance = torch.full((6,), .01)
    roles = torch.zeros(1, 6, dtype=torch.long)
    for role in (0, 1, 3):
        roles[0, 0] = role
        loss, _ = keep_patch_loss(RigidFK(), q, roles, fixture(), allowance)
        assert loss.item() == 0
    roles[0, 0] = 2
    loss, _ = keep_patch_loss(RigidFK(), q, roles, fixture(), allowance)
    torch.testing.assert_close(loss, torch.tensor([5.]).log())
    loss.sum().backward()
    assert q.grad[0, 0] > 0 and q.grad[0, 2] == 0


def test_missing_keep_is_reported_not_invented_as_observed_support():
    roles = torch.full((1, 6), 2)
    loss, metrics = keep_patch_loss(RigidFK(), torch.zeros(1, 3), roles, fixture(), torch.zeros(6))
    assert metrics['keep_patch_missing'].item() == 5
    assert loss.item() == 0


def test_unknown_allowance_cannot_dilute_known_keep_penalty():
    patch = fixture()
    patch.part = torch.tensor([0, 5])
    roles = torch.tensor([[2, 0, 0, 0, 0, 2]])
    allowance = torch.zeros(6)
    allowance[5] = torch.inf
    loss, _ = keep_patch_loss(RigidFK(), torch.tensor([[.03, 0., 0.]]), roles, patch, allowance)
    torch.testing.assert_close(loss, torch.tensor([10.]).log())


def test_allocation_failure_has_no_geometric_fallback():
    observed = dict(schema='newton_device_witness_batch_v1',
        pairs=dict(active=torch.tensor([True]), constraint_allocated=torch.tensor([False])))
    with pytest.raises(ValueError, match='Unallocated'):
        bind_keep_patch(RigidFK(), torch.zeros(1, 3), observed)


def test_calibration_preserves_own_demo_and_does_not_leak_validation_median():
    motion = torch.zeros(3, 6)
    motion[:, 0] = torch.tensor([.01, .02, .9])
    roles = torch.zeros(3, 6, dtype=torch.long)
    roles[:, 0] = 2
    allowance, fallback, known = calibrated_keep_allowance(motion,
        torch.ones_like(motion, dtype=torch.bool), roles, torch.tensor([0, 1]))
    torch.testing.assert_close(allowance[:, 0], motion[:, 0])
    assert fallback[0] == .01 and known[0] and not known[1]
    assert torch.isinf(allowance[:, 1]).all()
    assert torch.relu(motion-allowance).square().sum() == 0


def test_post_demo_keep_uses_training_statistics_not_final_action_allowance():
    final_action = torch.full((2, 6), .9)
    fallback = torch.full((6,), .01)
    known = torch.tensor([True, True, True, True, True, False])
    allowance = keep_allowance_for_observation(final_action, fallback, known, torch.tensor([True, False]))
    torch.testing.assert_close(allowance[0], final_action[0])
    torch.testing.assert_close(allowance[1, :5], fallback[:5])
    assert torch.isinf(allowance[1, 5])
    roles = torch.tensor([[2, 0, 0, 0, 0, 0]])
    q = torch.tensor([[.03, 0., 0.]], requires_grad=True)
    demo_loss, _ = keep_patch_loss(RigidFK(), q, roles, fixture(), allowance[:1])
    post_loss, _ = keep_patch_loss(RigidFK(), q, roles, fixture(), allowance[1:])
    assert demo_loss.item() == 0 and post_loss.item() > 0
    post_loss.sum().backward()
    assert q.grad[0, 0] > 0 and torch.isfinite(q.grad).all()


def test_rigid_rotation_about_patch_pivot_and_world_rotation_invariance():
    class YawFK:
        def link_poses(self, q, names):
            c, s = q[:, 3].cos(), q[:, 3].sin()
            zero, one = c*0, c*0+1
            rot = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 1, 3, 3)
            return q[:, None, :3], rot
    # Foot origin moves, but the midpoint of its current material patch stays.
    patch = fixture()
    patch.local = patch.local+torch.tensor([.2, 0., 0.])
    patch.position = patch.local.clone()
    q = torch.tensor([[.2, -.2, 0., torch.pi/2]], requires_grad=True)
    roles = torch.tensor([[2, 0, 0, 0, 0, 0]])
    loss, _ = keep_patch_loss(YawFK(), q, roles, patch, torch.zeros(6))
    assert loss.item() < 1e-10
    loss.sum().backward()
    assert torch.isfinite(q.grad).all()
    d = torch.tensor([[.02, .01], [.02, -.01], [.04, 0.]], dtype=torch.float64)
    angle = torch.tensor(.67, dtype=torch.float64)
    rotation = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos())).reshape(2, 2)
    a, _ = patch_minimum_motion(d, torch.zeros(3, dtype=torch.long), 1)
    b, _ = patch_minimum_motion(d@rotation.T, torch.zeros(3, dtype=torch.long), 1)
    torch.testing.assert_close(a, b)
