import pytest
import torch

from contact_solver.contact_layout import (
    endpoint_position_statistics, persistent_role_consistent, support_bound_targets,
)
from generator.full1000_training_variants import event_consistent_roles
from generator.endpoint_termination import endpoint_failure_reasons


def test_persistent_and_new_contacts_both_keep_real_endpoint_labels():
    current = torch.tensor([[[1., 2., 0.], [3., 2., 0.]]])
    future = current + torch.tensor([.3, .1, 0.])
    role = torch.tensor([[2, 1]])
    contact = torch.ones(1, 2, dtype=torch.bool)
    points, mask = support_bound_targets(future, current, contact, role)
    torch.testing.assert_close(points[:, 0], future[:, 0])
    torch.testing.assert_close(points[:, 1], future[:, 1])
    assert mask.tolist() == [[True, False]]
    # Moving the observed support does not drag the environment's new landing.
    changed, _ = support_bound_targets(future, current+.2, contact, role)
    torch.testing.assert_close(changed[:, 1], future[:, 1])
    torch.testing.assert_close(changed[:, 0], future[:, 0])


def test_scene_rigid_transform_equivariance_and_future_only_shift_penalty():
    current = torch.tensor([[[1., 2., 0.], [3., 2., 0.]]])
    future = current + .3
    role = torch.tensor([[2, 1]])
    contact = torch.ones(1, 2, dtype=torch.bool)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    def transform(x):
        return x @ rotation.T + torch.tensor([10., -7., 2.])
    targets, _ = support_bound_targets(future, current, contact, role)
    transformed, _ = support_bound_targets(transform(future), transform(current), contact, role)
    torch.testing.assert_close(transformed, transform(targets))
    output = (targets + torch.tensor([.2, 0., 0.])).requires_grad_()
    error, _ = endpoint_position_statistics(output, targets, contact)
    changed_error, _ = endpoint_position_statistics(transform(output), transformed, contact)
    torch.testing.assert_close(changed_error, error)
    assert error.item() > .039
    error.sum().backward()
    assert (output.grad[..., 0] > 0).all()  # descent reverses the unwanted shift


def test_no_support_keeps_environment_targets_and_single_contact_has_gradient():
    future = torch.tensor([[[.4, .2, 0.]]])
    targets, mask = support_bound_targets(future, future*0, torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 1, dtype=torch.long))
    torch.testing.assert_close(targets, future)
    assert not mask.any()
    output = torch.zeros_like(future, requires_grad=True)
    error, count = endpoint_position_statistics(output, targets, torch.ones(1, 1, dtype=torch.bool))
    error.sum().backward()
    assert count.item() == 1 and output.grad.abs().sum() > 0


def test_missing_support_is_touchdown_not_fictitious_anchor():
    contact = torch.tensor([[False, True]])
    roles = torch.tensor([[2, 1]])
    points = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match='actual current contact'):
        support_bound_targets(points, points, contact, roles)
    corrected = event_consistent_roles(roles, contact)
    assert corrected.tolist() == [[1, 1]]  # existing contact can still re-touch
    _, persistent = support_bound_targets(points, points, contact, corrected)
    assert not persistent.any()


def test_persistent_role_only_requires_current_contact_not_fixed_position():
    role = torch.tensor([[2, 1]])
    assert persistent_role_consistent(role, torch.tensor([[True, False]])).item()
    assert not persistent_role_consistent(role, torch.tensor([[False, True]])).item()
    record = dict(task_acceptance={'accepted': True}, persistent_role_consistent=False)
    assert endpoint_failure_reasons(record) == ['persistent_role_without_current_contact']
    # A changed endpoint representative is not itself a reason to stop.
    record['persistent_role_consistent'] = True
    assert endpoint_failure_reasons(record) == []


def test_actual_raster_labels_are_invariant_to_joint_xy_frame_change():
    from generator.contact_location_predictor import contact_points_to_observed_heightmap_map
    from generator.next_interaction_heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
    q = torch.zeros(1, 36)
    q[:, 2] = .8
    q[:, 3] = 1.
    points = torch.zeros(1, 6, 3)
    points[0, :2] = torch.tensor([[.2, .1, 0.], [.4, -.1, 0.]])
    contact = torch.tensor([[True, True, False, False, False, False]])
    heightmap = torch.full((1, HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8)
    _, original, valid = contact_points_to_observed_heightmap_map(q, contact, points, heightmap)
    moved_q = q.clone()
    moved_q[:, :2] += torch.tensor([2., -3.])
    moved_points = points + torch.tensor([2., -3., 0.])
    _, moved, moved_valid = contact_points_to_observed_heightmap_map(moved_q, contact, moved_points, heightmap)
    assert valid[contact].all() and moved_valid[contact].all()
    torch.testing.assert_close(original, moved)
    _, future_only, _ = contact_points_to_observed_heightmap_map(q, contact, points+torch.tensor([.2, 0., 0.]), heightmap)
    assert (future_only != original)[contact].all()
