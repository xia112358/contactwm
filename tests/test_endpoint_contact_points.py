import pytest
import torch

from contact_solver.endpoint_contact_points import (contact_targets, relation_points,
    endpoint_contact_statistics, observed_contact_statistics)
from contact_solver.device_contact_objective import DeviceWitnessRows
from contact_solver.contact_layout import endpoint_position_loss
from test_device_contact_objective import Bodies, observation


def fixture():
    q = torch.zeros(1, 36, requires_grad=True)
    obs = observation(q)
    p = obs['pairs']
    p['part'][:] = 0
    p['task_pair'][:] = p['eligible'][:] = True
    p['primary_surface'][:] = 0
    p['body_link0'][:] = -1
    p['body_link1'][:] = 0
    # Choose row 1 by activation depth despite row 0 being nearer the target
    # and having the smaller raw distance.
    p['dist'][:] = torch.tensor([-.02, -.01])
    p['includemargin'][:] = torch.tensor([0., .02])
    p['geometry_point1_w'][:] = torch.tensor([[0., 0., 0.], [.1, .2, .3]])
    active = torch.tensor([[True, False, False, False, False, False]])
    surfaces = torch.tensor([[0, -1, -1, -1, -1, -1]])
    return q, obs, active, surfaces


def test_keep_start_target_same_surface_other_roles_use_plan():
    q, obs, active, surfaces = fixture()
    ids = torch.tensor([[0, 1]])
    points, present = relation_points(obs, ids, q)
    state = dict(current_contact_relations=present, contact_surface_ids=ids,
                 contact_relation_points=points)
    planned = torch.ones(1, 6, 3, requires_grad=True)
    roles = torch.tensor([[2, 1, 0, 0, 0, 0]])
    target = contact_targets(planned, roles, surfaces, state)
    torch.testing.assert_close(target[0, 0], torch.tensor([.1, .2, .3]))
    torch.testing.assert_close(target[0, 1], planned[0, 1])
    assert not target.requires_grad
    with pytest.raises(ValueError, match='starting part/surface'):
        contact_targets(planned, roles, surfaces+1, state)


def test_xyz_error_uses_fresh_endpoint_witness_and_fk_gradient():
    q, obs, active, surfaces = fixture()
    rows = DeviceWitnessRows(Bodies(), q, obs)
    target = torch.zeros(1, 6, 3)
    error, complete, count = endpoint_contact_statistics(rows, target, active, surfaces)
    torch.testing.assert_close(error, torch.tensor([.14]))
    assert complete.item() and count.item() == 1
    gradient = torch.autograd.grad(error.sum(), q)[0]
    torch.testing.assert_close(gradient[0, :3], torch.tensor([.2, .4, .6]))
    # Read-only evaluation and differentiable training select the same point.
    actual = observed_contact_statistics(obs, target, active, surfaces)
    for a, b in zip(actual, (error, complete, count)):
        torch.testing.assert_close(a, b)


def test_missing_wrong_surface_and_inactive_are_not_completed():
    q, obs, active, surfaces = fixture()
    obs['pairs']['eligible'][:] = False
    rows = DeviceWitnessRows(Bodies(), q, obs)
    error, complete, count = endpoint_contact_statistics(rows, torch.zeros(1, 6, 3), active, surfaces)
    assert not complete.item() and count.item() == 0
    endpoint_position_loss(error).sum().backward()
    assert torch.isfinite(q.grad).all()
    obs['pairs']['eligible'][:] = True
    assert not endpoint_contact_statistics(rows, torch.zeros(1, 6, 3), active, surfaces+1)[1].item()


def test_common_coordinate_transform_preserves_endpoint_point_loss():
    q, obs, active, surfaces = fixture()
    origin = torch.tensor([[2., 3., 4.]])
    basis = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    target = torch.zeros(1, 6, 3)
    world = endpoint_contact_statistics(DeviceWitnessRows(Bodies(), q, obs), target, active, surfaces)[0]
    local = endpoint_contact_statistics(DeviceWitnessRows(Bodies(), q, obs, world_frame=(origin, basis)),
        target, active, surfaces)[0]
    torch.testing.assert_close(local, world)


def test_unallocated_contact_cannot_supply_keep_target():
    q, obs, active, surfaces = fixture()
    obs['pairs']['constraint_allocated'][:] = False
    with pytest.raises(ValueError, match='Unallocated'):
        relation_points(obs, surfaces, q)


def test_raw_record_evaluation_agrees_with_tensor_selection():
    q, obs, active, surfaces = fixture()
    p = obs['pairs']
    records = [dict(part=0, surface=0, dist=float(p['dist'][i]),
        includemargin=float(p['includemargin'][i]), position_w=p['geometry_point1_w'][i].tolist(),
        constraint_active=True, constraint_type=1, allocated=True, efc_address=[0]) for i in range(2)]
    target = torch.zeros(1, 6, 3)
    expected = observed_contact_statistics(obs, target, active, surfaces)
    actual = observed_contact_statistics(dict(pairs=records), target, active, surfaces)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
