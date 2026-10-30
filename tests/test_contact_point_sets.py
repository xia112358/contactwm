from types import SimpleNamespace

import pytest
import torch

from contact_solver.contact_point_sets import (contact_patch_observation, matched_targets,
    matched_part_statistics, differentiable_matched_parts, observed_matched_statistics,
    reference_keep_compatible, matched_endpoint_statistics)
from contact_solver.contact_point_sets import matched_endpoint_objective_terms
from contact_solver.device_contact_objective import DeviceWitnessRows
from test_device_contact_objective import Bodies, observation


def setup():
    q = torch.zeros(1, 36, requires_grad=True)
    obs = observation(q)
    p = obs['pairs']
    p['task_pair'][:] = p['eligible'][:] = True
    p['primary_surface'][:] = p['part'][:] = 0
    p['body_link0'][:] = -1
    p['body_link1'][:] = 0
    p['geometry_point1_w'][:] = torch.tensor([[0., 0., 0.], [.2, 0., 0.]])
    roles = torch.tensor([[2, 0, 0, 0, 0, 0]])
    surfaces = torch.tensor([[0, -1, -1, -1, -1, -1]])
    start = contact_patch_observation(obs, q)
    return q, obs, roles, surfaces, start


def test_deepest_point_switch_does_not_penalize_keep_and_permutation_is_invariant():
    q, obs, roles, surfaces, start = setup()
    targets = torch.ones(1, 6, 3)
    obs['pairs']['dist'][:] = torch.tensor([-.001, -.019])
    error, complete, count = observed_matched_statistics(obs, targets, roles, surfaces, start)
    assert error.item() == 0 and complete.item() and count.item() == 1
    obs['pairs'] = {k:v.flip(0).repeat_interleave(2, dim=0) for k,v in obs['pairs'].items()}
    again = observed_matched_statistics(obs, targets, roles, surfaces, start)
    for a, b in zip(again, (error, complete, count)):
        torch.testing.assert_close(a, b)


def test_true_disjoint_patch_motion_stays_penalized_and_has_endpoint_gradient():
    q, obs, roles, surfaces, start = setup()
    obs['pairs']['geometry_point1_w'][:, 0] += .5
    rows = DeviceWitnessRows(Bodies(), q, obs)
    error, present = differentiable_matched_parts(rows, torch.zeros(1, 6, 3), roles, surfaces, start)
    assert present[0, 0] and error[0, 0].item() == pytest.approx(.09)
    error.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad[0, 0] > 0
    endpoint = contact_patch_observation(obs, q)
    compatible, conflict = reference_keep_compatible(endpoint, roles, surfaces, start, .04)
    assert not compatible.item() and conflict[0, 0]
    assert roles[0, 0] == 2


def test_teacher_keep_label_selects_an_actual_start_point_nearest_end_patch():
    q, obs, roles, surfaces, start = setup()
    obs['pairs']['geometry_point1_w'][:] = torch.tensor([.2, .01, 0.])
    endpoint = contact_patch_observation(obs, q)
    target = matched_targets(torch.ones(1, 6, 3), roles, surfaces, start, endpoint)
    torch.testing.assert_close(target[0, 0], torch.tensor([.2, 0., 0.]))
    assert not target.requires_grad
    with pytest.raises(ValueError, match='starting part/surface'):
        matched_targets(target, roles, surfaces+1, start)


def test_establish_and_keep_share_point_metric_without_cross_part_surface_matching():
    q, obs, roles, surfaces, start = setup()
    roles[0, 0] = 1
    target = torch.zeros(1, 6, 3); target[0, 0] = torch.tensor([.2, .03, .04])
    error, complete, count = observed_matched_statistics(obs, target, roles, surfaces, start)
    assert error.item() == pytest.approx(.0025)
    assert complete.item()
    obs['pairs']['primary_surface'][:] = 1
    assert not observed_matched_statistics(obs, target, roles, surfaces, start)[1].item()
    obs['pairs']['primary_surface'][:] = 0
    obs['pairs']['part'][:] = 1
    assert not observed_matched_statistics(obs, target, roles, surfaces, start)[1].item()


def test_missing_contact_keeps_approach_gradient_but_cannot_pass():
    q, obs, roles, surfaces, start = setup()
    obs['pairs']['eligible'][:] = False
    rows = DeviceWitnessRows(Bodies(), q, obs)
    model = SimpleNamespace(fk=Bodies(), region_geometry=None)
    def router(fk, q, ids, targets, active, surfaces, regions, geometry, **kwargs):
        assert active[0, 0]
        return (q[:, :1]-.3).square().expand(-1, 6)*active
    error, complete, count = matched_endpoint_statistics(model, router, rows, torch.tensor([0]),
        torch.zeros(1, 6, 3), roles, surfaces, start, approach=True)
    assert not complete.item() and count.item() == 0
    error.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad[0, 0] < 0


def test_capacity_is_explicit_and_unallocated_contact_rejected():
    q, obs, roles, surfaces, start = setup()
    with pytest.raises(ValueError, match='capacity'):
        contact_patch_observation(obs, q, capacity=1)
    obs['pairs']['constraint_allocated'][:] = False
    with pytest.raises(ValueError, match='Unallocated'):
        contact_patch_observation(obs, q)


def test_missing_contact_recovery_is_separate_from_actual_endpoint_xyz_error():
    q, obs, roles, surfaces, start = setup()
    obs['pairs']['eligible'][:] = False
    rows = DeviceWitnessRows(Bodies(), q, obs)
    model = SimpleNamespace(fk=Bodies(), region_geometry=None)
    def router(fk, pose, ids, targets, active, surfaces, regions, geometry, **kwargs):
        assert kwargs['per_region'] and active[0,0]
        torch.testing.assert_close(targets[0,0], start['contact_patch_points'][0,0])
        return ((pose[:,:3]-.3).square().sum(-1)[:,None,None]
                * active[...,None]).expand(-1,6,4)
    (error, complete, count), recovery, missing = matched_endpoint_objective_terms(
        model, router, rows, torch.tensor([0]), torch.zeros(1,6,3), roles, surfaces, start)
    assert error.item() == 0 and not complete.item() and count.item() == 0
    assert missing[0,0] and missing.sum() == 1
    recovery.sum().backward()
    assert bool((q.grad[0,:3] < 0).all())


def test_observed_keep_uses_full_xyz_and_does_not_call_recovery():
    q, obs, roles, surfaces, start = setup()
    obs['pairs']['geometry_point1_w'][:,2] += .10
    rows = DeviceWitnessRows(Bodies(), q, obs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Observed contact must not receive geometric recovery')
    (error, complete, count), recovery, missing = matched_endpoint_objective_terms(
        SimpleNamespace(fk=Bodies()), forbidden, rows, torch.tensor([0]),
        torch.zeros(1,6,3), roles, surfaces, start)
    assert error.item() == pytest.approx(.01) and complete.item() and count.item() == 1
    assert not missing.any() and recovery.count_nonzero() == 0
    error.backward()
    assert q.grad[0,2] > 0
