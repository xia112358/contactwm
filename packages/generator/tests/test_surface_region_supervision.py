from types import SimpleNamespace

import pytest
import torch

from generator.planned_contact_predictor import contact_plan_objective


def example(logits=(0., 0., 0., 0.)):
    scores = torch.tensor(logits).reshape(1, 1, 4).expand(1, 6, 4).clone().requires_grad_()
    role = torch.zeros(1, 6, 4, requires_grad=True)
    points = torch.tensor([[[0., 0., 0.], [.03, 0., 0.], [.01, 0., 0.], [1., 0., 0.]]])
    pred = SimpleNamespace(role_logits=role, role=role.argmax(-1), contact=torch.ones(1, 6, dtype=torch.bool),
        location_logits=scores, cell=scores.argmax(-1), candidate_points_local=points,
        candidate_surfaces=torch.tensor([[1, 1, 0, 1]]), candidate_valid=torch.ones(1, 4, dtype=torch.bool),
        keep_uses_current_point=True)
    target = dict(role=torch.ones(1, 6, dtype=torch.long), contact_cell=torch.zeros(1, 6, dtype=torch.long),
        contact_cell_valid=torch.ones(1, 6, dtype=torch.bool), location_region_points_local=torch.zeros(1, 6, 3),
        location_region_surface=torch.ones(1, 6, dtype=torch.long), location_region_radius_m=torch.tensor([.05]))
    return pred, target


def test_nearby_same_surface_mass_replaces_single_point_ce():
    pred, target = example()
    loss, metrics = contact_plan_objective(pred, target)
    assert metrics['plan_location_loss'].item() == pytest.approx(torch.log(torch.tensor(2.)).item())
    loss.sum().backward()
    # Both nearby same-surface points gain probability; nearby ground and far
    # same-surface points lose probability.
    assert (pred.location_logits.grad[..., :2] < 0).all()
    assert (pred.location_logits.grad[..., 2:] > 0).all()
    target.pop('location_region_points_local')
    _, point = contact_plan_objective(pred, target)
    assert point['plan_location_loss'].item() == pytest.approx(torch.log(torch.tensor(4.)).item())


def test_same_surface_probability_far_from_demonstration_is_penalized():
    a, target = example((0., 4., 0., 0.))
    b, _ = example((0., 0., 0., 4.))
    assert contact_plan_objective(a, target)[1]['plan_location_loss'] < contact_plan_objective(b, target)[1]['plan_location_loss']


@pytest.mark.parametrize('excluded', ['keep', 'idle', 'unseen', 'invalid_candidates'])
def test_excluded_regions_have_zero_location_loss_and_finite_zero_gradients(excluded):
    pred, target = example()
    if excluded == 'keep':
        target['role'].fill_(2);target['contact_cell'].fill_(-1)
    elif excluded == 'idle':target['role'].zero_()
    elif excluded == 'unseen':target['location_region_surface'].fill_(9)
    else:pred.candidate_valid[:, :2] = False
    loss, metrics = contact_plan_objective(pred, target)
    assert metrics['plan_location_loss'].item() == 0
    loss.sum().backward()
    assert torch.isfinite(pred.location_logits.grad).all()
    assert pred.location_logits.grad.count_nonzero() == 0
    assert metrics['plan_region_supervised_parts'].item() == 0
    if excluded in ['unseen', 'invalid_candidates']:
        assert metrics['plan_region_uncovered_parts'].item() == 6


def test_label_coverage_uses_the_configured_region_radius():
    from somaforge_core.world_mesh_scan import mesh_location_labels
    points = torch.tensor([[[.15, 0., 0.]]])
    packet = dict(scan_points_world=torch.tensor([[[0., 0., 0.], [.15, 0., .001]]]),
        scan_surface=torch.tensor([[1, 0]]), scan_valid=torch.ones(1, 2, dtype=torch.bool),
        scan_spacing_m=torch.tensor([.05]))
    active = torch.tensor([[True]]);surface = torch.tensor([[1]])
    assert not mesh_location_labels(points, active, surface, packet)[1].item()
    # A radius larger than the historical two-spacing coverage still works;
    # the closer point on the wrong surface is never a positive.
    cell, valid = mesh_location_labels(points, active, surface, packet, radius_m=torch.tensor([.2]))
    assert valid.item() and cell.item() == 0
    assert not mesh_location_labels(points, active, surface, packet, radius_m=torch.tensor([.1]))[1].item()


def test_unmaterialized_zero_spacing_has_no_region_label():
    from somaforge_core.world_mesh_scan import mesh_location_labels
    packet = dict(scan_points_world=torch.zeros(1, 1, 3), scan_surface=torch.zeros(1, 1, dtype=torch.long),
        scan_valid=torch.zeros(1, 1, dtype=torch.bool), scan_spacing_m=torch.zeros(1))
    _, valid = mesh_location_labels(torch.zeros(1, 6, 3), torch.ones(1, 6, dtype=torch.bool),
        torch.zeros(1, 6, dtype=torch.long), packet, radius_m=torch.zeros(1))
    assert not valid.any()
