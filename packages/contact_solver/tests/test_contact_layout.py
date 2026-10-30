from types import SimpleNamespace

import torch

from contact_solver.contact_layout import (relative_layout_statistics, relative_plan_objective,
    endpoint_position_statistics, endpoint_position_loss, ENDPOINT_POSITION_TOLERANCE_M)
from generator.next_interaction_heightmap import HEIGHTMAP_ROWS as R, HEIGHTMAP_COLS as C
from generator.research.frozen_contact_position import frozen_contact_layout_diagnostic


def example(shift=0, wrong=False):
    logits = torch.full((1, 6, R*C), -15.)
    a = 25*C+20
    b = a+5
    logits[0, 0, a+shift] = 15
    logits[0, 1, b+shift+(4 if wrong else 0)] = 15
    prediction = SimpleNamespace(location_logits=logits.requires_grad_(),
        role_logits=torch.zeros(1, 6, 4, requires_grad=True),
        contact=torch.tensor([[True, True, False, False, False, False]]),
        role=torch.tensor([[1, 1, 0, 0, 0, 0]]))
    target = {'role': prediction.role, 'contact_cell': torch.tensor([[a, b, 0, 0, 0, 0]]),
              'contact_cell_valid': prediction.contact}
    return prediction, target, torch.zeros(1, R, C)


def test_planner_penalizes_future_only_translation_and_independent_limb_shift():
    a, label, terrain = example()
    b, _, _ = example(shift=3*C+7)
    wrong, _, _ = example(shift=3*C+7, wrong=True)
    first = relative_plan_objective(a, label, terrain)[0]
    moved = relative_plan_objective(b, label, terrain)[0]
    broken = relative_plan_objective(wrong, label, terrain)[0]
    assert moved > first + 5
    assert broken > first + 5
    broken.sum().backward()
    assert torch.isfinite(wrong.location_logits.grad).all()
    assert wrong.location_logits.grad.abs().sum() > 0


def test_translated_layout_cannot_change_observed_height_levels():
    prediction, label, terrain = example(shift=3*C)
    terrain[:, 28, :] = .9
    _, metrics = relative_plan_objective(prediction, label, terrain)
    assert metrics['plan_location_loss'] > 5


def test_no_visible_contact_positions_have_zero_location_loss_and_finite_gradient():
    prediction, label, terrain = example()
    label['contact_cell_valid'][:] = False
    loss, metrics = relative_plan_objective(prediction, label, terrain)
    assert metrics['plan_location_loss'] == 0
    loss.sum().backward()
    assert torch.isfinite(prediction.location_logits.grad).all()


def test_relative_statistics_translation_and_missing_witness_are_separate():
    target = torch.tensor([[[0., 0., 0.], [.2, .1, 0.]]])
    actual = target + torch.tensor([2., -3., 0.])
    error, shift, count = relative_layout_statistics(actual, target, torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(error, torch.zeros(1), atol=1e-12, rtol=0)
    torch.testing.assert_close(shift, torch.tensor([[2., -3.]]))
    assert count == 2
    _, _, missing = relative_layout_statistics(actual, target, torch.zeros(1, 2, dtype=torch.bool))
    assert missing == 0


def test_layout_runs_one_fk_for_entire_batch_and_matches_known_gradient():
    class FK:
        calls = 0
        def link_poses(self, q, names):
            self.calls += 1
            pos = torch.stack([q[:, :3] + torch.stack((q[:, 7+i], q[:, 9]*0, q[:, 9]*0), -1)
                               for i in range(len(names))], 1)
            return pos, torch.eye(3).expand(len(q), len(names), 3, 3)
    fk = FK(); q = torch.zeros(2, 36, requires_grad=True)
    target = torch.zeros(2, 6, 3); target[:, 1, 0] = .3
    active = torch.tensor([[True, True, False, False, False, False]]).expand(2, -1)
    surface = torch.zeros(2, 6, dtype=torch.long)
    rows = [[({'part': p, 'surface': 0, 'body_name': str(p), 'dist': 0,
              'position_w': [p*.2, 0, 0]}, None) for p in (0, 1)] for _ in range(2)]
    loss, _ = frozen_contact_layout_diagnostic(SimpleNamespace(fk=fk), q, target, active, surface, rows)
    assert fk.calls == 1
    rms = .1 / 2**.5
    torch.testing.assert_close(loss, torch.full((2,), (rms/.04-1)**2))
    loss.sum().backward()
    gradient = -.1/.04**2 * (1-.04/rms)
    torch.testing.assert_close(q.grad[:, 0], torch.full((2,), gradient))
    torch.testing.assert_close(q.grad[:, 7], torch.zeros(2))
    torch.testing.assert_close(q.grad[:, 8], torch.full((2,), gradient))


def test_position_loss_has_finite_zero_gradient_inside_and_at_accepted_boundary():
    tolerance = ENDPOINT_POSITION_TOLERANCE_M
    error = torch.tensor([0., (tolerance/2)**2, tolerance**2], dtype=torch.float64, requires_grad=True)
    loss = endpoint_position_loss(error)
    torch.testing.assert_close(loss, torch.zeros_like(error))
    loss.sum().backward()
    torch.testing.assert_close(error.grad, torch.zeros_like(error))


def test_position_loss_uses_the_existing_group_rms_instead_of_per_limb_bounds():
    intended = torch.zeros(1, 2, 3, dtype=torch.float64)
    actual = intended.clone(); actual[0, 0, 0] = .05
    actual.requires_grad_(True)
    error, _ = endpoint_position_statistics(actual, intended, torch.ones(1, 2, dtype=torch.bool))
    # One limb exceeds 4 cm, but the existing two-limb RMS check accepts 3.54 cm.
    assert error.item() < ENDPOINT_POSITION_TOLERANCE_M**2
    loss = endpoint_position_loss(error)
    assert loss.item() == 0
    loss.sum().backward()
    assert actual.grad.abs().sum() == 0


def test_position_loss_outside_range_has_correct_finite_difference_gradient():
    actual = torch.tensor([[[.06, .08, 0.], [.02, .01, 0.]]], dtype=torch.float64, requires_grad=True)
    intended = torch.zeros_like(actual)
    observed = torch.ones(1, 2, dtype=torch.bool)
    def objective(point):
        return endpoint_position_loss(endpoint_position_statistics(point, intended, observed)[0])
    assert torch.autograd.gradcheck(objective, (actual,))
    gradient = torch.autograd.grad(objective(actual).sum(), actual)[0]
    assert gradient[0, 0, 0] > 0 and gradient[0, 0, 1] > 0
    assert gradient[..., 2].abs().sum() == 0


def test_device_and_ordinary_layout_match_loss_gradient_and_missing_observation():
    from contact_solver.device_contact_objective import DeviceWitnessRows
    class FK:
        def link_poses(self, q, names):
            return q[:, None, :3], torch.eye(3).to(q).expand(len(q), 1, 3, 3)
    ordinary_q = torch.zeros(5, 36, dtype=torch.float64, requires_grad=True)
    device_q = ordinary_q.detach().clone().requires_grad_(True)
    witnesses = ordinary_q.new_tensor([[0., 0., 0.], [.02, 0., 0.], [.04, 0., 0.], [.08, 0., 0.]])
    sample = torch.arange(4)
    observed = dict(schema='newton_device_witness_batch_v1', link_names=('foot',), pairs=dict(
        sample=sample, part=torch.zeros(4, dtype=torch.long), body_link0=torch.full((4,), -1),
        body_link1=torch.zeros(4, dtype=torch.long), geometry_point0_w=witnesses.clone(),
        geometry_point1_w=witnesses.clone(), normal_w=ordinary_q.new_tensor([[0., 0., 1.]]).expand(4, -1),
        dist=torch.zeros(4), task_pair=torch.ones(4, dtype=torch.bool),
        primary_surface=torch.zeros(4, dtype=torch.long), eligible=torch.ones(4, dtype=torch.bool)))
    device = DeviceWitnessRows(FK(), device_q, observed)
    rows = [[(dict(part=0, surface=0, body_name='foot', position_w=w.tolist()), None)] for w in witnesses]
    rows.append([])
    points = ordinary_q.new_zeros(5, 6, 3, requires_grad=True)
    active = torch.zeros(5, 6, dtype=torch.bool); active[:, 0] = True
    surfaces = torch.zeros(5, 6, dtype=torch.long)
    ordinary_loss, ordinary_metrics = frozen_contact_layout_diagnostic(
        SimpleNamespace(fk=FK()), ordinary_q, points, active, surfaces, rows)
    device_loss, device_metrics = frozen_contact_layout_diagnostic(
        SimpleNamespace(fk=FK()), device_q, points, active, surfaces, device)
    torch.testing.assert_close(ordinary_loss, ordinary_q.new_tensor([0., 0., 0., 1., 0.]))
    torch.testing.assert_close(device_loss, ordinary_loss)
    for key in ordinary_metrics:
        torch.testing.assert_close(device_metrics[key], ordinary_metrics[key])
    assert ordinary_metrics['relative_layout_observed_parts'].tolist() == [1., 1., 1., 1., 0.]
    ordinary_loss.sum().backward(); device_loss.sum().backward()
    torch.testing.assert_close(device_q.grad, ordinary_q.grad)
    torch.testing.assert_close(ordinary_q.grad[:3], torch.zeros_like(ordinary_q.grad[:3]))
    assert ordinary_q.grad[3, 0] > 0 and points.grad is None
