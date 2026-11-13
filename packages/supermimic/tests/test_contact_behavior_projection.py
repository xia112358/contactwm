import pytest
import torch

from supermimic.contact_behavior_projection import project_behavior_step, verify_behavior_step


def test_remaining_behavior_budget_allows_progress_and_preserves_free_direction():
    proposed = torch.tensor([1., -1.], dtype=torch.float64)
    gradients = torch.tensor([[1., 0.]], dtype=torch.float64)
    actual = project_behavior_step(proposed, gradients, torch.tensor([.04]), .3)
    torch.testing.assert_close(actual, torch.tensor([.025, -1.], dtype=torch.float64), atol=1e-6, rtol=0)


def test_boundary_protection_keeps_nonconflicting_progress():
    proposed = torch.tensor([1., -2.], dtype=torch.float64)
    actual = project_behavior_step(proposed, torch.eye(2, dtype=torch.float64), torch.tensor([.09, .09]), .3)
    torch.testing.assert_close(actual, torch.tensor([0., -2.], dtype=torch.float64), atol=1e-6, rtol=0)


def test_zero_anchor_derivative_still_requires_actual_policy_verification():
    proposed = torch.ones(2)
    actual = project_behavior_step(proposed, torch.zeros(1, 2), torch.zeros(1), .3)
    torch.testing.assert_close(actual, proposed)
    verified, scale, measured = verify_behavior_step(actual, lambda step: float(step[0].abs()), .3)
    torch.testing.assert_close(verified, torch.full((2,), .25))
    assert scale == .25 and measured <= .3


def test_empty_protection_set_keeps_proposal():
    proposed = torch.tensor([1., -2.])
    actual = project_behavior_step(proposed, torch.empty(0, 2), torch.empty(0), .1)
    torch.testing.assert_close(actual, proposed)


def test_violated_feasible_state_contract_does_not_silently_drop_update():
    with pytest.raises(RuntimeError, match='No feasible'):
        verify_behavior_step(torch.ones(2), lambda step: 1., .1)


def test_quadratic_behavior_projection_matches_full_parameter_space_solution():
    from supermimic.contact_behavior_projection import project_behavior_quadratic
    gen = torch.Generator().manual_seed(84)
    jacobian = torch.randn(5, 9, generator=gen, dtype=torch.float64)
    proposed = torch.randn(9, generator=gen, dtype=torch.float64)
    residual = torch.randn(5, generator=gen, dtype=torch.float64)
    penalty = 3.
    actual = project_behavior_quadratic(proposed, jacobian, residual, penalty)
    expected = torch.linalg.solve(torch.eye(9) + penalty * jacobian.T @ jacobian,
                                  proposed - penalty * jacobian.T @ residual)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_quadratic_projection_preserves_progress_outside_old_behavior():
    from supermimic.contact_behavior_projection import project_behavior_quadratic
    actual = project_behavior_quadratic(torch.tensor([1., -2.]),
        torch.tensor([[1., 0.]]), torch.zeros(1), 9.)
    torch.testing.assert_close(actual, torch.tensor([.1, -2.]), atol=1e-6, rtol=0)


def test_quadratic_projection_repairs_accumulated_anchor_drift():
    from supermimic.contact_behavior_projection import project_behavior_quadratic
    actual = project_behavior_quadratic(torch.zeros(2),
        torch.tensor([[1., 0.]]), torch.tensor([.2]), 9.)
    torch.testing.assert_close(actual, torch.tensor([-.18, 0.]), atol=1e-6, rtol=0)


def test_quadratic_projection_supports_no_protected_behavior():
    from supermimic.contact_behavior_projection import project_behavior_quadratic
    proposed = torch.tensor([1., -2.])
    actual = project_behavior_quadratic(proposed, torch.empty(0, 2), torch.empty(0), 1.)
    torch.testing.assert_close(actual, proposed)
