"""Project updates into the recorded successful-policy behavior trust region.

The caller supplies squared per-trajectory action RMS errors relative to fixed
successful executions. These are policy-function constraints, not contact truth.
The nonlinear policy forward pass must verify the proposed step afterward.
"""
from collections.abc import Callable

import torch

from .contact_projection_gpu import solve_nonnegative_qp


@torch.no_grad()
def project_behavior_step(proposed, gradients, squared_errors, max_rms):
    """Consume the gradient buffer and spend half the remaining linear budget.

    The current policy must be feasible. At the unchanged anchor, squared-error
    derivatives are zero; an already feasible proposal needs no QP solve.
    """
    norms = gradients.norm(dim=1)
    unit = gradients.div_(norms[:, None].clamp_min(1e-20))
    allowance = .5 * (max_rms**2 - squared_errors) / norms.clamp_min(1e-20)
    linear = unit @ proposed - allowance
    if not bool((linear > 5e-8).any()):
        return proposed
    multipliers, _ = solve_nonnegative_qp(
        unit @ unit.T, linear, tolerance=5e-8, graph=unit.is_cuda)
    return proposed - unit.T @ multipliers


@torch.no_grad()
def verify_behavior_step(proposed, measure_rms: Callable[[torch.Tensor], float], max_rms):
    """Backtrack a projected step against the actual policy, from a feasible state.

    Return the nonzero-scale step, its scale, and the measured maximum RMS.
    A violated feasibility contract raises instead of silently rejecting updates.
    """
    scale = 1.
    for _ in range(24):
        candidate = proposed * scale
        measured = measure_rms(candidate)
        if measured <= max_rms:
            return candidate, scale, measured
        scale *= .5
    raise RuntimeError('No feasible policy-function update in the trust region')


@torch.no_grad()
def project_behavior_quadratic(proposed, jacobian, residual, penalty):
    """Closest update with a quadratic penalty on anchored behavior change.

    Minimize ||step-proposed||^2 / 2 + penalty * ||J step + residual||^2 / 2.
    The positive penalty is an optimizer parameter; J may be a weighted sketch
    of actual successful-policy output derivatives. No contact rule is changed.
    """
    gram = jacobian @ jacobian.T
    system = gram + torch.eye(len(gram), device=gram.device, dtype=gram.dtype) / penalty
    correction = torch.linalg.solve(system, jacobian @ proposed + residual)
    return proposed - jacobian.T @ correction
