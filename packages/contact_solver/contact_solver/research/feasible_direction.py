"""Preserved experiment helpers for regression tests; no training CLI."""
import numpy as np
import torch
from scipy.optimize import minimize

def constrained_direction(gradient, clearance_gradients, clearance, *, learning_rate=0.0001):
    """Nearest task descent step satisfying linearized clearance constraints.

Feasible examples may approach a surface but not cross zero; existing negative
clearance must improve by 20% in the linear model. Nonlinear acceptance still
requires independent geometry and fresh Newton queries, with zero tolerance.
"""
    g = gradient.detach().double()
    proposed = -learning_rate * g * (10 / g.norm().clamp_min(10))
    matrix = clearance_gradients.detach().double()
    c = clearance.detach().double()
    norms = matrix.norm(dim=1)
    if ((norms == 0) & (c < 0)).any():
        raise ValueError('Violated constraint has no parameter gradient')
    keep = norms > 0
    matrix = matrix[keep] / norms[keep, None]
    c = c[keep]
    bounds = -c * torch.where(c < 0, 0.2, 1.0) / norms[keep]
    if len(matrix) == 0:
        return (proposed.to(gradient), dict(constraints=0))
    gram = (matrix @ matrix.T).numpy()
    rhs = (bounds - matrix @ proposed).numpy()
    result = minimize(lambda z: (0.5 * z @ gram @ z - z @ rhs, gram @ z - rhs), np.zeros(len(rhs)), jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(rhs), options=dict(maxiter=1000, ftol=1e-15, gtol=1e-12))
    step = proposed + matrix.T @ torch.from_numpy(result.x)
    residual = float((bounds - matrix @ step).clamp_min(0).max())
    if residual > 1e-07:
        raise ValueError(f'Linear feasibility QP failed: {residual}')
    return (step.to(gradient), dict(constraints=len(rhs), linear_residual=residual, task_step_norm=float(proposed.norm()), constrained_step_norm=float(step.norm()), solver_success=bool(result.success)))
