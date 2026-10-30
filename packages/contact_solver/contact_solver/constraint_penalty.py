"""A common scalar penalty for dimensionless squared constraint residuals."""
import torch


CONSTRAINT_PENALTY_SCHEMA = 'log1p_squared_residual_v1'


def constraint_penalty(squared_residual):
    """Preserve zero bands and their true FK derivatives without a quadratic tail.

    Every caller supplies its own physical residual and normalization. Applying
    the same monotone penalty before aggregation prevents an unbounded keep
    cost from overwhelming a robust contact/clearance cost. This is a scalar
    objective, not a backward-only correction or a feasibility guarantee.
    """
    return torch.log1p(squared_residual)
