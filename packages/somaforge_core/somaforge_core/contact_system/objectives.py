"""Smooth geometric guidance on fixed surfaces; never a contact detector."""
from __future__ import annotations

import math
import torch


def region_cost(points, target, normal, *, halfwidth=0., normal_scale=.05,
                tangent_scale=.05, normal_weight=1., tangent_weight=1.):
    """Distance from every candidate to a bounded disk in the target plane.

    A zero radius is an explicit point target. Normals and scales belong to
    the task; this function does not infer an environment face or activation.
    """
    if any(not math.isfinite(x) or x <= 0 for x in (normal_scale, tangent_scale)):
        raise ValueError("Region loss scales must be positive and finite")
    if any(not math.isfinite(x) or x < 0 for x in (normal_weight, tangent_weight)):
        raise ValueError("Region loss weights must be nonnegative and finite")
    if points.shape[-1] != 3 or target.shape != points.shape[:-2]+(3,) or normal.shape != target.shape:
        raise ValueError("Surface/target/normal dimensions disagree")
    if not torch.isfinite(target).all() or not torch.isfinite(normal).all():
        raise ValueError("Nonfinite target geometry")
    if not torch.allclose(normal.norm(dim=-1), torch.ones_like(normal[..., 0]), atol=1e-5):
        raise ValueError("Region targets require unit normals")
    radius = torch.as_tensor(halfwidth, dtype=points.dtype, device=points.device)
    if not torch.isfinite(radius).all() or (radius < 0).any():
        raise ValueError("Region radius must be finite and nonnegative")
    radius = torch.broadcast_to(radius, target.shape[:-1])
    delta = points-target[..., None, :]
    dn = (delta*normal[..., None, :]).sum(-1)
    tangent = delta-dn[..., None]*normal[..., None, :]
    dt = (tangent.norm(dim=-1)-radius[..., None]).clamp_min(0)
    normal_cost = (dn/normal_scale).square()
    tangent_cost = (dt/tangent_scale).square()
    return normal_weight*normal_cost+tangent_weight*tangent_cost, normal_cost, tangent_cost


def smooth_region_loss(points, target, normal, areas, *, temperature=.02, mask=None, **kwargs):
    """Area-normalized log-sum-exp of scalar costs, without averaging points.

    Candidate selection and scoring use the SAME metric. Area normalization
    makes duplication/refinement of quadrature nodes invariant when area is
    conserved. Returned normal/tangent terms are diagnostic expectations; the
    loss also includes the relative-entropy contribution.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Surface temperature must be positive and finite")
    shape = points.shape[:-1]
    valid = torch.ones(shape, device=points.device, dtype=torch.bool) if mask is None else torch.broadcast_to(mask, shape)
    safe_points = torch.where(valid[..., None], points, target[..., None, :])
    cost, normal_cost, tangent_cost = region_cost(safe_points, target, normal, **kwargs)
    area = torch.broadcast_to(areas, cost.shape)
    if not valid.any(-1).all() or not torch.isfinite(area).all() or (area[valid] <= 0).any():
        raise ValueError("Every region requires candidates with positive finite areas")
    if not torch.isfinite(cost[valid]).all():
        raise ValueError("Nonfinite surface candidate cost")
    area = torch.where(valid, area, 0.)
    # Use safe values before log; masked nodes never contribute NaN gradients.
    log_prior = area.clamp_min(torch.finfo(area.dtype).tiny).log()-area.sum(-1, keepdim=True).log()
    logits = (log_prior-cost/temperature).masked_fill(~valid, -torch.inf)
    loss = -temperature*torch.logsumexp(logits, -1)
    weights = logits.softmax(-1)
    n = (weights*normal_cost).sum(-1)
    t = (weights*tangent_cost).sum(-1)
    expected = (weights*cost).sum(-1)
    return dict(loss=loss, normal=n, tangent=t, entropy_cost=loss-expected, weights=weights,
                hard_loss=cost.masked_fill(~valid, torch.inf).amin(-1))


def temporal_region_loss(cost, valid, *, dt, temperature=.02):
    """Smooth first-arrival alternative on an explicitly selected time window.

    This geometric objective does not determine success or add a dwell rule.
    It aggregates full per-frame losses, never contact coordinates. A window
    must stop at episode boundaries; unknown frames cannot be zero-filled.
    """
    if cost.ndim < 1 or valid.shape != cost.shape:
        raise ValueError("Temporal region masks must match [time,...] costs")
    if not valid.any(0).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temporal region loss needs an observed window and positive temperature")
    step = torch.broadcast_to(torch.as_tensor(dt, dtype=cost.dtype, device=cost.device), cost.shape)
    if not torch.isfinite(step).all() or (step <= 0).any() or not torch.isfinite(cost[valid]).all():
        raise ValueError("Invalid temporal costs or sample durations")
    weight = torch.where(valid, step, 0.)
    safe = torch.where(valid, cost, 0.)
    logits = weight.clamp_min(torch.finfo(cost.dtype).tiny).log()-weight.sum(0, keepdim=True).log()-safe/temperature
    return -temperature*torch.logsumexp(logits.masked_fill(~valid, -torch.inf), 0)
