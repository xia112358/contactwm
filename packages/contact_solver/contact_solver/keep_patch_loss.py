"""Kinematic keep-intent regularization over actual Newton material patches.

This is an endpoint geometry loss, not an executed support/slip measurement.
The minimum tangential motion over the convex material patch allows rotation
about a point in that patch, instead of freezing an arbitrary witness or link
origin. Contact activation, load-bearing and full-trajectory budgets are separate.
"""
from dataclasses import dataclass

import torch
from contact_solver.constraint_penalty import constraint_penalty

from somaforge_core.loaded_material_motion import DEFAULT_MATERIAL_RESIDUAL_SCALE_M


@dataclass
class KeepPatch:
    sample: torch.Tensor
    part: torch.Tensor
    link: torch.Tensor
    local: torch.Tensor
    position: torch.Tensor
    normal: torch.Tensor
    link_names: tuple[str, ...]
    batch_size: int


def bind_keep_patch(fk, current_q, observed, *, sample_indices=None):
    """Freeze geometry materials from allocated, task-eligible actual contacts."""
    if observed.get('schema') != 'newton_device_witness_batch_v1':
        raise ValueError('Keep patches require actual device Newton witnesses')
    p = observed['pairs']
    if bool((p['active'] & ~p['constraint_allocated']).any()):
        raise ValueError('Unallocated active Newton contact in keep patch')
    valid = p['eligible']
    if sample_indices is not None:
        mapping = torch.full((len(current_q),), -1, dtype=torch.long, device=current_q.device)
        mapping[sample_indices] = torch.arange(len(sample_indices), device=current_q.device)
        valid = valid & (mapping[p['sample']] >= 0)
    if bool((valid & (~p['active'] | ~p['constraint_allocated'] |
                      (p['body_link0'] >= 0) | (p['body_link1'] < 0))).any()):
        raise ValueError('Invalid terrain/robot mapping in keep patch')
    sample, part, link = (p[k][valid].detach() for k in ('sample', 'part', 'body_link1'))
    point = p['geometry_point1_w'][valid].detach().to(current_q)
    surface = p['primary_surface'][valid]
    face = observed['face_surface'][sample] == surface[:, None]
    if bool((face.sum(-1) != 1).any()):
        raise ValueError('Keep patch has missing/ambiguous actual primary face')
    normal = observed['face_normal'][sample, face.long().argmax(-1)].detach().to(current_q)
    if sample_indices is not None:
        sample = mapping[sample]
        current_q = current_q[sample_indices]
    if not bool(torch.isfinite(point).all() & torch.isfinite(normal).all()):
        raise ValueError('Nonfinite keep patch geometry')
    if not bool(torch.isclose(normal.norm(dim=-1), torch.ones_like(normal[:, 0]), atol=1e-6, rtol=0).all()):
        raise ValueError('Invalid actual keep-patch face normal')
    # Only the shared task selector decides eligible contacts. No distance or
    # force threshold is introduced here.
    names = tuple(observed['link_names'])
    with torch.no_grad():
        position, rotation = fk.link_poses(current_q.detach(), names)
        local = torch.einsum('nji,nj->ni', rotation[sample, link], point-position[sample, link])
    return KeepPatch(sample, part, link, local, point, normal, names, len(current_q))


def patch_minimum_motion(delta, groups, size):
    """Distance to the convex hull of 2D material displacements, without a solver.

    All edges are considered. If the origin lies inside the convex hull, the
    distance is zero; otherwise its closest point lies on one of these edges.
    Duplicate witnesses do not change this quantity. No particular site is
    designated as the support point.
    """
    if delta.ndim != 2 or delta.shape[-1] != 2:
        raise ValueError('Expected tangential material displacements [N,2]')
    counts = torch.bincount(groups, minlength=size)
    present = counts > 0
    if len(delta) == 0:
        return delta.sum()*torch.zeros(size, device=delta.device), present
    order = torch.argsort(groups, stable=True)
    starts = counts.cumsum(0)-counts
    slots = torch.arange(len(groups), device=groups.device)-starts[groups[order]]
    width = int(counts.max())
    cloud = delta.new_zeros((size, width, 2))
    cloud[groups[order], slots] = delta[order]
    valid = torch.arange(width, device=groups.device)[None] < counts[:, None]
    a, b = cloud[:, :, None], cloud[:, None, :]
    edge = b-a
    denominator = edge.square().sum(-1)
    t = (-a*edge).sum(-1)/denominator.clamp_min(torch.finfo(delta.dtype).tiny)
    nearest = a+t.clamp(0, 1)[..., None]*edge
    squared = nearest.square().sum(-1).masked_fill(~(valid[:, :, None] & valid[:, None, :]), torch.inf)
    minimum = squared.flatten(1).amin(-1)
    # Origin inside a planar convex hull iff angular samples are not contained
    # in an open semicircle. This discrete branch requires no derivative.
    angles = torch.atan2(cloud[..., 1].detach(), cloud[..., 0].detach()).masked_fill(~valid, torch.inf)
    angles = angles.sort(-1).values
    gaps = (angles[:, 1:]-angles[:, :-1]).masked_fill(~valid[:, 1:], 0)
    last = angles.gather(1, (counts-1).clamp_min(0)[:, None]).squeeze(1)
    wrap = 2*torch.pi-last+angles[:, 0]
    inside = (counts >= 3) & (torch.maximum(gaps.amax(-1) if width > 1 else wrap*0, wrap) <= torch.pi)
    minimum = torch.where(present & ~inside, minimum, 0)
    # sqrt(0) must not inject NaNs into otherwise valid zero-loss gradients.
    motion = torch.where(minimum > 0, minimum.clamp_min(torch.finfo(delta.dtype).tiny).sqrt(), 0)
    return motion, present


def keep_patch_motion(fk, predicted_q, patch):
    """Tangential endpoint motion of the same frozen material patch [B,6]."""
    position, rotation = fk.link_poses(predicted_q, patch.link_names)
    moved = position[patch.sample, patch.link] + torch.einsum(
        'nij,nj->ni', rotation[patch.sample, patch.link], patch.local)
    delta = moved-patch.position
    # Build an orthonormal tangent basis, also for non-axis-aligned test faces.
    axis = torch.nn.functional.one_hot(patch.normal.abs().argmin(-1), 3).to(delta)
    u = torch.nn.functional.normalize(torch.cross(patch.normal, axis, dim=-1), dim=-1)
    v = torch.cross(patch.normal, u, dim=-1)
    planar = torch.stack(((delta*u).sum(-1), (delta*v).sum(-1)), -1)
    motion, present = patch_minimum_motion(planar, patch.sample*6+patch.part, patch.batch_size*6)
    return motion.reshape(-1, 6), present.reshape(-1, 6)


def keep_patch_loss(fk, predicted_q, roles, patch, allowance_m,
                    *, scale_m=DEFAULT_MATERIAL_RESIDUAL_SCALE_M):
    """Penalize motion beyond demonstration-calibrated allowance for keep only."""
    if scale_m <= 0 or bool((allowance_m < 0).any()):
        raise ValueError('Invalid keep patch residual scale/allowance')
    motion, present = keep_patch_motion(fk, predicted_q, patch)
    keep = roles.detach() == 2
    active = keep & present & torch.isfinite(allowance_m)
    excess = torch.relu(motion-allowance_m)
    loss = (constraint_penalty((excess/scale_m).square())*active).sum(-1)/active.sum(-1).clamp_min(1)
    metrics = dict(keep_patch_loss=loss, keep_patch_motion_m=motion,
        keep_patch_excess_m=excess*active, keep_patch_present=present,
        keep_patch_missing=(keep & ~present).sum(-1).to(predicted_q))
    return loss, metrics


def calibrated_keep_allowance(reference_motion, reference_present, roles, training_indices):
    """Own-action allowances; other roles use training-only keep statistics."""
    keep = roles == 2
    fallback = reference_motion.new_zeros(6)
    known = torch.zeros(6, device=reference_motion.device, dtype=torch.bool)
    for part in range(6):
        selected = keep[training_indices, part] & reference_present[training_indices, part]
        values = reference_motion[training_indices, part][selected]
        if len(values):
            fallback[part] = values.median()
            known[part] = True
    allowance = torch.where(keep & reference_present, reference_motion, fallback[None])
    allowance = torch.where((keep & reference_present) | known[None], allowance, torch.inf)
    return allowance.detach(), fallback.detach(), known


def keep_allowance_for_observation(reference_allowance, fallback, fallback_known, supervised):
    """Autonomous observations have no demonstrated action allowance."""
    autonomous = torch.where(fallback_known, fallback, torch.inf)
    return torch.where(supervised[:, None], reference_allowance, autonomous[None]).detach()
