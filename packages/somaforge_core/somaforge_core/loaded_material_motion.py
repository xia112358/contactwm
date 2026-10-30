"""Source-load-weighted tangential material motion; never contact truth.

Each solve site is followed through TWO adjacent poses of its own rigid link.
Different sites are reduced by load-weighted RMS, not a pivot minimum. The
sites and loads are references after editing; fresh execution loads remain
unknown. Endpoint-only predictions cannot supply a phase path.
"""
import numpy as np
import torch

MATERIAL_MOTION_SCHEMA = 'source_loaded_tangent_material_motion_v1'
DEFAULT_MATERIAL_PATH_BUDGET_M = .06
DEFAULT_MATERIAL_RESIDUAL_SCALE_M = .01  # Loss normalization only, not a contact/slip tolerance.
DEFAULT_SOURCE_ADDED_MOTION_WEIGHT = .1  # Soft source preservation, never an acceptance gate.


def phase_material_paths(steps, phases, *, load_known, load_bearing):
    """Differentiable cumulative paths, with unknown load coverage kept explicit."""
    known = torch.as_tensor(load_known, device=steps.device, dtype=torch.bool)
    bearing = torch.as_tensor(load_bearing, device=steps.device, dtype=torch.bool)
    if steps.ndim != 2 or steps.shape[1] != 6 or known.shape != steps.shape or bearing.shape != steps.shape:
        raise ValueError('Material path/load clock dimensions differ')
    rows, complete = [], []
    for phase in phases:
        a, b, p = (int(phase[k]) for k in ('start', 'end', 'part'))
        if not 0 <= a < b <= len(steps) or not 0 <= p < 6:
            raise ValueError('Material phase outside reference clock')
        values, measured, on = steps[a:b, p], known[a:b, p], bearing[a:b, p]
        valid = measured & on & torch.isfinite(values)
        rows.append(torch.where(valid, values, 0.).sum())
        complete.append(measured.all() & (~on | valid).all())
    return (torch.stack(rows) if rows else steps.new_empty(0),
            torch.stack(complete) if complete else torch.empty(0, device=steps.device, dtype=torch.bool))


def tangent_material_rms(delta, normals, loads, *, xp):
    """Dense [..., sites, 3] reduction shared by Torch, NumPy and JAX.

    Zero-weight padded rows return zero; the caller must retain the independent
    load coverage mask. Guard the square root so stationary sites have finite
    derivatives, without introducing a motion threshold.
    """
    tangent = delta-xp.sum(delta*normals, axis=-1)[..., None]*normals
    weight = xp.sum(loads, axis=-1)
    square = xp.sum(loads*xp.sum(tangent*tangent, axis=-1), axis=-1)
    mean = square/xp.where(weight > 0, weight, 1.)
    return xp.where(mean > 0, xp.sqrt(xp.where(mean > 0, mean, 1.)), mean*0.)


def phase_added_motion(steps, source_steps, phases, *, load_known, load_bearing):
    """Differentiable positive per-interval excess; no cross-time cancellation.

    Unknown intervals do not become zero-motion certificates: completeness is
    returned separately, and loss callers must reject incomplete references.
    """
    source = torch.as_tensor(source_steps, device=steps.device, dtype=steps.dtype)
    known = torch.as_tensor(load_known, device=steps.device, dtype=torch.bool)
    bearing = torch.as_tensor(load_bearing, device=steps.device, dtype=torch.bool)
    if source.shape != steps.shape or known.shape != steps.shape or bearing.shape != steps.shape:
        raise ValueError('Added material motion/load clock dimensions differ')
    rows, complete = [], []
    for phase in phases:
        a, b, p = (int(phase[k]) for k in ('start', 'end', 'part'))
        if not 0 <= a < b <= len(steps) or not 0 <= p < 6:
            raise ValueError('Material phase outside reference clock')
        measured, on = known[a:b, p], bearing[a:b, p]
        valid = measured & on & torch.isfinite(source[a:b, p]) & torch.isfinite(steps[a:b, p])
        difference = torch.where(valid, steps[a:b, p], 0.)-torch.where(valid, source[a:b, p], 0.)
        rows.append(difference.relu().sum())
        complete.append(measured.all() & (~on | valid).all())
    return (torch.stack(rows) if rows else steps.new_empty(0),
            torch.stack(complete) if complete else torch.empty(0, device=steps.device, dtype=torch.bool))


def source_motion_allowances(source_steps, phases, *, load_known, load_bearing):
    """One median + 3 MAD source step per phase; never a contact threshold."""
    reference = material_path_report(source_steps, phases, load_known=load_known,
                                     load_bearing=load_bearing)
    rows = []
    for phase, report in zip(phases, reference['phases']):
        a, b, p = (int(phase[k]) for k in ('start', 'end', 'part'))
        valid = np.asarray(load_known)[a:b, p] & np.asarray(load_bearing)[a:b, p]
        values = np.asarray(source_steps)[a:b, p][valid]
        median = float(np.median(values)) if len(values) and report['measurement_complete'] else None
        allowance = None if median is None else median+3*float(np.median(abs(values-median)))
        rows.append(dict(**phase, allowance_m=allowance,
                         measurement_complete=report['measurement_complete'],
                         loaded_intervals=int(valid.sum())))
    return rows


def unknown_material_path(*, reason='endpoint_only_without_executed_trajectory',
                          budget_m=DEFAULT_MATERIAL_PATH_BUDGET_M):
    return dict(schema=MATERIAL_MOTION_SCHEMA, status='unknown', reason=reason,
                material_tangent_path_m=None, budget_m=budget_m,
                within_budget=None, actual_support_status='unknown',
                scope='original_loaded_site_geometry_reference_not_execution_truth')


def loaded_material_steps(positions, rotations, *, frames, links, parts, local,
                          normals, loads, surfaces):
    """Differentiable [T-1,6] RMS steps; no loaded site means NaN, not zero."""
    if positions.ndim != 3 or positions.shape[-1] != 3 or len(positions) < 2:
        raise ValueError('Loaded material motion requires an adjacent-pose trajectory')
    if rotations.shape != (*positions.shape[:-1], 3, 3):
        raise ValueError('Loaded material pose dimensions differ')
    device, dtype = positions.device, positions.dtype
    frames, links, parts, surfaces = [torch.as_tensor(x, device=device, dtype=torch.long)
                                    for x in (frames, links, parts, surfaces)]
    local, normals, loads = [torch.as_tensor(x, device=device, dtype=dtype)
                            for x in (local, normals, loads)]
    n = len(frames)
    if (any(x.shape != (n,) for x in (links, parts, surfaces, loads))
            or local.shape != (n, 3) or normals.shape != (n, 3)):
        raise ValueError('Loaded material site dimensions differ')
    if (bool(((frames < 0) | (frames >= len(positions)-1)).any())
            or bool(((links < 0) | (links >= positions.shape[1])).any())
            or bool(((parts < 0) | (parts >= 6) | (surfaces < 0)).any())):
        raise ValueError('Loaded material site mapping outside trajectory')
    if (not bool(torch.isfinite(positions).all() & torch.isfinite(rotations).all()
                 & torch.isfinite(local).all() & torch.isfinite(normals).all()
                 & torch.isfinite(loads).all()) or bool((loads <= 0).any())):
        raise ValueError('Invalid positive-load material reference')
    if not bool(torch.isclose(normals.norm(dim=-1), torch.ones(n, device=device, dtype=dtype),
                              atol=1.e-4, rtol=0).all()):
        raise ValueError('Unknown material surface normal')
    groups = frames*6+parts
    # A changing heel/toe is allowed; combining different terrain faces is not.
    if len(torch.unique(torch.stack((groups, surfaces), -1), dim=0)) != len(torch.unique(groups)):
        raise ValueError('Loaded endpoint spans different terrain surfaces')
    delta = positions[frames+1, links]-positions[frames, links]+torch.einsum(
        'nij,nj->ni', rotations[frames+1, links]-rotations[frames, links], local)
    tangent = delta-(delta*normals).sum(-1, keepdim=True)*normals
    size = (len(positions)-1)*6
    weight = positions.new_zeros(size).scatter_add(0, groups, loads)
    square = positions.new_zeros(size).scatter_add(0, groups, loads*tangent.square().sum(-1))
    mean = square/weight.clamp_min(torch.finfo(dtype).tiny)
    # Finite zero-motion derivatives, including an exactly stationary pivot.
    rms = torch.where(mean > 0, mean.clamp_min(torch.finfo(dtype).tiny).sqrt(), mean*0)
    return torch.where(weight > 0, rms, torch.full_like(rms, float('nan'))).reshape(-1, 6)


def material_path_report(steps, phases, *, load_known, load_bearing,
                         budget_m=DEFAULT_MATERIAL_PATH_BUDGET_M):
    """Apply a geometric path budget to explicit reference phases.

    Known unloaded samples are excluded. Unknown observations or missing
    samples at loaded sites cannot establish a pass. No off sample is treated
    as a new contact event, and no single endpoint is fixed across a transfer.
    """
    steps = np.asarray(steps, float)
    known, bearing = np.asarray(load_known, bool), np.asarray(load_bearing, bool)
    if steps.ndim != 2 or steps.shape[1] != 6 or known.shape != steps.shape or bearing.shape != steps.shape:
        raise ValueError('Material path/load clock dimensions differ')
    if not np.isfinite(budget_m) or budget_m <= 0:
        raise ValueError('Positive finite material path budget required')
    rows = []
    for phase in phases:
        a, b, p = (int(phase[k]) for k in ('start', 'end', 'part'))
        if not 0 <= a < b <= len(steps) or not 0 <= p < 6:
            raise ValueError('Material phase outside reference clock')
        values, on, measured = steps[a:b, p], bearing[a:b, p], known[a:b, p]
        finite = np.isfinite(values)
        if np.any(finite & (values < 0)):
            raise ValueError('Negative material motion')
        usable = on & measured & finite
        complete = bool(measured.all() and finite[on].all())
        path = float(values[usable].sum())
        verdict = bool(path <= budget_m) if complete and bool(on.any()) else None
        rows.append(dict(**phase, material_tangent_path_m=path,
            budget_m=budget_m, measured_loaded_intervals=int(usable.sum()),
            unloaded_intervals=int((measured & ~on).sum()),
            unknown_intervals=int((~measured | (on & ~finite)).sum()),
            measurement_complete=complete, within_budget=verdict,
            actual_edited_support='unknown_without_new_execution'))
    return dict(schema=MATERIAL_MOTION_SCHEMA,
        scope='original_loaded_site_geometry_reference_not_execution_truth',
        budget_m=budget_m, phases=rows,
        within_budget=(None if not rows or any(r['within_budget'] is None for r in rows)
                       else all(r['within_budget'] for r in rows)))
