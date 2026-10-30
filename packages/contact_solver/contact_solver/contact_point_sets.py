"""Match actual endpoint contact points without selecting a deepest witness.

Keep compares two same-part, same-surface observed point sets. Establish
compares the endpoint set with one planned point. Missing contacts remain
missing; a separate geometric value can provide approach gradients only.
"""
import torch

CONTACT_SET_SCHEMA = 'newton_matched_endpoint_contact_points_xyz_v2'
PATCH_KEYS = ('contact_patch_points', 'contact_patch_part', 'contact_patch_surface', 'contact_patch_valid')


def contact_patch_observation(observed, reference, capacity=64):
    """Retain every eligible robot-side witness; never silently truncate."""
    b = len(reference)
    out = dict(contact_patch_points=reference.new_zeros(b, capacity, 3),
        contact_patch_part=torch.full((b, capacity), -1, device=reference.device, dtype=torch.long),
        contact_patch_surface=torch.full((b, capacity), -1, device=reference.device, dtype=torch.long),
        contact_patch_valid=torch.zeros(b, capacity, device=reference.device, dtype=torch.bool))
    if observed.get('schema') == 'newton_device_witness_batch_v1':
        p = observed['pairs']
        if bool((p['active'] & ~p['constraint_allocated']).any()):
            raise ValueError('Unallocated active Newton contact')
        good = p['eligible']
        if bool((good & ~(p['task_pair'] & p['active'] & p['constraint_allocated'])).any()):
            raise ValueError('Invalid eligible Newton contact')
        for sample in range(b):
            chosen = ((p['sample'] == sample) & good).nonzero().flatten()
            n = len(chosen)
            if n > capacity:
                raise ValueError('Actual contact patch capacity exceeded')
            out['contact_patch_points'][sample, :n] = p['geometry_point1_w'][chosen].to(reference)
            out['contact_patch_part'][sample, :n] = p['part'][chosen]
            out['contact_patch_surface'][sample, :n] = p['primary_surface'][chosen]
            out['contact_patch_valid'][sample, :n] = True
    else:
        rows = observed['pairs']
        if b == 1 and (not rows or isinstance(rows[0], dict)):
            rows = [rows]
        from somaforge_core.contact_face_selection import select_contact_pairs
        catalogs = observed.get('surface_catalog_by_sample', [observed.get('surface_catalog')]*b)
        for sample, row in enumerate(rows):
            selected = select_contact_pairs([row], catalogs[sample])['contact_pairs'][0]
            if len(selected) > capacity:
                raise ValueError('Actual contact patch capacity exceeded')
            for j, p in enumerate(selected):
                if not (p['constraint_active'] and p['constraint_type'] & 1 and all(a >= 0 for a in p['efc_address'])):
                    raise ValueError('Actual patch requires active allocated contact')
                out['contact_patch_points'][sample, j] = reference.new_tensor(p['position_w'])
                out['contact_patch_part'][sample, j] = p['part']
                out['contact_patch_surface'][sample, j] = p['surface']
                out['contact_patch_valid'][sample, j] = True
    return {k: v.detach() for k, v in out.items()}


def patch_support(observation, surfaces):
    part = torch.arange(6, device=surfaces.device)
    return (observation['contact_patch_valid'][:, None]
        & (observation['contact_patch_part'][:, None] == part[None, :, None])
        & (observation['contact_patch_surface'][:, None] == surfaces[..., None]))


def matched_targets(planned, roles, surfaces, start, endpoint=None):
    """Pick an observed start point for keep supervision/recovery.

    For demonstrations choose the starting point nearest the actual endpoint
    set. For prediction use proximity to the issued point. No future point is
    added to the start observation, and the loss still considers the full set.
    """
    support = patch_support(start, surfaces)
    keep = roles == 2
    if bool((keep & ~support.any(-1)).any()):
        raise ValueError('Keep requires an actual starting part/surface patch')
    points = start['contact_patch_points']
    if endpoint is None:
        cost = (points[:, None]-planned[:, :, None]).square().sum(-1)
    else:
        end_support = patch_support(endpoint, surfaces)
        distances = torch.cdist(points, endpoint['contact_patch_points']).square()
        cost = distances[:, None].masked_fill(~end_support[:, :, None], torch.inf).amin(-1)
        # A missing demonstrated endpoint is explicitly incompatible elsewhere;
        # this deterministic fallback only supplies a finite location label.
        fallback = (points[:, None]-planned[:, :, None]).square().sum(-1)
        cost = torch.where(end_support.any(-1)[..., None], cost, fallback)
    chosen = cost.masked_fill(~support, torch.inf).argmin(-1)
    target = points.gather(1, chosen[..., None].expand(-1, -1, 3))
    return torch.where(keep[..., None], target, planned).detach()


def matched_part_statistics(endpoint, planned, roles, surfaces, start, *, points=None):
    """Minimum observed point-pair error per active contact relation.

    Duplicating/reordering witnesses does not change the value. The zero set
    means at least one common contact location, not stationary whole-patch
    material and not proof of temporal contact continuity.
    """
    active, keep = roles != 0, roles == 2
    end_support = patch_support(endpoint, surfaces)
    start_support = patch_support(start, surfaces)
    if bool((keep & ~start_support.any(-1)).any()):
        raise ValueError('Keep requires an actual starting part/surface patch')
    end_points = endpoint['contact_patch_points'] if points is None else points
    target_cost = (end_points[:, None]-planned.detach()[:, :, None]).square().sum(-1)
    distances = (end_points[:, :, None]-start['contact_patch_points'].detach()[:, None]).square().sum(-1)
    keep_cost = distances[:, None].masked_fill(~start_support[:, :, None], torch.inf).amin(-1)
    cost = torch.where(keep[..., None], keep_cost, target_cost)
    present = active & end_support.any(-1)
    error = cost.masked_fill(~end_support, torch.inf).amin(-1)
    return torch.where(present, error, 0.), present


def summarize_parts(error, active, present):
    count = present.sum(-1)
    complete = active.any(-1) & (present | ~active).all(-1)
    return (error*active).sum(-1)/active.sum(-1).clamp_min(1), complete, count


def observed_matched_statistics(observed, targets, roles, surfaces, start):
    endpoint = contact_patch_observation(observed, targets)
    error, present = matched_part_statistics(endpoint, targets, roles, surfaces, start)
    return summarize_parts(error, roles != 0, present)


def differentiable_matched_parts(rows, targets_world, roles, surfaces, start):
    endpoint = contact_patch_observation(rows.observed, rows.q)
    # Packing order is the eligible solver row order in each sample. Bind only
    # the freshly queried endpoint witnesses, never starting material points.
    values = endpoint['contact_patch_points'].clone()
    moving = rows.moving[:, 1]
    if rows.world_frame is not None:
        origin, basis = rows.world_frame
        moving = origin[rows.sample]+torch.einsum('nij,nj->ni', basis[rows.sample], moving)
    for sample in range(len(rows)):
        chosen = ((rows.sample == sample) & rows.pair['eligible']).nonzero().flatten()
        values[sample, :len(chosen)] = moving[chosen]
    # Keep a zero gradient connection for snapshots with no actual contacts.
    values = values+rows.q.sum()*0
    return matched_part_statistics(endpoint, targets_world, roles, surfaces, start, points=values)


def matched_endpoint_statistics(model, router, rows, scene_ids, targets, roles, surfaces,
                                start, *, regions=None, approach=False):
    error, present = differentiable_matched_parts(rows, targets, roles, surfaces, start)
    active = roles != 0
    if approach and bool((active & ~present).any()):
        # Shape-to-target interval is loss geometry only. Never turn proximity
        # into observed contact or accept a missing pair.
        recovery_targets = matched_targets(targets, roles, surfaces, start)
        geometric = router(model.fk, rows.q, scene_ids, recovery_targets, active & ~present,
            surfaces, regions, model.region_geometry, world_frame=rows.world_frame)
        error = torch.where(present, error, geometric)
    return summarize_parts(error, active, present)


def matched_endpoint_objective_terms(model, router, rows, scene_ids, targets, roles,
                                     surfaces, start, *, regions=None):
    """Actual XYZ endpoint error plus one regional recovery geometry query.

    The recovery is consumed ONLY by the unified contact objective. It never
    contributes to endpoint position loss or changes observed completeness.
    The normal segment uses the actual scene's configured margin; distance
    retains all three dimensions, including excess above/below that segment.
    """
    error, present = differentiable_matched_parts(rows, targets, roles, surfaces, start)
    active = roles != 0
    missing = active & ~present
    recovery = rows.q.sum(-1)[:, None, None].expand(-1, 6, 4)*0
    if bool(missing.any()):
        recovery = router(model.fk, rows.q, scene_ids,
            matched_targets(targets, roles, surfaces, start), missing, surfaces,
            regions, model.region_geometry, world_frame=rows.world_frame, per_region=True)
    return summarize_parts(error, active, present), recovery, missing


def reference_keep_compatible(endpoint, roles, surfaces, start, tolerance_m):
    """Detect, report and mask conflicting pose imitation; never relabel keep."""
    dummy = endpoint['contact_patch_points'].new_zeros(len(roles), 6, 3)
    error, present = matched_part_statistics(endpoint, dummy, roles, surfaces, start)
    conflict = (roles == 2) & (~present | (error > tolerance_m**2))
    return ~conflict.any(-1), conflict
