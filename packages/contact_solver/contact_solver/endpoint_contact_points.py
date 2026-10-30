"""Actual endpoint contact-point error, shared by establish and keep.

Only fresh, eligible Newton witnesses are observations. This endpoint metric
cannot certify uninterrupted contact between the two queried poses.
"""
import torch

CONTACT_POINT_SCHEMA = 'newton_endpoint_contact_points_xyz_v1'


def relation_points(observed, surface_ids, reference):
    """Actual point per part/surface, selected by minimum dist-includemargin.

    This matches the Newton contact aggregation, independently of the target.
    The nested-record path receives the shared primary-face selector's output.
    """
    from contact_solver.device_contact_objective import first_minimum
    b, f = surface_ids.shape
    points = reference.new_zeros(b, 6, f, 3)
    present = torch.zeros(b, 6, f, device=reference.device, dtype=torch.bool)
    if observed.get('schema') == 'newton_device_witness_batch_v1':
        p = observed['pairs']
        if bool((p['active'] & ~p['constraint_allocated']).any()):
            raise ValueError('Unallocated active contact in endpoint point observation')
        for face in range(f):
            valid = (p['eligible'] & p['task_pair'] & p['active'] & p['constraint_allocated']
                     & (p['primary_surface'] == surface_ids[p['sample'], face])
                     & (surface_ids[p['sample'], face] >= 0))
            chosen, known = first_minimum(p['dist']-p['includemargin'],
                p['sample']*6+p['part'].clamp(0, 5), valid, b*6)
            values = torch.cat((p['geometry_point1_w'], reference.new_zeros(1, 3)))
            points[:, :, face] = values[chosen].reshape(b, 6, 3)
            present[:, :, face] = known.reshape(b, 6)
    else:
        rows = observed['pairs']
        if b == 1 and (not rows or isinstance(rows[0], dict)):
            rows = [rows]
        for sample, row in enumerate(rows):
            for part in range(6):
                for face, surface in enumerate(surface_ids[sample].tolist()):
                    matches = [p for p in row if int(p['part']) == part and int(p['surface']) == surface]
                    if not matches:
                        continue
                    p = min(matches, key=lambda p: float(p['dist'])-float(p['includemargin']))
                    if not (p['constraint_active'] and p['constraint_type'] & 1
                            and all(a >= 0 for a in p['efc_address'])):
                        raise ValueError('Endpoint point requires active, allocated Newton contact')
                    points[sample, part, face] = reference.new_tensor(p['position_w'])
                    present[sample, part, face] = True
    return points.detach(), present


def contact_targets(planned_points, roles, surfaces, observation):
    """Keep targets the start point on the same surface; other roles the plan."""
    keep = roles == 2
    match = (observation['current_contact_relations']
             & (observation['contact_surface_ids'][:, None] == surfaces[..., None]))
    if bool((keep & ~match.any(-1)).any()):
        raise ValueError('Keep endpoint target requires actual starting part/surface contact')
    index = match.long().argmax(-1)
    start = observation['contact_relation_points'].gather(2,
        index[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    return torch.where(keep[..., None], start, planned_points).detach()


def point_statistics(actual_points, targets, active, present):
    known = active & present
    count = known.sum(-1)
    delta = torch.where(known[..., None], actual_points-targets.detach(), 0.)
    error = delta.square().sum((-1, -2))/count.clamp_min(1)
    complete = active.any(-1) & (present | ~active).all(-1)
    return error, complete, count


def endpoint_contact_statistics(rows, targets_world, active, surfaces):
    """Fresh endpoint robot witnesses with FK gradients at that endpoint only."""
    from contact_solver.device_contact_objective import first_minimum
    p = rows.pair
    match = (p['eligible'] & p['task_pair'] & p['active'] & p['constraint_allocated']
             & active[rows.sample, rows.part]
             & (p['primary_surface'] == surfaces[rows.sample, rows.part]))
    chosen, present = first_minimum(p['dist']-p['includemargin'], rows.group, match, len(rows)*6)
    values = torch.cat((rows.moving[:, 1], rows.q[:1, :3]*0))
    actual = values[chosen].reshape(len(rows), 6, 3)
    targets = targets_world.detach()
    if rows.world_frame is not None:
        origin, basis = rows.world_frame
        targets = torch.einsum('bji,bpj->bpi', basis, targets-origin[:, None])
    return point_statistics(actual, targets, active, present.reshape(len(rows), 6))


def observed_contact_statistics(observed, targets, active, surfaces):
    """Read-only evaluation of the identical metric from selected raw records."""
    points, present = relation_points(observed, surfaces, targets)
    # Each intention's surface occupies its part's column in this catalog.
    part = torch.arange(6, device=targets.device)
    return point_statistics(points[:, part, part], targets, active, present[:, part, part])
