"""Contact locations tied to the observed scene; Newton owns contact truth."""
import torch
import torch.nn.functional as F

from somaforge_core.heightmap import HEIGHTMAP_COLS, HEIGHTMAP_ROWS, HEIGHTMAP_RESOLUTION_M
from contact_solver.endpoint_task import ENDPOINT_POSITION_TOLERANCE_M


# Spatial task tolerance is independent of heightmap/depth pixel resolution.


def representative_pairs(pairs, active, surfaces):
    """One witness per intention, chosen by solver separation, never target XY.

    Callers supply eligible primary faces. Candidate pairs are suitable for
    optimization only; evaluation must supply actual active+allocated pairs.
    """
    result = []
    for part, enabled in enumerate(active):
        matches = [p for p in pairs if enabled and int(p['part']) == part
                   and int(p['surface']) == int(surfaces[part])]
        result.append(min(matches, key=lambda p: float(p['dist'])) if matches else None)
    return result


def relative_layout_statistics(actual, intended, observed):
    """Return squared XY residual, common shift and witness count per sample.

    Zero residual with fewer than two witnesses is not successful execution.
    Consumers must separately check completeness and Newton contact/safety.
    """
    count = observed.sum(-1)
    delta = (actual[..., :2] - intended[..., :2]) * observed[..., None]
    shift = delta.sum(-2) / count.clamp_min(1)[..., None]
    centered = (delta - shift[..., None, :]) * observed[..., None]
    error = centered.square().sum((-1, -2)) / count.clamp_min(1)
    return error, shift, count


def endpoint_position_statistics(actual, intended, observed):
    """XY endpoint error without removing a future-only common displacement.

    Both tensors must use the same frame. A joint change of coordinates leaves
    the error unchanged. Missing witnesses are counted, never certified here.
    """
    count = observed.sum(-1)
    delta = torch.where(observed[..., None], actual[..., :2]-intended[..., :2], 0.)
    return delta.square().sum((-1, -2))/count.clamp_min(1), count


def endpoint_position_loss(error_squared, tolerance_m=ENDPOINT_POSITION_TOLERANCE_M):
    """Squared excess outside the accepted endpoint RMS range.

    Average squared part-position errors before applying the existing task
    tolerance; independently bounding each limb would change task semantics.
    Zero spatial loss does not certify completeness, activation or safety.
    Clamping before sqrt keeps the zero-error gradient finite and makes both
    loss and gradient zero throughout the accepted range, including its edge.
    """
    import math
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Invalid endpoint spatial tolerance')
    return ((error_squared / tolerance_m**2).clamp_min(1).sqrt()-1).square()


def support_bound_targets(future_points, current_points, current_contact, roles):
    """Legacy API: retain real endpoints, including persistent contact parts.

    Persistent activation does not certify a fixed material point. Current
    representative positions must never overwrite demonstrated endpoints.
    """
    return observed_endpoint_targets(future_points, current_contact, roles)


def observed_endpoint_targets(future_points, current_contact, roles):
    """Keep real endpoint labels; current contact validates roles only."""
    persistent = roles == 2
    if bool((persistent & ~current_contact).any()):
        raise ValueError('Persistent support requires an actual current contact')
    return future_points.detach(), persistent


def persistent_role_consistent(roles, current_contact):
    """A persistent intention requires current actual contact, not a fixed point."""
    return ~((roles == 2) & ~current_contact).any(-1)


def nearest_spatial_pairs(pairs, active, surfaces, points, *, regions=None):
    """Match a plan to eligible geometry, never define contact activation.

    Caller supplies candidates for loss, actual Newton pairs for evaluation.
    If regions are required, each pair must carry canonical ``contact_region``.
    Contact truth and region coverage must still be checked independently.
    """
    result = []
    for part, enabled in enumerate(active):
        matches = [p for p in pairs if enabled and int(p['part']) == part
                   and int(p['surface']) == int(surfaces[part])
                   and (regions is None or bool(regions[part][int(p['contact_region'])]))]
        result.append(min(matches, key=lambda p: sum(
            (float(p['position_w'][axis])-float(points[part][axis]))**2
            for axis in (0, 1))) if matches else None)
    return result


def relative_plan_objective(prediction, target, heightmap):
    """Compatibility entry for scene-fixed real endpoint supervision.

    Labels come from actual future contacts for all roles, including persistent
    parts. Root-yaw raster coordinates are only a
    representation, not a constraint on future root/body position.
    Future-only translations are no longer marginalized out.
    """
    if (getattr(prediction, 'candidate_points_local', None) is None
            and prediction.location_logits.shape[-1] != HEIGHTMAP_ROWS * HEIGHTMAP_COLS):
        raise ValueError('plan objective requires the observed raster')
    if not bool(torch.isfinite(prediction.role_logits.gather(-1, target['role'][..., None])).all()):
        raise ValueError('Supervised role is outside the actual-contact action space')
    from generator.limb_contact import role_objective
    role = role_objective(prediction, target['role'])
    valid = target['contact_cell_valid'] & (target['role'] != 0)
    locations = (prediction.location_logits_for_roles(target['role'])
        if hasattr(prediction, 'location_logits_for_roles') else prediction.location_logits)
    safe_cells = torch.where(valid, target['contact_cell'], locations.argmax(-1))
    if bool((valid & ~torch.isfinite(locations.gather(-1, safe_cells[..., None]).squeeze(-1))).any()):
        raise ValueError('Supervised keep location is outside the observed contact relation')
    ce = F.cross_entropy(locations.transpose(1, 2), safe_cells, reduction='none')
    location = (ce * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return role + location, {'plan_role_loss': role, 'plan_location_loss': location,
        'plan_contact_exact': (prediction.contact == (target['role'] != 0)).all(-1).float(),
        'plan_role_exact': (prediction.role == target['role']).all(-1).float()}
