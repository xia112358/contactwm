"""Newton observations and separately named geometry diagnostics.

Observed contacts require actual q and an authoritative Newton query worker.
Legacy geometry-only entry points fail instead of producing distance labels.
"""
import torch


def surface_gap(point, surface, center, rotation, half, ground, *, squared=False):
    local = torch.einsum('bij,bpj->bpi', rotation.transpose(-1, -2), point-center[:, None])
    outside = (local[..., :2].abs()-half[:, None, :2]).relu()
    box2 = (local[..., 2]-half[:, None, 2]).square()+outside.square().sum(-1)
    ground2 = (point[..., 2]-ground[:, None]).square()
    gap = torch.where(surface == 1, box2, ground2)
    if not squared:
        gap = gap.sqrt()
    return torch.where((surface == 0) | (surface == 1), gap, torch.full_like(gap, float('inf')))


def audit_geometric_anchors(data, scene, tolerance_m=.005):
    """Fail closed; do not silently project, relabel or discard demonstrations."""
    errors = []
    for prefix, point, mask, surface in [
        ('current', 'current_contacts', 'start_contacts', 'current_surfaces'),
        ('target', 'target_contacts', 'end_contact', 'target_surfaces')]:
        p = torch.as_tensor(data[point], device=scene[0].device)
        s = torch.as_tensor(data[surface], device=p.device)
        active = torch.as_tensor(data[mask], device=p.device) > .5
        gap = surface_gap(p, s, *scene)
        bad = active & (~torch.isfinite(gap) | (gap > tolerance_m))
        if bad.any():
            errors.append(f'{prefix}: {int(bad.sum())} invalid anchors, first row/part={bad.nonzero()[0].tolist()}, max={float(gap[active].max()*100):.4f} cm')
    if errors:
        raise ValueError('Geometric anchor diagnostic (not Newton contact validity): ' + '; '.join(errors))


def geometric_contact_proximity(point, anchor, active, persistent, surface, center, rotation, half, ground, tolerance_m=.005):
    """Diagnostic proximity mask only; NOT observed contact or training labels."""
    point_valid = surface_gap(point, surface, center, rotation, half, ground) <= tolerance_m
    anchor_valid = surface_gap(anchor, surface, center, rotation, half, ground) <= tolerance_m
    retained = anchor_valid & ((point-anchor).norm(dim=-1) <= tolerance_m)
    return active.bool() & point_valid & (~persistent.bool() | retained)


def _require_newton_observation():
    raise RuntimeError(
        'Newton/MJWarp solver contact observation required. Legacy geometry-only '
        'contact validation is disabled: distances, predicted intentions and forces '
        'are not activation labels. Record solver_contact_* and use '
        'scripts/extract_solver_contact_labels.py; online callers must first '
        'integrate solver observations with matching environment/body/time mapping.'
    )


def query_q_contacts(qpos, scene):
    """Non-differentiable supervision/observation; geometry losses stay separate."""
    from somaforge_core.newton_contact_query import query_contacts
    values = query_contacts(qpos.detach().cpu().numpy(),
        [x.detach().cpu().numpy() for x in scene])
    from somaforge_core.contact_labels import process_independent_contacts
    from somaforge_core.contact_face_selection import select_contact_pairs
    selected = select_contact_pairs(values['pairs'], values['surface_catalog'])
    task = process_independent_contacts(selected['contact_part_mask'], selected['contact_surface'], selected['contact_position_w'])
    return {key: torch.as_tensor(task[field], device=qpos.device)
            for key, field in [('active','contact_part_mask'),('position_w','contact_position_w'),
                               ('surface','contact_surface')]}


def query_q_contact_sequence(qpos, scene, *, fps, stream=None, shared_boundary=False):
    """Chronological q, not independent candidates; retain stream across segments.

    With an overlapping boundary, return the previous label without counting
    that observation twice. Raw evidence stays available under raw_* keys.
    """
    import numpy as np
    from somaforge_core.newton_contact_query import query_contacts
    from somaforge_core.contact_labels import ContactLabelStream, label_contract
    if stream is None:
        if shared_boundary:
            raise ValueError('Shared boundary requires the previous contact stream')
        stream = ContactLabelStream(fps=fps)
    if stream.contract != label_contract(fps):
        raise ValueError('Sequence sample rate/policy differs from its contact stream')
    values = query_contacts(qpos.detach().cpu().numpy(), [x.detach().cpu().numpy() for x in scene])
    from somaforge_core.contact_face_selection import select_contact_pairs
    selected = select_contact_pairs(values['pairs'], values['surface_catalog'])
    rows = []
    if shared_boundary:
        if stream.surface is None:
            raise ValueError('Contact stream has no previous boundary')
        if not hasattr(stream, '_last_q') or not np.array_equal(stream._last_q, qpos[0].detach().cpu().numpy()):
            raise ValueError('Shared contact boundary q does not match previous endpoint')
        rows.append(dict(contact_part_mask=stream.surface >= 0, contact_surface=stream.surface.copy(),
                         contact_position_w=stream.position.copy()))
    for i in range(int(shared_boundary),len(qpos)):
        rows.append(stream.step(selected['contact_part_mask'][i],selected['contact_surface'][i],selected['contact_position_w'][i]))
    stream._last_q = qpos[-1].detach().cpu().numpy().copy()
    result = {}
    for key,field in [('active','contact_part_mask'),('surface','contact_surface'),('position_w','contact_position_w')]:
        result[key] = torch.as_tensor(np.stack([r[field] for r in rows]),device=qpos.device)
        result['raw_'+key] = torch.as_tensor(values[key],device=qpos.device)
    return result, stream


def validate_demonstration_anchors(data, scene, tolerance_m=.005, *, qpos=None):
    """Verify labels at their actual q, never reject margin gaps as no contact."""
    if qpos is None:
        _require_newton_observation()
    for frame, mask, surface in [(0, 'start_contacts', 'current_surfaces'),
                                  (-1, 'end_contact', 'target_surfaces')]:
        q = torch.as_tensor(qpos[:, frame], device=scene[0].device)
        observed = query_q_contacts(q, scene)
        expected = torch.as_tensor(data[mask], device=q.device).bool()
        target_surface = torch.as_tensor(data[surface], device=q.device)
        if (observed['active'] != expected).any() or (
                expected & (observed['surface'] != target_surface)).any():
            raise ValueError('Demonstration contact/surface disagrees with Newton at its supplied q; '
                             'relabel and resegment the source, not just the endpoints')


def valid_contact(point, anchor, active, persistent, surface, center, rotation, half, ground, tolerance_m=.005):
    """Legacy DAgger gate: do not admit geometry-only contact labels."""
    _require_newton_observation()


def observed_contacts(prediction, previous_contact, previous_anchor, center, rotation, half, ground,
                      *, qpos=None, previous_surface=None):
    """Realized contacts include unplanned touches and exclude failed intentions."""
    qpos = getattr(prediction, 'qpos', None) if qpos is None else qpos
    if qpos is None:
        _require_newton_observation()
    observed = query_q_contacts(qpos, (center, rotation, half, ground))
    active, anchor, surface = (observed[k] for k in ('active', 'position_w', 'surface'))
    # Persistent anchors are memory, not proof of contact or a distance gate.
    if previous_surface is not None:
        retained = active & previous_contact.bool() & (previous_surface == surface)
        anchor = torch.where(retained[..., None], previous_anchor, anchor)
    return active, anchor, surface


def validate_touchdown_cache(cached, samples, phase_frames):
    """Equal lengths are insufficient: row/frame identity must match exactly."""
    import numpy as np
    for key, attr in [('motion_ids', 'motion_id'), ('current_frames', 'current_frame'), ('target_frames', 'target_frame')]:
        expected = np.asarray([getattr(sample, attr) for sample in samples], dtype=np.int64)
        if key not in cached or not np.array_equal(cached[key], expected):
            raise ValueError(f'touchdown q cache {key} does not match event segmentation')
    q = cached['target_q']
    if q.shape != (len(samples), phase_frames, 36) or not np.isfinite(q).all():
        raise ValueError('touchdown q cache has invalid shape or nonfinite poses')
