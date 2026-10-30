"""Event continuity only: never substitute these masks for Newton contact truth."""
import numpy as np

MAX_DROPOUT_FRAMES = 3


def event_contract(fps):
    return dict(schema='same_surface_event_dropout_v1', max_dropout_frames=MAX_DROPOUT_FRAMES,
                fps=float(fps), timing='offline_bounded_lookahead',
                endpoint='actual_contact_required', persistent='actual_uninterrupted_contact')


def bridge_dropouts(contact, surfaces=None):
    """Bridge bounded inactive runs only when both ends contact the same surface.

    No extrapolation at sequence edges and no bridging across other surfaces.
    Return a copy; physical labels are never modified.
    """
    raw = np.asarray(contact, dtype=bool)
    if raw.ndim != 1:
        raise ValueError('Expected one part contact sequence')
    if surfaces is not None and np.shape(surfaces) != raw.shape:
        raise ValueError('Surface shape mismatch')
    result = raw.copy()
    observed = np.flatnonzero(raw)
    for a, b in zip(observed[:-1], observed[1:]):
        if 1 <= b-a-1 <= MAX_DROPOUT_FRAMES:
            if surfaces is None or (surfaces[a] == surfaces[b] and np.all(np.asarray(surfaces)[a+1:b] == -1)):
                result[a+1:b] = True
    return result


def group_endpoint(group):
    return max(e.get('cluster_endpoint_frame', e['frame']) for e in group)


def load_segment_rows(entry, labels, fps):
    """Read certified event boundaries, never silently re-segment a new corpus."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(entry['event_segments_file'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['event_segments_sha256']:
        raise ValueError('Event segmentation content changed')
    expected = event_contract(fps)
    mask, surfaces = labels['contact_part_mask'], labels['contact_surface']
    from .newton_contacts import PARTS
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get('event_contract') != expected:
            raise ValueError('Event policy mismatch')
        if 'event_selection' in entry and row.get('event_selection') != entry['event_selection']:
            raise ValueError('Event selection policy mismatch')
        a, b = row['start_frame'], row['end_frame']
        if not 0 <= a < b < len(mask):
            raise ValueError('Invalid segment boundary')
        for event in row['touchdown_events']:
            p = event['part_index']
            if not mask[b, p] or surfaces[b, p] != event['surface']:
                raise ValueError('Unrealized endpoint contact')
        active = {PARTS[e['part_index']] for e in row['touchdown_events']}
        persistent = {PARTS[p] for p in range(len(PARTS)) if mask[a:b+1,p].all()
                      and (surfaces[a:b+1,p] == surfaces[a,p]).all() and PARTS[p] not in active}
        if active != set(row['active_parts']) or persistent != set(row['persistent_parts']):
            raise ValueError('Segment contact roles disagree with actual Newton labels')
    return rows
