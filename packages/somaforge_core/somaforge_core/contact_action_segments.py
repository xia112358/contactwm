"""Select action boundaries from native contacts without editing contact truth.

Short same-face interruptions belong to the same action for boundary selection
only. Endpoint roles and targets always come from the unchanged observations.
"""
import numpy as np

from .contact_events import event_contract
from .demonstration_events import materialize
from .newton_contacts import PARTS


def action_segments(data, *, stable_frames=3, same_face_gap=6,
                    cluster_window=6, minimum_segment_frames=20):
    for name, value, minimum in (
        ('stable_frames', stable_frames, 1), ('same_face_gap', same_face_gap, 0),
        ('cluster_window', cluster_window, 0),
        ('minimum_segment_frames', minimum_segment_frames, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f'Invalid {name}')
    mask = np.asarray(data['contact_part_mask'], dtype=bool)
    surfaces = np.asarray(data['contact_surface'])
    if mask.ndim != 2 or mask.shape[1] != len(PARTS) or surfaces.shape != mask.shape:
        raise ValueError('Expected six-part native contact timelines')
    n = len(mask)
    if n < 2 or np.any(surfaces[mask] < 0):
        raise ValueError('Invalid native contact timeline')
    candidates, suppressed = [], []
    for part, name in enumerate(PARTS):
        previous_frame, previous_surface = None, None
        for frame in np.flatnonzero(mask[:, part]):
            frame, surface = int(frame), int(surfaces[frame, part])
            onset = (previous_frame is None or frame != previous_frame+1
                     or surface != previous_surface)
            stable = (frame+stable_frames <= n
                and mask[frame:frame+stable_frames, part].all()
                and (surfaces[frame:frame+stable_frames, part] == surface).all())
            if frame > 0 and onset and stable:
                changed = previous_surface is not None and surface != previous_surface
                event = dict(frame=frame, part_index=part, part=name,
                    surface=surface, surface_change=changed,
                    priority=2 if changed else 1 if name.endswith(('hand', 'knee')) else 0)
                gap = None if previous_frame is None else frame-previous_frame-1
                if previous_surface == surface and gap <= same_face_gap:
                    suppressed.append(dict(event, reason='short_same_face_recontact', gap_frames=gap))
                else:
                    candidates.append(event)
            previous_frame, previous_surface = frame, surface
    candidates.sort(key=lambda event: (event['frame'], event['part_index']))

    groups = []
    for event in candidates:
        frame = event['frame']
        previous = groups[-1] if groups else None
        coherent = (previous is not None and frame-previous['first_frame'] <= cluster_window
            and all(mask[e['frame']:frame+stable_frames, e['part_index']].all()
                and (surfaces[e['frame']:frame+stable_frames, e['part_index']] == e['surface']).all()
                for e in previous['events']))
        if coherent:
            previous['frame'] = frame
            previous['events'].append(event)
            previous['priority'] = max(previous['priority'], event['priority'])
        else:
            groups.append(dict(first_frame=frame, frame=frame,
                               priority=event['priority'], events=[event]))

    # Preserve higher-priority surface transfers and hand/knee actions first.
    # The endpoints are mandatory; no artificial equal-duration cuts are added.
    retained, rejected = [], []
    cuts = [0, n-1]
    for group in sorted(groups, key=lambda g: (-g['priority'], g['frame'])):
        conflicts = sorted(c for c in cuts if abs(c-group['frame']) < minimum_segment_frames)
        if conflicts:
            rejected.append(dict(group, reason='minimum_duration_conflict', conflicting_cuts=conflicts))
        else:
            cuts.append(group['frame'])
            retained.append(group)
    cuts.sort()
    events, audit = materialize(data, [dict(start_frame=a, end_frame=b)
                                      for a, b in zip(cuts, cuts[1:])])
    selection = dict(schema='native_limb_action_boundaries_v1', stable_frames=stable_frames,
        same_face_gap=same_face_gap, cluster_window=cluster_window,
        minimum_segment_frames=minimum_segment_frames,
        priority_order=['surface_change', 'hand_or_knee_establishment', 'foot_establishment'],
        tie_break='earliest frame', clustering='bounded total span; uninterrupted common same-face contact',
        original_reference_boundaries_used=False,
        endpoint_convention='inclusive; adjacent segments share one endpoint sample')
    by_frame = {group['frame']: group for group in retained}
    for event in events:
        event.update(event_contract=event_contract(data['fps']), event_selection=selection,
                     boundary_evidence=by_frame.get(event['end_frame']))
    audit.update(selection=selection, complete_timeline=True,
        action_clock_source='own native limb action groups; complete prefix and tail',
        candidate_events=candidates, suppressed_recontacts=suppressed,
        retained_boundaries=sorted(retained, key=lambda g: g['frame']),
        rejected_boundaries=sorted(rejected, key=lambda g: g['frame']),
        short_segments=[dict(start_frame=a, end_frame=b, reason='whole_motion_shorter_than_minimum')
                        for a, b in zip(cuts, cuts[1:]) if b-a < minimum_segment_frames],
        contact_labels_modified=False, poses_modified=False)
    return events, audit
