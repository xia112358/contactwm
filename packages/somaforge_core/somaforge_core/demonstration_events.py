"""Extract predictor observations from each demonstration's own Newton labels.

The supplied action clock contributes only frame intervals. Contact targets,
roles, and surface IDs are never inherited from its authored semantics.
No interval is shifted, subdivided, or discarded based on contact quality.
"""
import numpy as np

from .newton_contact_data import load_contact_labels
from .newton_contacts import PARTS
from .robot_assets import decode_robot_asset_json

SCHEMA = 'observed_demonstration_events_v1'


def observations(motion, labels):
    data = load_contact_labels(motion, labels)
    with np.load(motion, allow_pickle=False) as z:
        decode_robot_asset_json(z['robot_asset_json'], context=str(motion))
        data['fps'] = float(z['fps'].reshape(-1)[0])
        if len(z['joint_pos']) != len(data['contact_part_mask']):
            raise ValueError('Motion/contact evidence clock mismatch')
    return data


def materialize(data, intervals):
    """Use the demonstration clock, observing roles at whole-endpoint level.

    Keep means actual same-face contact throughout this interval, not loaded or
    stationary support. Establish means the final contact episode began inside
    it. Raw contact interruptions are reported, never turned into new actions.
    Foot shape/heel/toe changes do not break an endpoint contact episode.
    """
    mask = np.asarray(data['contact_part_mask'], bool)
    faces = np.asarray(data['contact_surface'])
    if mask.ndim != 2 or mask.shape[1] != len(PARTS) or faces.shape != mask.shape:
        raise ValueError('Invalid six-part contact timeline')
    fps = float(data['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid demonstration frame rate')
    clock = [(int(e['start_frame']), int(e['end_frame'])) for e in intervals]
    if (not clock or clock[0][0] != 0 or clock[-1][1] != len(mask)-1
            or any(not 0 <= a < b < len(mask) for a, b in clock)
            or any(left[1] != right[0] for left, right in zip(clock, clock[1:]))):
        raise ValueError('Action clock must cover the complete demonstration continuously')
    events = []
    for index, (a, b) in enumerate(clock):
        keeps, arrivals, releases = [], [], []
        for part in range(len(PARTS)):
            if mask[b, part]:
                on = mask[a:b+1, part] & (faces[a:b+1, part] == faces[b, part])
                if on.all():
                    keeps.append(part)
                else:
                    # Last actual off/different-face sample precedes the final
                    # episode. Its onset is observed, not a commanded touchdown.
                    onset = a + int(np.flatnonzero(~on)[-1]) + 1
                    arrivals.append(dict(frame=onset, part_index=part,
                        part=PARTS[part], surface=int(faces[b, part])))
            if mask[a, part] and (not mask[b, part] or faces[a, part] != faces[b, part]):
                releases.append(part)
        active = {r['part_index'] for r in arrivals}
        events.append(dict(schema=SCHEMA, segment_id=f'observed_action_{index:04d}',
            start_frame=a, end_frame=b, inclusive_end=True,
            duration_frames=b-a, duration_s=(b-a)/fps,
            terminal_event=('observed_contact_establishment' if arrivals else
                            'observed_contact_release' if releases else 'pose_adjustment'),
            touchdown_events=arrivals, touchdown_at_end=[r['part'] for r in arrivals],
            active_parts=[PARTS[p] for p in sorted(active)],
            persistent_parts=[PARTS[p] for p in keeps],
            released_parts=[PARTS[p] for p in releases],
            unconstrained_parts=[PARTS[p] for p in range(len(PARTS))
                                 if p not in active and p not in keeps],
            source_contact_parts=[PARTS[p] for p in np.flatnonzero(mask[a])],
            target_contact_parts=[PARTS[p] for p in np.flatnonzero(mask[b])],
            source_surfaces=np.where(mask[a], faces[a], -1).astype(int).tolist(),
            target_surfaces=np.where(mask[b], faces[b], -1).astype(int).tolist(),
            include_for_infiller=True,
            contact_truth='own_native_primary_face_active_allocated',
            role_semantics='observed_endpoint_contact_episode; not stationary or loaded support',
            observed_support_status='unknown_without_same_solve_loads'))
    return events, dict(schema=SCHEMA, event_count=len(events),
        action_clock_source='demonstration_intervals_only',
        retained_frame_range=[0, len(mask)-1], omitted_event_count=0,
        contact_labels_modified=False, old_contact_requirements_used=False,
        boundary_shifts=0, new_frame_toggle_events=0)
