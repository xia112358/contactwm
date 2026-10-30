"""Preserved experiment helpers for regression tests; no training CLI."""
from dataclasses import dataclass
import numpy as np
from somaforge_core.contact_events import bridge_dropouts, group_endpoint, MAX_DROPOUT_FRAMES

@dataclass(frozen=True)
class Config:
    release_seconds: float = 0.06
    landing_window_seconds: float = 0.12
    landing_occupancy: float = 2 / 3
    cluster_seconds: float = 0.12
    minimum_normal_range_includemargin_ratio: float = 0.5

def events_for_part(contact, gap, penetration, position, fps, cfg, surface_sequence, surface_normal, includemargin):
    actual_contact = np.asarray(contact, dtype=bool)
    surface_normal = np.asarray(surface_normal, dtype=float)
    includemargin = np.asarray(includemargin, dtype=float)
    if surface_normal.shape != (3,) or includemargin.shape != actual_contact.shape:
        raise ValueError('Invalid event geometry evidence')
    contact = bridge_dropouts(actual_contact, surface_sequence)
    release_n = max(1, round(fps * cfg.release_seconds))
    window = max(1, round(fps * cfg.landing_window_seconds))
    needed = int(np.ceil(window * cfg.landing_occupancy))
    events = []
    rejected = []
    away_count = 0
    armed = False
    origin = 0
    last_contact = 0
    for f in range(len(contact)):
        clearly_away = not contact[f]
        away_count = away_count + 1 if clearly_away else 0
        if not armed and away_count >= release_n:
            armed = True
            origin = last_contact
            release = f - release_n + 1
        if armed and actual_contact[f] and (f + window <= len(contact)) and (contact[f:f + window].sum() >= needed):
            path = position[origin:f + 1]
            excursion = float(np.linalg.norm(path - path[0], axis=-1).max())
            chord = path[-1] - path[0]
            u = np.linspace(0, 1, len(path))[:, None]
            arc = float(np.linalg.norm(path - (path[0] + u * chord), axis=-1).max())
            normal_coordinate = path @ surface_normal
            normal_range = float(np.ptp(normal_coordinate))
            endpoint_margin = float(np.nanmin(includemargin[[origin, f]]))
            if not np.isfinite(endpoint_margin) or endpoint_margin <= 0:
                raise ValueError('Missing actual Newton includemargin at event endpoint')
            required = cfg.minimum_normal_range_includemargin_ratio * endpoint_margin
            record = dict(frame=f, release_frame=release, previous_contact_frame=origin, excursion_m=excursion, arc_m=arc, normal_range_m=normal_range, endpoint_includemargin_m=endpoint_margin, required_normal_range_m=required, normal_range_includemargin_ratio=normal_range / endpoint_margin, landing_occupancy=float(contact[f:f + window].mean()))
            record['motion_qualified'] = bool(normal_range >= required)
            events.append(record)
            if not record['motion_qualified']:
                rejected.append(dict(**record, reason='insufficient_surface_normal_motion'))
            armed = False
            away_count = 0
        if contact[f] and (not armed):
            last_contact = f
    return (events, rejected)

def cluster_events(events, contact, fps, cfg, surfaces=None):
    continuity = np.stack([bridge_dropouts(contact[:, p], None if surfaces is None else surfaces[:, p]) for p in range(contact.shape[1])], axis=1)
    groups = []
    for event in sorted(events, key=lambda e: (e['frame'], e['part_index'])):
        event = dict(event)
        if groups:
            previous = groups[-1]
            parts = [e['part_index'] for e in previous]
            merged = False
            if event['frame'] - previous[0]['frame'] <= round(cfg.cluster_seconds * fps) and event['part_index'] not in parts:
                candidate = previous + [event]
                first = max(event['frame'], group_endpoint(previous))
                for end in range(first, min(len(contact), event['frame'] + MAX_DROPOUT_FRAMES + 1)):
                    realized = all((contact[end, e['part_index']] and (surfaces is None or surfaces[end, e['part_index']] == e['surface']) for e in candidate))
                    continuous = all((continuity[e['frame']:end + 1, e['part_index']].all() and (surfaces is None or np.all(surfaces[e['frame']:end + 1, e['part_index']][contact[e['frame']:end + 1, e['part_index']]] == e['surface'])) for e in candidate))
                    if realized and continuous:
                        for e in candidate:
                            e['cluster_endpoint_frame'] = end
                        previous.append(event)
                        merged = True
                        break
            if merged:
                continue
        if event.get('motion_qualified', True):
            groups.append([event])
    return groups
