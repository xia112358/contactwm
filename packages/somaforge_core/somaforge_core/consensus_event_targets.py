"""Optimization intent compiled from the same observed events as the audit.

These masks are movable fitting guides, never contact labels. In particular,
an episode vote must not disappear because its executions occur at different
frames, and noncontact at one population timestamp is not a shared release.
"""
import numpy as np


def event_targets(consensus, reports, masks, surfaces, depths=None):
    masks, surfaces = np.asarray(masks, bool), np.asarray(surfaces)
    if masks.ndim != 3 or masks.shape[-1] != 6 or surfaces.shape != masks.shape:
        raise ValueError('Expected native execution contact arrays')
    if len(reports) != len(masks):
        raise ValueError('Event reports must match native executions')
    depth = None if depths is None else np.asarray(depths)
    if depth is not None and depth.shape != masks.shape:
        raise ValueError('Native depths must match execution contacts')
    n = masks.shape[1]
    records = [{r['action']: r['episode'] for r in report['checks'] if r['passed']}
               for report in reports]
    wanted = np.zeros((n, 6), bool)
    faces = np.full((n, 6), -1, dtype=surfaces.dtype)
    release = np.zeros((n, 6), bool)
    target_depth = np.full((n, 6), np.nan)
    penetration = np.zeros((n, 6))
    events, releases = [], []
    for event in consensus['events']:
        part, face, key = event['part'], event['surface'], event['id']
        observed = [(i, r[key]) for i, r in enumerate(records) if key in r]
        if not observed:
            raise ValueError(f'No transported native observations for event {key}')
        # Use observed episode extents, not a new fixed frame deadline. A later
        # native refresh can transport this guide with the output's own events.
        a, b = np.median([e['interval'] for _, e in observed], axis=0)
        a, b = max(0, int(np.floor(a))), min(n, int(np.ceil(b)))
        if b <= a:
            raise ValueError('Empty observed event interval')
        if np.any(wanted[a:b, part] & (faces[a:b, part] != face)):
            raise ValueError('Incompatible surface guides in the same event phase')
        wanted[a:b, part] = True
        faces[a:b, part] = face
        values = []
        for i, episode in observed:
            start, end = episode['interval']
            valid = masks[i, start:end, part] & (surfaces[i, start:end, part] == face)
            if depth is not None:
                values.append(depth[i, start:end, part][valid])
        if depth is not None:
            # Equal execution contribution: first reduce each execution, then
            # the population. Long episodes do not receive extra votes.
            values = [v[np.isfinite(v)] for v in values]
            if any(not len(v) for v in values):
                raise ValueError('Missing measured native event depth')
            target_depth[a:b, part] = np.median([np.median(v) for v in values])
            penetration[a:b, part] = min(0., min(float(v.min()) for v in values))
        events.append(dict(id=key, part=part, surface=face, interval=[a, b],
                           source_count=len(observed)))
    by_id = {e['id']: e for e in events}
    for part in range(6):
        sequence = [e for e in consensus['events'] if e['part'] == part]
        for previous, current in zip(sequence, sequence[1:]):
            # Only the audit's shared release is binding. Disputed/minority
            # gaps must not fight a valid ongoing contact during optimization.
            if current.get('release_votes', 0) != consensus['source_count']:
                continue
            left, right = by_id[previous['id']]['interval'][1], by_id[current['id']]['interval'][0]
            if right > left:
                release[left:right, part] = True
                releases.append(dict(part=part, before=previous['id'], after=current['id'],
                                     interval=[left, right]))
    if np.any(wanted & release):
        raise ValueError('Event acquisition and shared release guides conflict')
    return dict(wanted=wanted, faces=faces, release=release, target_depth=target_depth,
                allowed_penetration=penetration, events=events, releases=releases,
                schema='observed_population_event_targets_v1', contact_truth=False)
