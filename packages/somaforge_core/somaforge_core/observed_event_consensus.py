"""Population event consensus from native contacts, independent of authored refs.

Evidence filtering never changes raw Newton labels. Votes describe observed
execution structure, not task success or dynamic feasibility.
"""
import numpy as np
from .contact_order_acceptance import _runs


def episodes(mask, surface, *, stable_frames=3, max_dropout_frames=3):
    mask, surface = np.asarray(mask, bool), np.asarray(surface)
    if mask.ndim != 2 or mask.shape[1] != 6 or surface.shape != mask.shape:
        raise ValueError('Expected six-part native contact evidence')
    if np.any(mask & (surface < 0)):
        raise ValueError('Missing actual primary surface')
    result = []
    for part in range(6):
        sequence = []
        for face in np.unique(surface[mask[:, part], part]):
            raw = mask[:, part] & (surface[:, part] == face)
            continuous = raw.copy()
            for a, b in _runs(~raw):
                if a and b < len(raw) and b-a <= max_dropout_frames and not mask[a:b, part].any():
                    continuous[a:b] = True
            for a, b in _runs(continuous):
                evidence = [(int(a+c), int(a+d)) for c, d in _runs(raw[a:b]) if d-c >= stable_frames]
                if evidence:
                    sequence.append(dict(part=part, surface=int(face), kind='establish',
                        interval=[int(a), int(b)], stable_evidence=[evidence[0][0], evidence[0][0]+stable_frames]))
        result.append(sorted(sequence, key=lambda e: e['interval']))
    return result


def correspondence_score(left, right, typical):
    a, b = left['interval'], right['interval']
    span = max(a[1]-a[0], b[1]-b[0], typical, 1.)
    distance = abs(np.mean(a)-np.mean(b))/span
    # Duration-scaled robust evidence, not a fixed frame window/deadline.
    # Old cardinality-first matching paired climb contacts with descent contacts.
    return 2/(1+distance*distance)-.5


def match(left, right):
    """Ordered soft correspondence; skipping beats pairing unrelated occurrences."""
    n, m = len(left), len(right)
    score = np.full((n+1, m+1), -np.inf); score[0, 0] = 0
    back = {}
    typical = float(np.median([e['interval'][1]-e['interval'][0] for e in left+right] or [1]))
    for i in range(n+1):
        for j in range(m+1):
            for di, dj in ((1, 0), (0, 1), (1, 1)):
                if i+di > n or j+dj > m: continue
                gain = 0.
                if di and dj:
                    if left[i]['surface'] != right[j]['surface']: continue
                    gain = correspondence_score(left[i], right[j], typical)
                value = score[i, j]+gain
                if value > score[i+di, j+dj]:
                    score[i+di, j+dj] = value; back[i+di, j+dj] = (i, j)
    pairs = []; i, j = n, m
    while i or j:
        a, b = back[i, j]
        pairs.append((a if i>a else None, b if j>b else None)); i, j = a, b
    return pairs[::-1]


def build_consensus(masks, surfaces, *, stable_frames=3, max_dropout_frames=3):
    """Progressive sequence profiles with equal execution votes, no ref input.

    Only strict-majority episodes and endpoint relations become requirements.
    Minority episodes remain in the audit. Disputed relations remain explicit.
    """
    sequences = [episodes(m, s, stable_frames=stable_frames, max_dropout_frames=max_dropout_frames)
                 for m, s in zip(masks, surfaces)]
    if len(sequences) < 3: raise ValueError('Need three complete executions')
    count = len(sequences); events = []; discarded = []
    # Content order makes consensus invariant to caller/environment ordering.
    order = sorted(range(count), key=lambda i: repr(sequences[i]))
    for part in range(6):
        columns = []
        for index in order:
            sequence = sequences[index][part]
            profile = [dict(surface=next(iter(c.values()))['surface'],
                interval=np.median([e['interval'] for e in c.values()], axis=0).tolist()) for c in columns]
            merged = []
            for a, b in match(profile, sequence):
                column = {} if a is None else dict(columns[a])
                if b is not None: column[index] = sequence[b]
                merged.append(column)
            columns = merged
        for column in columns:
            times = np.median([e['interval'] for e in column.values()], axis=0).tolist()
            row = dict(id=len(events), part=part, surface=next(iter(column.values()))['surface'],
                interval=times, votes=len(column), voters=sorted(column),
                observations={str(i): e for i, e in column.items()})
            if len(column)*2 > count: events.append(row)
            else: discarded.append(row)
    for part in range(6):
        sequence = [e for e in events if e['part'] == part]
        for previous, current in zip(sequence, sequence[1:]):
            releases = []
            for k in set(previous['voters']) & set(current['voters']):
                a = previous['observations'][str(k)]['interval'][1]
                b = current['observations'][str(k)]['interval'][0]
                if b > a and any(y-x >= stable_frames for x, y in _runs(~np.asarray(masks[k][a:b, part], bool))):
                    releases.append(k)
            current['distinct_from_previous'] = (previous['surface'] != current['surface'] or len(releases) == count)
            current['release_votes'] = len(releases)
    relations, disputed = [], []
    # Endpoint votes retain cross-limb order AND interval overlap. No original
    # action IDs, support owners, or authored deadlines participate.
    for i, a in enumerate(events):
        for b in events[i+1:]:
            shared = sorted(set(a['voters']) & set(b['voters']))
            if len(shared)*2 <= count: continue
            for u in (0, 1):
                for v in (0, 1):
                    differences = [a['observations'][str(k)]['interval'][u]-b['observations'][str(k)]['interval'][v] for k in shared]
                    negative = sum(d < 0 for d in differences); positive = sum(d > 0 for d in differences)
                    row = dict(left=[a['id'], u], right=[b['id'], v],
                               before_votes=negative, after_votes=positive, total=len(shared))
                    # Strict population majority; consensus median clock must
                    # realize the relation, avoiding voting cycles.
                    if negative*2 > count and a['interval'][u] < b['interval'][v]:
                        relations.append(dict(row, information_required=(a['part'] == b['part'] or
                            (u == 0 and v == 1 and negative == count)),
                            role='observed_relation_not_task_rule'))
                    elif positive*2 > count and b['interval'][v] < a['interval'][u]:
                        relations.append(dict(row, left=row['right'], right=row['left'],
                            information_required=(a['part'] == b['part'] or
                                (v == 0 and u == 1 and positive == count)),
                            role='observed_relation_not_task_rule'))
                    else: disputed.append(row)
    return dict(schema='native_population_information_consensus_v2', events=events, relations=relations,
        disputed_relations=disputed, minority_episodes=discarded, source_count=count,
        evidence_policy=dict(stable_frames=stable_frames, max_dropout_frames=max_dropout_frames),
        clock='population median observed episode endpoints', original_reference_used=False,
        task_success_verified=False)


def audit_consensus(consensus, mask, surface):
    sequences = episodes(mask, surface, **consensus['evidence_policy'])
    assigned = {}
    for part in range(6):
        required = [e for e in consensus['events'] if e['part'] == part]
        observed = sequences[part]
        typical = float(np.median([e['interval'][1]-e['interval'][0] for e in required+observed] or [1]))
        # Monotone assignment can reuse an ongoing contact when there is no
        # shared release evidence. A smoother contact is not a missing event.
        states = {-1: (0., {})}
        for event in required:
            updated = {}
            for last, (score, history) in states.items():
                options = [(last, score, history)]
                for j, episode in enumerate(observed):
                    if j < last or (j == last and event.get('distinct_from_previous', True)): continue
                    if episode['surface'] != event['surface']: continue
                    gain = correspondence_score(event, episode, typical)
                    if gain <= 0: continue
                    options.append((j, score+gain, dict(history, **{str(event['id']): episode})))
                for j, value, history in options:
                    if j not in updated or value > updated[j][0]: updated[j] = (value, history)
            states = updated
        if states:
            _, history = max(states.values(), key=lambda state: state[0])
            assigned.update({int(k): v for k, v in history.items()})
    checks = []
    for e in consensus['events']:
        observed = assigned.get(e['id'])
        checks.append(dict(action=e['id'], kind='establish', part=e['part'], surface=e['surface'],
                           passed=observed is not None, episode=observed))
    relation_checks = []
    for r in consensus['relations']:
        a, u = r['left']; b, v = r['right']
        available = a in assigned and b in assigned
        same_episode = available and assigned[a] is assigned[b]
        passed = available and (assigned[a]['interval'][u] < assigned[b]['interval'][v] or same_episode)
        relation_checks.append(dict(r, passed=bool(passed) if available else None,
            evaluable=available, status='observed' if available else 'missing_event_already_counted',
            information_required=r.get('information_required', False)))
    missing = sum(not r['passed'] for r in checks)
    relation_loss = sum(r['evaluable'] and r['information_required'] and not r['passed'] for r in relation_checks)
    failed = missing+relation_loss
    return dict(schema=consensus['schema'], passed=bool(checks) and failed == 0,
        failed_checks=failed+(not checks), missing_event_count=missing, relation_information_loss_count=relation_loss,
        relation_variation_count=sum(r['evaluable'] and not r['passed'] for r in relation_checks), check_count=len(checks)+len(relation_checks), checks=checks,
        relation_checks=relation_checks, evidence_policy=consensus['evidence_policy'],
        consensus_clock={str(e['id']): e['interval'] for e in consensus['events']},
        task_success_verified=False, scope='retention of stable observed information; source executions are observations, not errors')


def strategy_groups(masks, surfaces):
    """Separate different per-part surface itineraries before event voting.

    Repeated acquisitions of the same surface stay for population episode
    matching, rather than fragmenting groups on incidental contact dropout.
    """
    groups = {}
    for index, (mask, surface) in enumerate(zip(masks, surfaces)):
        signature = []
        for sequence in episodes(mask, surface):
            faces = []
            for event in sequence:
                if not faces or faces[-1] != event['surface']: faces.append(event['surface'])
            signature.append(tuple(faces))
        groups.setdefault(tuple(signature), []).append(index)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def observed_consensus_subset(consensus, source_report):
    """Transport audit for this source's observed population events only.

    Minority incidental episodes are not promoted to hard constraints merely
    because one source contains them. Missing population events are not invented.
    """
    observed = {row['action']: row['episode'] for row in source_report['checks'] if row['passed']}
    events = [dict(e, interval=observed[e['id']]['interval']) for e in consensus['events'] if e['id'] in observed]
    relations = []
    for row in consensus['relations']:
        a, u = row['left']; b, v = row['right']
        if a in observed and b in observed and observed[a]['interval'][u] < observed[b]['interval'][v]:
            relations.append(row)
    return dict(consensus, events=events, relations=relations, scope='observed population events during transport')
