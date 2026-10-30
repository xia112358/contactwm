"""Offline task selection from aligned repeated rollouts, not solver truth."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .contact_face_selection import select_contact_pairs, process_face_contacts

POLICY = dict(schema='short_contact_repeatability_v1', short_seconds=.10,
              timing_tolerance_seconds=.04, minimum_vote_fraction=.5,
              vote_rule='strict_majority_unique_rollouts_same_part_primary_face',
              alignment='equal_recording_frames_verified_by_collection_seed',
              long_runs='unchanged', fill_missing_contacts=False)


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def filter_short_unrepeatable(pairs, source_pairs, *, fps):
    """One vote per rollout for the SAME part/face near an observed short run.

    Duration alone never rejects a run. No gap filling, altered points, or
    borrowing another part/face's support. Episode boundaries are kept separate.
    """
    n = len(pairs)
    if not source_pairs or any(len(row) != n for row in source_pairs):
        raise ValueError('Repeated contact evidence must be aligned complete rollouts')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid contact sample rate')
    radius = int(round(POLICY['timing_tolerance_seconds'] * fps))
    short_frames = max(1, int(np.floor(POLICY['short_seconds'] * fps + 1e-9)))
    keys = sorted({(p['part'], p['surface']) for row in pairs for p in row})
    reject = [set() for _ in pairs]
    report = []
    for key in keys:
        mask = np.array([any((p['part'], p['surface']) == key for p in row) for row in pairs])
        evidence = np.array([[any((p['part'], p['surface']) == key for p in row)
                              for row in rollout] for rollout in source_pairs])
        for a, b in intervals(mask):
            if b-a > short_frames:
                continue
            votes = evidence[:, max(0,a-radius):min(n,b+radius)].any(1)
            keep = votes.mean() > POLICY['minimum_vote_fraction']
            report.append(dict(part=int(key[0]), surface=int(key[1]), start=int(a),
                end_exclusive=int(b), frames=int(b-a), supporting_rollout_indices=np.flatnonzero(votes).tolist(),
                votes=int(votes.sum()), rollouts=len(source_pairs), kept=bool(keep),
                reason='repeatable_short_contact' if keep else 'short_unrepeatable_graze'))
            if not keep:
                for f in range(a,b):
                    reject[f].add(key)
    kept = [[p for p in row if (p['part'],p['surface']) not in reject[f]] for f,row in enumerate(pairs)]
    excluded = [[p for p in row if (p['part'],p['surface']) in reject[f]] for f,row in enumerate(pairs)]
    return kept, excluded, report


def process_with_consensus(pairs, catalog, *, fps, evidence):
    """Verify source identities on every load, including downstream cache reads."""
    from .newton_contact_data import load_raw_contact_labels
    if evidence['policy'] != POLICY:
        raise ValueError('Unknown consensus selection policy; rebuild explicitly')
    for key in ('motion', 'labels', 'seed'):
        if hashlib.sha256(Path(evidence[key]).read_bytes()).hexdigest() != evidence[key+'_sha256']:
            raise ValueError(f'Consensus evidence changed: {key}')
    load_raw_contact_labels(evidence['motion'], evidence['labels'])
    with np.load(evidence['seed'], allow_pickle=False) as z:
        ids = z['env_ids'].tolist(); n = int(z['frames_per_env'])
    if ids != evidence['env_ids'] or len(set(ids)) != len(ids) or n != len(pairs):
        raise ValueError('Consensus alignment/rollout identity mismatch')
    with np.load(evidence['motion'], allow_pickle=False) as z:
        if float(z['fps'].reshape(-1)[0]) != fps or len(z['joint_pos']) != n*len(ids):
            raise ValueError('Consensus sampling/frame count mismatch')
    with np.load(evidence['labels'], allow_pickle=False) as z:
        source_catalog = json.loads(z['contact_semantics_json'].item())['scene']['surface_catalog']
        if source_catalog != catalog:
            raise ValueError('Consensus sources use a different terrain face mapping')
        source = select_contact_pairs(json.loads(z['contact_pairs_json'].item()), catalog)['contact_pairs']
    if len(source) != n*len(ids):
        raise ValueError('Consensus pair frame count mismatch')
    selected = select_contact_pairs(pairs, catalog)
    kept, grazes, report = filter_short_unrepeatable(selected['contact_pairs'],
        [source[i*n:(i+1)*n] for i in range(len(ids))], fps=fps)
    result = process_face_contacts(kept, catalog, fps=fps)
    result['abnormal_contact_pairs'] = [side+graze for side,graze in zip(selected['abnormal_contact_pairs'],grazes)]
    result['contact_label_contract'] = dict(**result['contact_label_contract'], consensus=evidence)
    result['contact_consensus_report'] = report
    return result
