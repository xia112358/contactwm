"""Inspectable event-aligned aggregation, without whole-trajectory pose descent.

All distances here are fitting diagnostics, never contact definitions. Callers
must obtain fresh Newton evidence after every pose or clock transformation.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def compatible_sources(reports):
    """Group successful executions by physical-episode identity and support.

    Independent limbs have no imposed total order. Repeated confirmations of
    one episode must mean the same thing within a group; keep dependencies are
    part of compatibility but never added as time anchors.
    """
    groups = {}
    rejected = []
    for index, report in enumerate(reports):
        if not report['passed']:
            rejected.append(dict(index=index, reason='source_contact_contract_failed'))
            continue
        episodes, signature = {}, []
        for row in report['checks']:
            episode = row.get('support_episode') if row['kind'] == 'keep' else row.get('episode')
            if episode is None:
                identity = None
            else:
                key = (row['part'], row['surface'], row['kind'] == 'release', tuple(episode['interval']))
                identity = episodes.setdefault(key, len(episodes))
            signature.append((row['action'], row['kind'], row['part'], row['surface'], identity,
                              tuple(sorted((d['action'], d['part']) for d in row.get('dependent_events', [])))))
        groups.setdefault(tuple(signature), []).append(index)
    clusters = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    return clusters, dict(schema='physical_episode_compatibility_v1', groups=clusters, rejected=rejected,
                         cross_limb_total_order=False, minimum_contributors=3)


def pose_residual(sources, base):
    """Translation, root SO(3) tangent, and bounded (nonperiodic) joints."""
    sources = np.asarray(sources)
    shape = sources.shape[:-1]
    ref = np.broadcast_to(base, sources.shape)
    r = Rotation.from_quat(sources[..., [4, 5, 6, 3]].reshape(-1, 4))
    b = Rotation.from_quat(ref[..., [4, 5, 6, 3]].reshape(-1, 4))
    angle = (b.inv()*r).as_rotvec().reshape(*shape, 3)
    return np.concatenate((sources[..., :3]-ref[..., :3], angle, sources[..., 7:]-ref[..., 7:]), -1)


def apply_residual(base, delta):
    out = np.asarray(base).copy()
    r = Rotation.from_quat(base[:, [4, 5, 6, 3]])*Rotation.from_rotvec(delta[:, 3:6])
    out[:, :3] += delta[:, :3]
    out[:, 3:7] = r.as_quat()[:, [3, 0, 1, 2]]
    out[:, 7:] += delta[:, 6:]
    return out


def motion_medoid(sources):
    """One actual execution, with a single whole-body distance per pair."""
    sources = np.asarray(sources)
    scale = np.r_[np.full(3, .1), np.full(3, .3), np.full(29, .3)]
    distances = np.asarray([np.sqrt(np.mean((pose_residual(sources, base)/scale)**2, axis=(1, 2)))
                            for base in sources])
    return int(distances.sum(1).argmin()), distances


def regularize(values, strength):
    """Second-difference least squares; natural endpoints, no frame snapping."""
    n = len(values)
    d = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    return spsolve((eye(n)+strength*(d.T@d)).tocsc(), values)


def fusion_stages(aligned, reference_index, anchors, *, fps):
    """Whole-body phase weights, smoothly interpolated; residuals on SO(3).

    Cauchy weights use one score per execution/phase, never one per joint.
    Residual smoothing is scaled in physical time and preserves the medoid's
    fine motion, rather than smoothing its contacts away globally.
    """
    aligned = np.asarray(aligned)
    base = aligned[reference_index]
    residual = pose_residual(aligned, base)
    n = len(base)
    anchors = np.unique(np.clip(np.asarray(anchors, float), 0, n-1))
    scale = np.r_[np.full(3, .1), np.full(3, .3), np.full(29, .3)]
    weights, centers = [], []
    for left, right in zip(anchors[:-1], anchors[1:]):
        ix = (np.arange(n) >= left) & (np.arange(n) <= right)
        score = np.sqrt(np.mean((residual[:, ix]/scale)**2, axis=(1, 2)))
        width = max(float(np.median(score)), np.finfo(float).eps)
        w = 1/(1+(score/width)**2)
        weights.append(w/w.sum()); centers.append((left+right)/2)
    if not weights:
        raise ValueError('No nonempty physical phases')
    centers = np.asarray(centers); weights = np.asarray(weights)
    frame_weights = np.empty((n, len(aligned)))
    # Quintic interpolation has zero first/second derivative at phase centers.
    for frame in range(n):
        upper = int(np.searchsorted(centers, frame))
        if upper == 0: frame_weights[frame] = weights[0]
        elif upper == len(centers): frame_weights[frame] = weights[-1]
        else:
            t = (frame-centers[upper-1])/(centers[upper]-centers[upper-1])
            blend = t**3*(10-15*t+6*t*t)
            frame_weights[frame] = (1-blend)*weights[upper-1]+blend*weights[upper]
    delta = np.einsum('te,etd->td', frame_weights, residual)
    # 80 ms is a smoothing scale, not an event duration or contact tolerance.
    smooth = regularize(delta, (.08*fps)**4)
    return base, delta, smooth, dict(schema='representative_residual_fusion_v1',
        reference_index=reference_index, phase_centers=centers.tolist(), phase_weights=weights.tolist(),
        smoothing_time_scale_seconds=.08, rotation='right tangent of representative SO(3)',
        weight_semantics='one smoothly varying weight per whole execution; shared by all body channels')


def localized_transfer(candidate, representative, failed_report, *, fps,
                       representative_report=None, candidate_mask=None, candidate_surface=None):
    """Try a bounded local return toward known execution around failed events.

    This is not a contact solver. The resulting pose requires a fresh query.
    A quintic envelope makes the correction C2 at its support boundary. When
    there is no localized failed-event evidence, refuse a global correction.
    """
    n = len(candidate)
    envelope = np.zeros(n)
    for row in failed_report['checks']:
        if row['passed']:
            continue
        target = row
        if representative_report is not None:
            matches = [r for r in representative_report['checks'] if
                       (r['action'], r['kind'], r['part']) == (row['action'], row['kind'], row['part'])]
            if len(matches) != 1 or not matches[0]['passed']:
                continue
            target = matches[0]
        episode = target.get('support_episode') or target.get('episode')
        if not episode:
            continue
        a, b = episode['interval']
        radius = max(2, int(round(.12*fps)))
        if representative_report is not None and candidate_mask is not None:
            active = candidate_mask[a:b, row['part']] & (candidate_surface[a:b, row['part']] == target['surface'])
            bad = np.flatnonzero(active if row['kind'] == 'release' else ~active)+a
            if not len(bad):
                continue
            split = np.flatnonzero(np.diff(bad) > 1)+1
            windows = [(int(part[0]), int(part[-1])) for part in np.split(bad, split)]
        else:
            if b-a > 2*radius:
                continue
            windows = [(a, b)]
        for a, b in windows:
            for frame in range(max(0, a-radius), min(n, b+radius+1)):
                t = min(1., max(0., min((frame-a+radius)/radius, (b+radius-frame)/radius)))
                envelope[frame] = max(envelope[frame], t**3*(10-15*t+6*t*t))
    if np.count_nonzero(envelope) > n/4:
        return None, dict(reason='correction_is_not_local', proposed_frames=int(np.count_nonzero(envelope)))
    delta = pose_residual(representative[None], candidate)[0]
    # Trust region is a modification budget, never a contact threshold.
    if (np.linalg.norm(delta[envelope > 0, :3], axis=-1).max(initial=0) > .02
            or np.abs(delta[envelope > 0, 3:]).max(initial=0) > .1):
        return None, dict(reason='local_correction_exceeds_modification_budget')
    if not envelope.any():
        return None, dict(reason='no_localizable_failed_episode')
    return apply_residual(candidate, delta*envelope[:, None]), dict(
        reason='local_representative_transfer', changed_frames=np.flatnonzero(envelope).tolist(),
        translation_budget_m=.02, angular_budget_rad=.1, envelope='quintic C2')
