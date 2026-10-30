"""Observed event clocks for multi-execution denoising; no reference deadlines."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression


def event_times(report):
    """Keep event identity; repeated state confirmations may share an anchor."""
    result = {}
    for row in report["checks"]:
        if not row["passed"]:
            continue
        key = (row["action"], row["kind"], row["part"])
        # Support is an interval/dependency, not an extra clock deadline.
        if row["kind"] != "keep" and row.get("episode"):
            result[key+("start",)] = row["episode"]["interval"][0]
            if report.get("consensus_clock"):
                result[key+("end",)] = row["episode"]["interval"][1]
    return result


def align_observed_events(reports, frame_count, *, rate_bounds=(.25, 4.), reference_index=None,
                          motions=None, pose_band=10):
    """Soft event alignment with a strictly progressing, bounded C1 clock.

    Rate bounds are resampling controls, not contact or feasibility criteria.
    Incompatible event times retain an explicit residual instead of forcing
    a frozen clock followed by an arbitrarily fast catch-up.
    """
    if reports and reports[0].get('consensus_clock'):
        from .population_event_clock import align_population_events
        return align_population_events(reports, frame_count, rate_bounds=rate_bounds,
            reference_index=reference_index, motions=motions, pose_band=pose_band)
    if len(reports) < 3 or frame_count < 3:
        raise ValueError("Need at least three observed complete event sequences")
    if motions is not None and np.shape(motions) != (len(reports), frame_count, 36):
        raise ValueError('Pose evidence must match the observed executions')
    records = [event_times(r) for r in reports]
    keys = sorted(set.intersection(*(set(r) for r in records)))
    if not keys:
        raise ValueError("No shared observed events")
    source = np.asarray([[r[k] for k in keys] for r in records], float)
    if reference_index is None:
        distances = np.abs(source[:, None]-source[None]).mean(-1)
        reference_index = int(distances.sum(1).argmin())
    common = source[reference_index].copy()
    # Endpoints remain exact. An episode's exclusive end can equal frame_count.
    inside = (common > 0) & (common < frame_count-1)
    source, common = source[:, inside], common[inside]
    keys = [k for k, valid in zip(keys, inside) if valid]
    low, high = rate_bounds
    if not 0 < low < 1 < high:
        raise ValueError('Clock rate bounds must straddle one and exclude zero')
    # Identical physical episodes may satisfy several authored confirmations.
    # Merge their anchors; do not give repeated confirmations extra weight.
    groups = []
    for i in np.argsort(common):
        duplicate = next((g for g in groups if keys[g[0]][1:] == keys[i][1:]
                          and np.mean(source[:, g[0]] == source[:, i]) > .5), None)
        if duplicate is None: groups.append([int(i)])
        else: duplicate.append(int(i))
    groups.sort(key=lambda g: float(np.median(common[g])))
    resolution = max((r.get('evidence_policy', {}).get('stable_frames', 1) for r in reports), default=1)
    pooled = []
    for group in groups:
        if pooled:
            previous = pooled[-1]
            disjoint = not ({keys[i][2] for i in previous} & {keys[i][2] for i in group})
            delta = np.median(source[:, group], axis=1)-np.median(source[:, previous], axis=1)
            close_reversal = disjoint and np.median(np.abs(delta)) <= resolution and np.any(delta <= 0)
            same_time = np.median(common[group]) == np.median(common[previous])
            if close_reversal or same_time:
                previous.extend(group); continue
        pooled.append(group)
    grouped_times = np.asarray([np.median(source[:, g], axis=1) for g in pooled]).T if pooled else np.zeros((len(reports), 0))
    grouped_common = np.asarray([np.median(common[g]) for g in pooled])
    anchors = np.unique(grouped_common)
    mappings, clocks, diagnostics = [], [], []
    for index, row in enumerate(grouped_times):
        selected = np.asarray([np.median(row[grouped_common == t]) for t in anchors])
        if len(selected):
            selected = isotonic_regression(selected).x
        x = np.r_[0, anchors, frame_count-1]
        y = np.r_[0, np.clip(selected, 0, frame_count-1), frame_count-1]
        observed_y = y.copy()
        pose_status = 'disabled'
        if motions is not None and index != reference_index:
            from .contact_aggregation import match_pose_anchors
            try:
                # Pose matching only resolves the local event evidence window;
                # it cannot replace the observed event with a distant pose match.
                y = match_pose_anchors(motions[reference_index], motions[index], x, y,
                                       band=min(pose_band, int(resolution))).astype(float)
                pose_status = 'original_ordered_pose_dp_with_observed_event_centers'
            except ValueError as error:
                if str(error) != 'No event-order-preserving alignment':
                    raise
                pose_status = 'incompatible_pose_anchors_retained_observed_clock'
        curve = PchipInterpolator(x, y)
        derivative = curve.derivative()
        critical = curve.derivative(2).roots(extrapolate=False)
        probes = np.r_[x, critical[np.isfinite(critical)]]
        minimum, maximum = float(derivative(probes).min()), float(derivative(probes).max())
        blend = min(1., (1-low)/(1-minimum) if minimum < low else 1.,
                    (high-1)/(maximum-1) if maximum > high else 1.)
        frames = np.arange(frame_count, dtype=float)
        mapping = frames+blend*(curve(frames)-frames)
        mapping[0], mapping[-1] = 0., float(frame_count-1)
        if np.any(np.diff(mapping) < low-1e-8) or np.any(np.diff(mapping) > high+1e-8):
            raise AssertionError("Invalid shared event clock")
        mappings.append(mapping); clocks.append(y.tolist())
        diagnostics.append(dict(identity_blend=blend, original_rate_range=[minimum, maximum],
                                final_rate_range=[1+blend*(minimum-1), 1+blend*(maximum-1)],
                                anchor_residual=(x+blend*(curve(x)-x)-observed_y).tolist(),
                                observed_source_anchors=observed_y.tolist(), pose_matching=pose_status,
                                pose_anchor_adjustment=(y-observed_y).tolist()))
    return np.asarray(mappings), dict(
        schema="observed_partial_order_event_alignment_v3", reference_index=reference_index,
        timeline="representative physical execution; keep intervals are validation only",
        event_keys=[list(k) for k in keys], event_groups=[[list(keys[i]) for i in g] for g in pooled],
        common_event_frames=common.tolist(), output_anchors=np.r_[0, anchors, frame_count-1].tolist(),
        source_anchors=clocks, reference_frame_deadlines=False,
        clock_diagnostics=diagnostics, rate_bounds=list(rate_bounds),
        independent_event_reversal="pool unresolved nearby independent events; soften incompatible anchors",
        pose_transport="continuous shared clock, linear translation/joints and quaternion slerp; fresh Newton labels required")


def resample_motion(q, mapping):
    """C2 whole-body transport; contacts must be queried at the new poses."""
    from .continuous_motion import ContinuousMotion
    mapping = np.asarray(mapping)
    if not np.isfinite(mapping).all() or (mapping < 0).any() or (mapping > len(q)-1).any():
        raise ValueError('Invalid continuous pose clock')
    result = ContinuousMotion(q).evaluate(mapping)[0]
    exact = mapping == np.rint(mapping)
    result[exact] = np.asarray(q)[mapping[exact].astype(int)]
    return result.astype(np.asarray(q).dtype)


def contact_intent(report, mask, surface):
    """Aligned observed episodes supply fitting intent, never replacement labels.

    Only explicitly matched releases are supervised. Bounded holes already
    allowed by the evidence policy become optimization targets inside a matched
    support episode; raw observations are kept unchanged.
    """
    wanted = np.zeros_like(mask, dtype=bool)
    faces = np.full_like(surface, -1)
    release = np.zeros_like(wanted)
    for row in report["checks"]:
        if not row["passed"]:
            raise ValueError("Cannot derive consensus intent from a failed execution")
        part = row["part"]
        episode = row.get("support_episode") if row["kind"] == "keep" else row.get("episode")
        if episode is None:
            continue
        if row["kind"] == "release":
            a, b = episode["stable_evidence"]
            release[a:b, part] = True
        else:
            a, b = episode["interval"]
            observed = surface[a:b, part][mask[a:b, part]]
            if not len(observed) or len(np.unique(observed)) != 1:
                raise ValueError("Matched episode lacks a unique actual primary face")
            face = int(observed[0])
            if np.any((faces[a:b, part] >= 0) & (faces[a:b, part] != face)):
                raise ValueError("Conflicting matched contact episodes")
            wanted[a:b, part] = True; faces[a:b, part] = face
    if np.any(wanted & release):
        raise ValueError("Matched contact/release intents overlap")
    return wanted, faces, release
