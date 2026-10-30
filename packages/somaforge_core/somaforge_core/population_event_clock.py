"""Continuous population clocks with explicit missing-event observations."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression


def smooth_clock_rates(mapping, resolution, rate_bounds):
    """Remove rapid changes in clock speed without changing its endpoints."""
    from scipy.ndimage import gaussian_filter1d
    mapping = np.asarray(mapping, float)
    rates = gaussian_filter1d(np.diff(mapping), float(resolution), mode='nearest')
    rates *= (mapping[-1]-mapping[0])/rates.sum()
    low, high = rate_bounds
    minimum, maximum = rates.min(), rates.max()
    blend = min(1., (1-low)/(1-minimum) if minimum < low else 1.,
                (high-1)/(maximum-1) if maximum > high else 1.)
    rates = 1+blend*(rates-1)
    result = np.r_[mapping[0], mapping[0]+np.cumsum(rates)]
    result[-1] = mapping[-1]
    return result


def align_population_events(reports, frame_count, *, rate_bounds=(.25, 4.),
                            reference_index=None, motions=None, pose_band=10):
    from .consensus_event_alignment import event_times, resample_motion
    from .contact_aggregation import match_pose_anchors
    if len(reports) < 3 or frame_count < 3:
        raise ValueError('Need three nonempty execution sequences')
    if motions is not None and np.shape(motions) != (len(reports), frame_count, 36):
        raise ValueError('Pose evidence must match executions')
    clock = reports[0]['consensus_clock']
    if any(r.get('consensus_clock') != clock for r in reports):
        raise ValueError('Executions must share the same population consensus')
    records = [event_times(r) for r in reports]
    keys = sorted(set.union(*(set(r) for r in records)))
    common = {k: float(clock[str(k[0])][0 if k[-1] == 'start' else 1]) for k in keys}
    keys = [k for k in keys if 0 < common[k] < frame_count-1]
    low, high = rate_bounds
    if not 0 < low < 1 < high: raise ValueError('Invalid clock rate bounds')
    frames = np.arange(frame_count, dtype=float)
    reference_index = 0 if reference_index is None else reference_index
    if not 0 <= reference_index < len(records): raise ValueError('Invalid pose representative')

    def curve_map(x, y):
        curve = PchipInterpolator(x, y)
        critical = curve.derivative(2).roots(extrapolate=False)
        probes = np.r_[x, critical[np.isfinite(critical)]]
        rates = curve.derivative()(probes)
        minimum, maximum = float(rates.min()), float(rates.max())
        blend = min(1., (1-low)/(1-minimum) if minimum < low else 1.,
                    (high-1)/(maximum-1) if maximum > high else 1.)
        mapping = frames+blend*(curve(frames)-frames)
        mapping[0], mapping[-1] = 0, frame_count-1
        raw_bend = float(np.abs(np.diff(mapping, n=2)).max())
        resolution = max(r['evidence_policy']['stable_frames'] for r in reports)
        mapping = smooth_clock_rates(mapping, resolution, rate_bounds)
        if np.any(np.diff(mapping) < low-1e-8) or np.any(np.diff(mapping) > high+1e-8):
            raise AssertionError('Invalid population clock rates')
        return mapping, dict(identity_blend=blend, original_rate_range=[minimum, maximum],
            final_rate_range=[1+blend*(minimum-1), 1+blend*(maximum-1)],
            anchor_residual=(np.interp(x, frames, mapping)-y).tolist(),
            raw_max_clock_second_difference=raw_bend,
            max_clock_second_difference=float(np.abs(np.diff(mapping, n=2)).max()),
            rate_smoothing_samples=int(resolution))

    knots, provisional = [], []
    for record in records:
        present = [k for k in keys if k in record]
        x = np.asarray(sorted({0., float(frame_count-1), *(common[k] for k in present)}))
        observed = np.asarray([0. if t == 0 else float(frame_count-1) if t == frame_count-1
            else np.median([record[k] for k in present if common[k] == t]) for t in x])
        y = np.clip(isotonic_regression(observed).x, 0, frame_count-1)
        mapping, _ = curve_map(x, y)
        knots.append((x, y, observed, present)); provisional.append(mapping)
    pose_reference = None if motions is None else resample_motion(motions[reference_index], provisional[reference_index])
    mappings, diagnostics, source_anchors = [], [], []
    for index, (x, y, observed, present) in enumerate(knots):
        status = 'event_clock_only'
        if pose_reference is not None and index != reference_index:
            resolution = int(reports[index]['evidence_policy']['stable_frames'])
            try:
                y = match_pose_anchors(pose_reference, motions[index], x, y,
                                       band=min(pose_band, resolution)).astype(float)
                status = 'ordered_pose_dp_on_population_clock'
            except ValueError as error:
                if str(error) != 'No event-order-preserving alignment': raise
                status = 'pose_dp_incompatible_retained_event_clock'
        mapping, diagnostic = curve_map(x, y)
        diagnostic.update(output_anchors=x.tolist(), observed_source_anchors=observed.tolist(),
            anchor_residual=(np.interp(x, frames, mapping)-observed).tolist(),
            pose_matching=status, pose_anchor_adjustment=(y-observed).tolist(),
            missing_event_keys=[list(k) for k in keys if k not in present])
        mappings.append(mapping); diagnostics.append(diagnostic); source_anchors.append(y.tolist())
    registration = None
    if motions is not None:
        from .continuous_motion import joint_motion_clocks
        mappings, registration = joint_motion_clocks(motions, mappings,
            [(x, observed) for x, _, observed, _ in knots],
            resolution=max(r['evidence_policy']['stable_frames'] for r in reports), rate_bounds=rate_bounds)
        for mapping, diagnostic, (x, _, observed, _) in zip(mappings, diagnostics, knots):
            diagnostic.update(final_rate_range=[float(np.diff(mapping).min()), float(np.diff(mapping).max())],
                anchor_residual=(np.interp(x, frames, mapping)-observed).tolist(),
                max_clock_second_difference=float(np.abs(np.diff(mapping, n=2)).max()))
    return np.asarray(mappings), dict(schema='population_event_clock_v3', reference_index=reference_index,
        joint_registration=registration,
        timeline='population event consensus', event_keys=[list(k) for k in keys],
        common_event_frames=[common[k] for k in keys],
        output_anchors=sorted({0., float(frame_count-1), *(common[k] for k in keys)}),
        source_anchors=source_anchors, clock_diagnostics=diagnostics, rate_bounds=list(rate_bounds),
        reference_frame_deadlines=False, missing_events='no invented anchor; align observed endpoints only',
        pose_transport='shared continuous whole-body clock; fresh Newton labels required')
