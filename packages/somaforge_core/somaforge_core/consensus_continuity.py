"""Local, event-phase-relative continuity acceptance, separate from contacts.

This is an empirical anomaly gate, not a proof of dynamic feasibility. RMS is
deliberately absent: every frame and every channel must pass independently.
"""
import numpy as np
import torch

from . import G1_29DOF_JOINT_ORDER
from .consensus_event_alignment import event_times

SCHEMA = 'event_phase_local_continuity_v2'


def continuous_source_clock(q, output_anchors, source_anchors):
    """Calibration only: avoid integer clock repeats/skips inflating bounds.

    Interpolated poses never supply contact labels or replace fitted output.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation, Slerp
    frames = np.arange(len(q))
    clock = np.clip(PchipInterpolator(output_anchors, source_anchors)(frames), 0, len(q)-1)
    result = np.stack([np.interp(clock, frames, column) for column in q.T], -1)
    rotation = Rotation.from_quat(q[:, [4, 5, 6, 3]])
    result[:, 3:7] = Slerp(frames, rotation)(clock).as_quat()[:, [3, 0, 1, 2]]
    return result


def _angular_increment(q):
    """Shortest world-frame rotation vector; invariant to quaternion signs."""
    q = torch.nn.functional.normalize(q, dim=-1)
    a, b = q[..., :-1, :], q[..., 1:, :]
    w = (a*b).sum(-1, keepdim=True)
    xyz = a[..., :1]*b[..., 1:]-b[..., :1]*a[..., 1:]+torch.linalg.cross(a[..., 1:], b[..., 1:])
    xyz = torch.where(w < 0, -xyz, xyz)
    norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2*torch.atan2(norm, w.abs())
    return xyz*torch.where(norm > 1e-6, angle/norm.clamp_min(1e-6), 2*torch.ones_like(norm))


class ContinuityGuard:
    def __init__(self, sources, alignment, clock_indices, fps, fk, *, source_reports=None, mad_scale=3.0):
        sources = np.asarray(sources)
        if (sources.ndim != 3 or sources.shape[0] < 3 or sources.shape[1] < 3
                or sources.shape[2] != 36 or not np.isfinite(sources).all()
                or not np.isfinite(fps) or fps <= 0 or not np.isfinite(mad_scale) or mad_scale < 0
                or len(clock_indices) != len(sources)):
            raise ValueError('Continuity needs at least three finite qualified source motions and valid clocks/fps')
        if np.any(np.linalg.norm(sources[..., 3:7], axis=-1) < .99):
            raise ValueError('Invalid source root quaternion')
        self.fps, self.n, self.mad_scale = float(fps), sources.shape[1], mad_scale
        self.anchors = np.asarray(alignment['output_anchors'], float)
        if (self.anchors[0] != 0 or self.anchors[-1] != self.n-1 or (np.diff(self.anchors) <= 0).any()):
            raise ValueError('Invalid continuity event phases')
        self.events = dict(keys=alignment['event_keys'], frames=alignment['common_event_frames'])
        self.native_intervals = []
        records = None if source_reports is None else [event_times(r) for r in source_reports]
        if records is not None and len(records) != len(sources):
            raise ValueError('Continuity source event report count differs')
        for source_index, clock_index in enumerate(clock_indices):
            intervals = []
            for phase, (left, right) in enumerate(zip(self.anchors[:-1], self.anchors[1:])):
                if records is None:
                    # Synthetic/pose-only callers can supply clocks without
                    # event identities. Authored tasks always pass raw reports.
                    times = alignment['source_anchors'][clock_index][phase:phase+2]
                else:
                    times = [records[source_index][tuple(key)] for key, frame in
                             zip(self.events['keys'], self.events['frames']) if frame in (left, right)]
                    if left == 0: times.append(0)
                    if right == self.n-1: times.append(self.n-1)
                intervals.append([max(0, min(times)), min(self.n-1, max(times))])
            self.native_intervals.append(intervals)
        self.links = (fk.root_link,)+tuple(row[1] for row in fk.joint_records)
        self.channels = ['root_position', 'root_orientation']+list(G1_29DOF_JOINT_ORDER)+[
            f'{link}/marker_{axis}' for link in self.links for axis in ('origin', 'x', 'y', 'z')]
        calibrated = np.asarray([continuous_source_clock(q, self.anchors, alignment['source_anchors'][i])
                                 for q, i in zip(sources, clock_indices)])
        tensor = torch.as_tensor(calibrated, dtype=fk.joint_lower.dtype, device=fk.joint_lower.device)
        self.bounds, self.phase_bounds, self.phase_members = [], [], []
        # A floating-point allowance, not a task distance/contact threshold.
        coordinate_scale = max(1., float(np.abs(sources).max()))
        self.roundoff = 64*np.finfo(np.float32).eps*coordinate_scale
        with torch.no_grad():
            features = self.features(tensor, fk)
            original = self.features(torch.as_tensor(sources, dtype=tensor.dtype, device=tensor.device), fk)
        for order, (feature, native_feature) in enumerate(zip(features, original), 1):
            values = feature.cpu().numpy()
            native_values = native_feature.cpu().numpy()
            starts = np.arange(self.n-order)
            members, phase_limits = [], []
            for phase, (left, right) in enumerate(zip(self.anchors[:-1], self.anchors[1:])):
                # Include a derivative stencil crossing an event in both
                # adjacent phases; no fixed +/- frame tolerance is introduced.
                member = (starts < right) & (starts+order > left)
                peaks = values[:, member].max(axis=1)
                # Isotonic pooling can freeze a clock; that is not evidence
                # of physical rest. Preserve each source's original dynamics
                # in the RAW event interval, allowing independent reversals.
                for source_index in range(len(sources)):
                    a, b = self.native_intervals[source_index][phase]
                    native_member = (starts <= b) & (starts+order >= a)
                    peaks[source_index] = np.maximum(peaks[source_index], native_values[source_index, native_member].max(0))
                median = np.median(peaks, axis=0)
                mad = np.median(np.abs(peaks-median), axis=0)
                phase_limits.append(median+mad_scale*1.4826*mad+self.roundoff*self.fps**order)
                members.append(member)
            phase_limits, members = np.asarray(phase_limits), np.asarray(members)
            limits = np.max(np.where(members[..., None], phase_limits[:, None], 0), axis=0)
            self.bounds.append(torch.as_tensor(limits, dtype=tensor.dtype, device=tensor.device))
            self.phase_bounds.append(phase_limits.tolist()); self.phase_members.append(members)

    def features(self, q, fk):
        p, r = fk.link_poses(q, self.links)
        offsets = q.new_tensor([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
        markers = (p[..., None, :]+torch.einsum('...ij,kj->...ki', r, offsets)).flatten(-3, -2)
        vectors = [torch.diff(q[..., :3], dim=-2), _angular_increment(q[..., 3:7]),
                   torch.diff(q[..., 7:], dim=-2), torch.diff(markers, dim=-3)]
        def magnitudes(parts):
            return torch.cat((parts[0].norm(dim=-1, keepdim=True), parts[1].norm(dim=-1, keepdim=True),
                              parts[2].abs(), parts[3].norm(dim=-1)), dim=-1)
        return [magnitudes(vectors)*self.fps,
                magnitudes([torch.diff(v, dim=-3 if i == 3 else -2) for i, v in enumerate(vectors)])*self.fps**2]

    def objective(self, q, fk):
        ratios = [value/bound.to(q) for value, bound in zip(self.features(q, fk), self.bounds)]
        excess = torch.cat([(ratio-1).relu().flatten() for ratio in ratios])
        # Max prevents one bad frame from disappearing in a whole-motion mean.
        cost = excess.square().max()+excess.square().mean()
        return cost, int((excess.detach() > 1e-5).sum()), float(excess.detach().max())

    def check(self, q, fk):
        q = torch.as_tensor(q, dtype=fk.joint_lower.dtype, device=fk.joint_lower.device)
        if q.shape != (self.n, 36) or not torch.isfinite(q).all():
            raise ValueError('Invalid output pose for continuity audit')
        if not torch.allclose(q[:, 3:7].norm(dim=-1), torch.ones_like(q[:, 0]), atol=1e-4, rtol=0):
            raise ValueError('Output root quaternion is not unit length')
        violations, maxima = [], []
        with torch.no_grad():
            features = self.features(q, fk)
        for order, (value, bound, members) in enumerate(zip(features, self.bounds, self.phase_members), 1):
            values, limits = value.cpu().numpy(), bound.cpu().numpy()
            ratios = values/limits
            maxima.append(float(ratios.max()))
            for frame, channel in np.argwhere(ratios > 1+1e-5):
                violations.append(dict(frame=int(frame+1), stencil=[int(frame), int(frame+order)],
                    derivative_order=order, channel=self.channels[channel], value=float(values[frame, channel]),
                    limit=float(limits[frame, channel]), ratio=float(ratios[frame, channel]),
                    phases=np.flatnonzero(members[:, frame]).tolist()))
        return dict(schema=SCHEMA, passed=not violations, violations=violations,
            violation_count=len(violations), violation_frames=sorted({r['frame'] for r in violations}),
            max_ratio_by_derivative=maxima, fps=self.fps,
            channels=self.channels, phase_anchors=self.anchors.tolist(), events=self.events,
            phase_bounds=self.phase_bounds, native_event_intervals=self.native_intervals, mad_scale=self.mad_scale,
            calibration='per-channel phase peak: source median + scaled MAD + float32 roundoff',
            source_clock='continuous original shared PCHIP; linear position/joints and shortest quaternion slerp; calibration only',
            clock_pooling='phase peak is at least the raw source event-interval peak; a pooled clock plateau is not physical rest',
            derivative_units='m/s or rad/s; m/s^2 or rad/s^2; world-frame root rotation vectors',
            physical_feasibility_certified=False, rms_is_acceptance_condition=False)


def acceptance_status(task, continuity):
    if task.get('schema') == 'native_population_information_consensus_v2':
        if not task['passed']: return 'stable_information_loss'
        return 'accepted' if continuity['passed'] else 'introduced_variation_unresolved'
    if not task['passed']:
        return 'contact_guard_unresolved'
    return 'accepted' if continuity['passed'] else 'local_continuity_unresolved'
