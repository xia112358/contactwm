"""Continuity on the OUTPUT's observed event phases, without speed deadlines."""
import numpy as np
import torch

from . import G1_29DOF_JOINT_ORDER
from .consensus_continuity import ContinuityGuard
from .consensus_event_alignment import event_times


class EventContinuityGuard:
    features = ContinuityGuard.features

    def __init__(self, sources, reports, fps, fk, *, mad_scale=3., reference_clock='native'):
        sources = np.asarray(sources)
        if sources.ndim != 3 or sources.shape[0] < 3 or sources.shape[-1] != 36 or not np.isfinite(sources).all():
            raise ValueError('Need three finite canonical source executions')
        if len(reports) != len(sources) or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid source reports/fps')
        self.n, self.fps, self.mad_scale = sources.shape[1], float(fps), float(mad_scale)
        if reference_clock not in ('native','transported'):
            raise ValueError('Unknown continuity reference clock')
        self.reference_clock = reference_clock
        self.records = [event_times(r) for r in reports]
        self.links = (fk.root_link,)+tuple(row[1] for row in fk.joint_records)
        self.channels = ['root_position', 'root_orientation']+list(G1_29DOF_JOINT_ORDER)+[
            f'{link}/marker_{axis}' for link in self.links for axis in ('origin', 'x', 'y', 'z')]
        self.roundoff = 64*np.finfo(np.float32).eps*max(1., float(np.abs(sources).max()))
        self.device, self.dtype = fk.joint_lower.device, fk.joint_lower.dtype
        with torch.no_grad():
            self.source_features = [v.cpu().numpy() for v in self.features(
                torch.as_tensor(sources, dtype=self.dtype, device=self.device), fk)]
        self.bounds = None

    def update(self, report):
        output = event_times(report)
        keys = set(output).intersection(*(set(r) for r in self.records))
        self.events = {k: output[k] for k in keys if 0 < output[k] < self.n-1}
        self.anchors = np.asarray(sorted({0, self.n-1, *self.events.values()}), float)
        self.native_intervals = []
        for record in self.records:
            phases = []
            for left, right in zip(self.anchors[:-1], self.anchors[1:]):
                times = [record[k] for k, t in self.events.items() if t in (left, right)]
                if left == 0: times.append(0)
                if right == self.n-1: times.append(self.n-1)
                phases.append([float(np.clip(min(times), 0, self.n-1)),
                               float(np.clip(max(times), 0, self.n-1))])
            self.native_intervals.append(phases)
        self.bounds, self.phase_bounds, self.phase_members, self.statistical_bounds = [], [], [], []
        for order, features in enumerate(self.source_features, 1):
            starts = np.arange(self.n-order)
            limits, statistical, members = [], [], []
            for phase, (left, right) in enumerate(zip(self.anchors[:-1], self.anchors[1:])):
                peaks = []
                for i, intervals in enumerate(self.native_intervals):
                    a, b = intervals[phase]
                    peaks.append(features[i, (starts <= b) & (starts+order >= a)].max(0))
                peaks = np.asarray(peaks); center = np.median(peaks, axis=0)
                statistical.append(center+self.mad_scale*1.4826*np.median(np.abs(peaks-center), axis=0)
                                   +self.roundoff*self.fps**order)
                # Existing executions define the observed envelope. A source's
                # deviation from a population median is not a new discontinuity.
                limits.append(peaks.max(axis=0)+self.roundoff*self.fps**order)
                members.append((starts < right) & (starts+order > left))
            limits, members = np.asarray(limits), np.asarray(members)
            bound = np.max(np.where(members[..., None], limits[:, None], 0), axis=0)
            self.bounds.append(torch.as_tensor(bound, dtype=self.dtype, device=self.device))
            self.phase_bounds.append(limits.tolist()); self.phase_members.append(members)
            self.statistical_bounds.append(np.max(np.where(members[..., None], np.asarray(statistical)[:, None], 0), axis=0))

    def objective(self, q, fk, validation=None):
        if validation is not None: self.update(validation)
        if self.bounds is None: raise ValueError('Continuity requires actual output event observations')
        # A sustained high speed is not itself a discontinuity. Isolated pose
        # jumps and angular/linear velocity bends are caught by second differences.
        ratio = self.features(q, fk)[1]/self.bounds[1]
        excess = (ratio-1).relu()
        interior = (ratio-.85).relu().square()
        return interior.max()+interior.mean(), int((excess.detach() > 1e-5).sum()), float(excess.detach().max())

    def check(self, q, fk, report):
        self.update(report)
        q = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        if q.shape != (self.n, 36) or not torch.isfinite(q).all(): raise ValueError('Invalid output pose')
        if not torch.allclose(q[:, 3:7].norm(dim=-1), torch.ones_like(q[:, 0]), atol=1e-4, rtol=0):
            raise ValueError('Invalid root quaternion')
        with torch.no_grad(): features = [v.cpu().numpy() for v in self.features(q, fk)]
        violations, speed_flags = [], []
        maxima = []
        for order, (values, bounds, members) in enumerate(zip(features, self.bounds, self.phase_members), 1):
            limits = bounds.cpu().numpy(); ratios = values/limits; maxima.append(float(ratios.max()))
            for frame, channel in np.argwhere(ratios > 1+1e-5):
                item = dict(frame=int(frame+1), stencil=[int(frame), int(frame+order)], derivative_order=order,
                    channel=self.channels[channel], value=float(values[frame, channel]), limit=float(limits[frame, channel]),
                    ratio=float(ratios[frame, channel]), phases=np.flatnonzero(members[:, frame]).tolist())
                (speed_flags if order == 1 else violations).append(item)
        return dict(schema=('transported_source_information_continuity_v3' if self.reference_clock=='transported'
                            else 'source_envelope_information_continuity_v2'), passed=not violations,
            reference_clock=self.reference_clock,
            violation_count=len(violations), violation_frames=sorted({v['frame'] for v in violations}), violations=violations,
            speed_diagnostics=speed_flags, speed_is_discontinuity=False, max_ratio_by_derivative=maxima,
            fps=self.fps, channels=self.channels, phase_anchors=self.anchors.tolist(),
            events=[dict(key=list(k), frame=t) for k, t in sorted(self.events.items())],
            native_event_intervals=self.native_intervals, phase_bounds=self.phase_bounds,
            phase_clock='actual matched Newton output events; no reference-frame deadlines',
            calibration=f'{self.reference_clock} source corresponding-phase maximum envelope + float32 roundoff',
            statistical_deviation_count=int((features[1] > self.statistical_bounds[1]*(1+1e-5)).sum()),
            statistical_deviation_is_error=False, source_observations_are_baseline=True,
            rms_is_acceptance_condition=False, physical_feasibility_certified=False)
