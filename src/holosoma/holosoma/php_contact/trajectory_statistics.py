"""Whole-trajectory head attempts and same-solve action-boundary observations."""
import csv
from pathlib import Path

import torch


class TrajectoryStatistics:
    motion_fields = ('sampled', 'head_sampled', 'head_finished', 'head_reached_end', 'head_all_contacts')
    boundary_fields = ('observed', 'contact_success', 'unknown', 'first_failure')

    def __init__(self, motion_keys, segment_keys, device):
        self.motion_keys, self.segment_keys = list(motion_keys), list(segment_keys)
        self.counts = torch.zeros(len(motion_keys), 5, dtype=torch.long, device=device)
        self.boundaries = torch.zeros(len(segment_keys), 4, dtype=torch.long, device=device)
        self.errors = torch.zeros(len(segment_keys), 2, dtype=torch.float64, device=device)
        self.progress = torch.zeros(len(motion_keys), dtype=torch.float64, device=device)
        self.clear_interval()

    def clear_interval(self):
        self.interval_counts = torch.zeros_like(self.counts)
        self.interval_boundaries = torch.zeros_like(self.boundaries)
        self.interval_errors = torch.zeros_like(self.errors)
        self.interval_progress = torch.zeros_like(self.progress)

    def sampled(self, motions, heads):
        values = self.counts.new_zeros(len(motions), 5)
        values[:, 0], values[:, 1] = 1, heads.long()
        self.counts.index_add_(0, motions, values)
        self.interval_counts.index_add_(0, motions, values)

    def boundary(self, segments, known, success, root_error, limb_error):
        values = self.boundaries.new_zeros(len(segments), 4)
        values[:, 0], values[:, 1], values[:, 2] = 1, (success & known).long(), (~known).long()
        errors = torch.stack((root_error, limb_error), -1).to(self.errors)
        for target in (self.boundaries, self.interval_boundaries): target.index_add_(0, segments, values)
        for target in (self.errors, self.interval_errors): target.index_add_(0, segments, errors)

    def finished(self, motions, segments, reached, all_contacts, time_progress):
        values = self.counts.new_zeros(len(motions), 5)
        values[:, 2], values[:, 3], values[:, 4] = 1, reached.long(), (reached & all_contacts).long()
        for target in (self.counts, self.interval_counts): target.index_add_(0, motions, values)
        for target in (self.progress, self.interval_progress):
            target.index_add_(0, motions, time_progress.clamp(0, 1).to(target))
        failures = self.boundaries.new_zeros(len(segments), 4)
        failures[:, 3] = (~reached).long()
        for target in (self.boundaries, self.interval_boundaries): target.index_add_(0, segments, failures)

    def metrics(self, cumulative=False):
        counts = self.counts if cumulative else self.interval_counts
        boundaries = self.boundaries if cumulative else self.interval_boundaries
        errors = self.errors if cumulative else self.interval_errors
        progress = self.progress if cumulative else self.interval_progress
        sampled, heads, ended, reached, all_contacts = counts.sum(0).cpu().tolist()
        observed, contact, unknown, failures = boundaries.sum(0).cpu().tolist()
        root, limb = errors.sum(0).cpu().tolist()
        return dict(trajectory_head_attempts=ended,
            trajectory_head_run_completion_rate=reached/max(ended, 1),
            trajectory_head_all_boundary_contacts_rate=all_contacts/max(ended, 1),
            trajectory_head_time_progress=float(progress.sum())/max(ended, 1),
            trajectory_start_sample_fraction=heads/max(sampled, 1),
            trajectory_boundary_observations=observed,
            trajectory_boundary_contact_success_rate=contact/max(observed, 1),
            trajectory_boundary_unknown_count=unknown,
            trajectory_boundary_root_error_m=root/max(observed, 1),
            trajectory_boundary_limb_error_m=limb/max(observed, 1),
            trajectory_head_first_failures=failures)

    def state_dict(self):
        return dict(schema='php_continuous_trajectory_statistics_v1',
            motion_keys=self.motion_keys, segment_keys=self.segment_keys,
            **{name:getattr(self, name).cpu().clone() for name in ('counts', 'boundaries', 'errors', 'progress')})

    def load_state_dict(self, state):
        from somaforge_core.published_data import relocate_published_values
        motion_keys = relocate_published_values(state['motion_keys'])
        segment_keys = relocate_published_values(state['segment_keys'])
        segment_keys = [tuple(k) for k in segment_keys]
        if (state.get('schema') != 'php_continuous_trajectory_statistics_v1'
                or motion_keys != self.motion_keys or segment_keys != self.segment_keys):
            raise ValueError('Continuous trajectory statistics identities changed')
        for name in ('counts', 'boundaries', 'errors', 'progress'):
            value, target = state[name], getattr(self, name)
            if value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError('Invalid continuous statistics tensors')
        c, b = state['counts'], state['boundaries']
        if ((c[:, 1]>c[:, 0]).any() or (c[:, 2]>c[:, 1]).any() or (c[:, 3]>c[:, 2]).any()
                or (c[:, 4]>c[:, 3]).any() or (b[:, 1]+b[:, 2]>b[:, 0]).any()
                or (state['progress']>c[:, 2]+1e-5).any()):
            raise ValueError('Invalid continuous statistics counts')
        for name in ('counts', 'boundaries', 'errors', 'progress'): getattr(self, name).copy_(state[name])
        self.clear_interval()

    def write_csv(self, directory, iteration):
        rows = (
            ('trajectory_statistics.csv', ('motion_file', *self.motion_fields, 'time_progress_sum'),
             [(key, *counts, progress) for key, counts, progress in
              zip(self.motion_keys, self.counts.cpu().tolist(), self.progress.cpu().tolist())]),
            ('trajectory_boundaries.csv', ('motion_file','start_frame','end_frame', *self.boundary_fields,
                                           'root_error_sum_m','limb_error_sum_m'),
             [(*key, *counts, *errors) for key, counts, errors in
              zip(self.segment_keys, self.boundaries.cpu().tolist(), self.errors.cpu().tolist())]))
        for filename, header, values in rows:
            path = Path(directory)/filename
            new = not path.exists() or not path.stat().st_size
            with path.open('a', newline='') as stream:
                writer = csv.writer(stream)
                if new: writer.writerow(('iteration', *header))
                writer.writerows((iteration, *row) for row in values)
