"""Head-start segment statistics; no sampling weights or physical snapshots."""
from __future__ import annotations

import csv
from pathlib import Path

import torch


FIELDS = ('sampled', 'head_sampled', 'head_finished', 'head_reached_end',
          'head_contact_success', 'head_unknown_end')


class SegmentStatistics:
    def __init__(self, keys, device):
        self.keys = tuple(tuple(key) for key in keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError('Segment statistics require unique motion/interval keys')
        self.counts = torch.zeros(len(keys), len(FIELDS), dtype=torch.long, device=device)
        self.progress = torch.zeros(len(keys), dtype=torch.float64, device=device)
        self.interval_counts = torch.zeros_like(self.counts)
        self.interval_progress = torch.zeros_like(self.progress)

    def _add(self, ids, counts, progress=None):
        for target in (self.counts, self.interval_counts):
            target.index_add_(0, ids, counts)
        if progress is not None:
            for target in (self.progress, self.interval_progress):
                target.index_add_(0, ids, progress.to(target))

    def sampled(self, ids, from_head):
        counts = self.counts.new_zeros(len(ids), len(FIELDS))
        counts[:, 0] = 1
        counts[:, 1] = from_head.long()
        self._add(ids, counts)

    def finished(self, ids, from_head, reached_end, contact_success, unknown_end, progress):
        # RSI is useful training data, but never a completion-rate denominator.
        ids, reached_end, contact_success, unknown_end, progress = (
            value[from_head] for value in (ids, reached_end, contact_success, unknown_end, progress))
        counts = self.counts.new_zeros(len(ids), len(FIELDS))
        counts[:, 2] = 1
        counts[:, 3] = reached_end.long()
        counts[:, 4] = (reached_end & contact_success & ~unknown_end).long()
        counts[:, 5] = (reached_end & unknown_end).long()
        self._add(ids, counts, progress.clamp(0, 1))

    def metrics(self, *, cumulative=False):
        counts = self.counts if cumulative else self.interval_counts
        progress = self.progress if cumulative else self.interval_progress
        sampled, heads, finished, reached, success, unknown = counts.sum(0).cpu().tolist()
        return dict(segment_head_attempts=finished,
            segment_head_mean_progress=float(progress.sum())/max(finished, 1),
            segment_head_completion_rate=reached/max(finished, 1),
            segment_head_contact_success_rate=success/max(finished, 1),
            segment_head_unknown_end_count=unknown,
            segment_head_samples=heads, segment_samples=sampled,
            segment_start_sample_fraction=heads/max(sampled, 1))

    def clear_interval(self):
        self.interval_counts.zero_()
        self.interval_progress.zero_()

    def state_dict(self):
        return dict(schema='php_segment_head_statistics_v1', keys=self.keys,
                    counts=self.counts.cpu().clone(), progress=self.progress.cpu().clone())

    def load_state_dict(self, state):
        if state.get('schema') != 'php_segment_head_statistics_v1':
            raise ValueError('Unknown segment statistics checkpoint schema')
        from somaforge_core.published_data import relocate_published_values
        keys = [(relocate_published_values(key[0]), *key[1:]) for key in state['keys']]
        counts, progress = state['counts'], state['progress']
        if (len(set(keys)) != len(keys) or counts.shape != (len(keys), len(FIELDS))
                or counts.dtype != torch.long or progress.shape != (len(keys),)
                or not torch.isfinite(progress).all() or (counts < 0).any()
                or (progress < 0).any() or (progress > counts[:, 2]+1e-5).any()
                or (counts[:, 1] > counts[:, 0]).any() or (counts[:, 2] > counts[:, 1]).any()
                or (counts[:, 3] > counts[:, 2]).any()
                or (counts[:, 4]+counts[:, 5] > counts[:, 3]).any()):
            raise ValueError('Invalid segment statistics checkpoint')
        if set(keys) != set(self.keys):
            raise ValueError('Segment statistics checkpoint motions or intervals changed')
        old = {key: i for i, key in enumerate(keys)}
        pairs = [(i, old[key]) for i, key in enumerate(self.keys) if key in old]
        self.counts.zero_()
        self.progress.zero_()
        if pairs:
            current, saved = zip(*pairs)
            self.counts[list(current)] = counts[list(saved)].to(self.counts)
            self.progress[list(current)] = progress[list(saved)].to(self.progress)
        self.clear_interval()

    def write_csv(self, directory, iteration):
        path = Path(directory)/'segment_head_statistics.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        counts, progress = self.counts.cpu().tolist(), self.progress.cpu().tolist()
        new = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='') as stream:
            writer = csv.writer(stream)
            if new:
                writer.writerow(('iteration', 'motion_file', 'start_frame', 'end_frame', *FIELDS,
                                 'head_mean_progress', 'head_completion_rate', 'head_contact_success_rate'))
            for key, row, value in zip(self.keys, counts, progress, strict=True):
                n = row[2]
                rates = (value/n, row[3]/n, row[4]/n) if n else ('', '', '')
                writer.writerow((iteration, *key, *row, *rates))
        return path
