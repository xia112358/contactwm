"""Preserved experiment helpers for regression tests; no training CLI."""
import numpy as np

class OrderedContactTeacher:

    def __init__(self, events, start_frame=0):
        self.events = sorted(events, key=lambda e: e['target_frame'])
        for e in self.events:
            if not 0 <= e['current_frame'] < e['target_frame']:
                raise ValueError('Invalid transition')
        self.event_index = next((i for (i, e) in enumerate(self.events) if e['target_frame'] > start_frame), len(self.events))

    def align(self, q, active, surface):
        if self.event_index >= len(self.events):
            return (None, dict(reason='demonstration_finished', completed_events=self.event_index))
        e = self.events[self.event_index]
        return (e['sample'], dict(reason='ordered_target', completed_events=self.event_index, sample=e['sample'], target_frame=e['target_frame'], transition_start_frame=e['current_frame'], duration_frames=e['target_frame'] - e['current_frame']))

    def advance(self, previous_q, q, actual_active, actual_surface, *, terminal):
        if self.event_index >= len(self.events) or terminal:
            return False
        if not np.isfinite(q).all():
            return False
        e = self.events[self.event_index]
        wanted = np.asarray(e['active'], bool)
        if np.array_equal(previous_q, q):
            return False
        if not np.array_equal(actual_active, wanted):
            return False
        if not np.array_equal(np.asarray(actual_surface)[wanted], np.asarray(e['surface'])[wanted]):
            return False
        self.event_index += 1
        return True
