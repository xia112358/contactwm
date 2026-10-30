"""Hindsight contact-onset labels without rewriting native contact evidence."""
from __future__ import annotations

import numpy as np

PARTS = ("left_foot", "right_foot", "left_knee", "right_knee", "left_hand", "right_hand",
         "left_elbow", "right_elbow", "pelvis")
BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link",
              "left_wrist_yaw_link", "right_wrist_yaw_link", "left_elbow_link", "right_elbow_link", "pelvis")
REFERENCE_SCHEMA = "somaforge_acp_nine_body_native_reference_v1"


def hindsight_labels(active, force_w, positions_w, known, *, horizon=75, min_duration=10, force_threshold=1.0):
    """Next strictly future onset, bounded to this episode; unknown is not negative.

    Force/duration select supervision events among native activated constraints.
    The caller's activation, known masks and force arrays remain immutable.
    A segment already active at the recorder boundary is left-censored and
    cannot establish an onset. Missing future observations mask occupancy loss.
    """
    active, known = np.asarray(active, bool), np.asarray(known, bool)
    force_w, positions_w = np.asarray(force_w), np.asarray(positions_w)
    if active.ndim != 2 or active.shape[1] != 9 or known.shape != active.shape:
        raise ValueError("Nine native contact channels and explicit known masks required")
    if force_w.shape != (*active.shape, 3) or positions_w.shape != force_w.shape:
        raise ValueError("Contact force/position arrays have incompatible shapes")
    if horizon < 1 or min_duration < 1 or not np.isfinite(force_w).all() or not np.isfinite(positions_w).all():
        raise ValueError("Invalid hindsight inputs")
    size = len(active)
    targets = np.zeros((*active.shape, 3), np.float32)
    occupancy, label_known = np.zeros_like(active), np.zeros_like(active)
    onset_index = np.full(active.shape, -1, np.int64)
    for part in range(9):
        loaded = active[:, part] & known[:, part] & (np.abs(force_w[:, part, 2]) > force_threshold)
        changes = np.diff(np.r_[False, loaded, False].astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        onsets = [int(start) for start, end in zip(starts, ends)
                  if start > 0 and known[start - 1, part] and end - start >= min_duration]
        unknown = np.r_[0, np.cumsum(~known[:, part])]
        for frame in range(size):
            # Certifying a negative also requires observing the minimum hold
            # time of an onset at the far end of the horizon.
            end = min(size, frame + horizon + min_duration)
            upcoming = next((onset for onset in onsets if frame < onset <= frame + horizon), None)
            if upcoming is not None:
                if unknown[upcoming + min_duration] == unknown[frame]:
                    targets[frame, part] = positions_w[upcoming, part]
                    occupancy[frame, part] = label_known[frame, part] = True
                    onset_index[frame, part] = upcoming
            elif end == frame + horizon + min_duration and unknown[end] == unknown[frame]:
                label_known[frame, part] = True
    return {"target_w": targets, "occupancy": occupancy, "known": label_known, "onset_index": onset_index}
