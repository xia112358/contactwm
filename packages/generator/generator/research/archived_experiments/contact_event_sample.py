"""Preserved experiment helpers for regression tests; no training CLI."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ContactEventSample:
    motion_id: int
    source: str
    height: float
    current_frame: int
    target_frame: int
    x: np.ndarray
    touchdown: np.ndarray
    end_contact: np.ndarray
    surface_class: np.ndarray
    surface_uv: np.ndarray
    duration_s: float

def observed_action_vocabulary(samples: list[ContactEventSample]) -> np.ndarray:
    """Return every certified event action, including settled-pose waypoints.

    An all-zero touchdown mask is not a no-op: its end-contact, duration and
    pose-conditioned successor describe a posture phase boundary without a
    newly established contact.
    """
    masks = np.unique(np.asarray([sample.touchdown for sample in samples], dtype=bool), axis=0)
    if not len(masks):
        raise ValueError('observed action vocabulary is empty')
    return masks

def action_labels(samples: list[ContactEventSample], vocabulary: np.ndarray) -> np.ndarray:
    lookup = {tuple(mask.tolist()): index for (index, mask) in enumerate(vocabulary)}
    return np.asarray([lookup[tuple(sample.touchdown.astype(bool).tolist())] for sample in samples], dtype=np.int64)
