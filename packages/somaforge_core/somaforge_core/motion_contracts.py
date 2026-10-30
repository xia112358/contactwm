"""Data contracts shared by the selector and closed-loop runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BODY_NAMES = (
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_knee_link",
    "right_knee_link",
)

CONTACT_PARTS = ("LF", "RF", "LH", "RH", "LK", "RK")
CONTACT_BODY_INDEX = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class SparseKeyframe:
    """Complete 63D selector boundary in the current torso-yaw frame."""

    position: NDArray[np.float32]
    rotation6d: NDArray[np.float32]
    contact: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.position.shape != (len(BODY_NAMES), 3):
            raise ValueError(f"expected position {(len(BODY_NAMES), 3)}, got {self.position.shape}")
        if self.rotation6d.shape != (len(BODY_NAMES), 6):
            raise ValueError(f"expected rotation6d {(len(BODY_NAMES), 6)}, got {self.rotation6d.shape}")
        if self.contact.shape != (len(CONTACT_PARTS),):
            raise ValueError(f"expected contact {(len(CONTACT_PARTS),)}, got {self.contact.shape}")

    @property
    def vector63(self) -> NDArray[np.float32]:
        return np.concatenate((self.position.reshape(-1), self.rotation6d.reshape(-1))).astype(np.float32)


@dataclass(frozen=True)
class InteractionBoundary:
    """One indivisible pose/contact interaction at an event boundary.

    Every contact-capable body always has a pose.  ``active_contact`` says
    which subset is constrained to its contact point/surface at this boundary;
    it does not remove inactive bodies from the state.
    """

    position: NDArray[np.float32]
    rotation6d: NDArray[np.float32]
    active_contact: NDArray[np.bool_]
    touchdown: NDArray[np.bool_]
    persistent_support: NDArray[np.bool_]
    contact_position: NDArray[np.float32]
    contact_surface: NDArray[np.int64]
    duration_s: float

    def __post_init__(self) -> None:
        expected_bodies = len(BODY_NAMES)
        expected_contacts = len(CONTACT_PARTS)
        if self.position.shape != (expected_bodies, 3):
            raise ValueError(f"expected position {(expected_bodies, 3)}, got {self.position.shape}")
        if self.rotation6d.shape != (expected_bodies, 6):
            raise ValueError(f"expected rotation6d {(expected_bodies, 6)}, got {self.rotation6d.shape}")
        if self.active_contact.shape != (expected_contacts,):
            raise ValueError(f"expected active_contact {(expected_contacts,)}, got {self.active_contact.shape}")
        if self.touchdown.shape != (expected_contacts,):
            raise ValueError(f"expected touchdown {(expected_contacts,)}, got {self.touchdown.shape}")
        if np.any(self.touchdown & ~self.active_contact):
            raise ValueError("touchdown must be a subset of active_contact")
        if self.persistent_support.shape != (expected_contacts,):
            raise ValueError(f"expected persistent_support {(expected_contacts,)}, got {self.persistent_support.shape}")
        if np.any(self.persistent_support & ~self.active_contact):
            raise ValueError("persistent_support must be a subset of active_contact")
        if self.contact_position.shape != (expected_contacts, 3):
            raise ValueError(f"expected contact_position {(expected_contacts, 3)}, got {self.contact_position.shape}")
        if self.contact_surface.shape != (expected_contacts,):
            raise ValueError(f"expected contact_surface {(expected_contacts,)}, got {self.contact_surface.shape}")
        if not np.isfinite(self.position).all() or not np.isfinite(self.rotation6d).all():
            raise ValueError("interaction pose must be finite")
        if not np.isfinite(self.contact_position).all():
            raise ValueError("interaction contact positions must be finite")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("interaction duration_s must be finite and positive")
        if np.any(self.contact_surface[self.active_contact] < 0):
            raise ValueError("active contacts require a non-negative surface index")

    @property
    def contact(self) -> NDArray[np.bool_]:
        """Compatibility spelling for consumers that call the active mask contact."""

        return self.active_contact

    def persistent_from(self, current: InteractionBoundary) -> NDArray[np.bool_]:
        return self.persistent_support & current.active_contact

    def touchdown_from(self, current: InteractionBoundary) -> NDArray[np.bool_]:
        return self.active_contact & ~self.persistent_from(current)

    def liftoff_from(self, current: InteractionBoundary) -> NDArray[np.bool_]:
        return current.active_contact & ~self.persistent_from(current)

    def swing_from(self, current: InteractionBoundary) -> NDArray[np.bool_]:
        return ~current.active_contact & ~self.active_contact


@dataclass(frozen=True)
class TeacherSegment:
    """One generated-teacher transition and its aligned event supervision."""

    trajectory_id: int
    event_index: int
    action: int
    fps: float
    position: NDArray[np.float32]
    quaternion: NDArray[np.float32]
    contact: NDArray[np.bool_]
    start: SparseKeyframe
    end: SparseKeyframe
    touchdown: NDArray[np.bool_]
    target_contact: NDArray[np.float32]
    target_surface: NDArray[np.int64]
    duration: float
    geometry: NDArray[np.float32]
