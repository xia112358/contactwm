from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from somaforge_core.contact_schema import CONTACT_FORCE_PART_NAMES, CONTACT_FORCE_PART_ORDER

DEFAULT_CONTACT_FORCE_PART_ORDER = CONTACT_FORCE_PART_NAMES

WBT_8PART_CONTACT_FORCE_PART_ORDER = CONTACT_FORCE_PART_ORDER

WBT_8PART_CANONICAL_TO_SHORT = {
    "left_heel": "LHEE",
    "left_toe": "LTOE",
    "right_heel": "RHEE",
    "right_toe": "RTOE",
    "left_hand": "LH",
    "right_hand": "RH",
    "left_knee": "LK",
    "right_knee": "RK",
    "LHEE": "LHEE",
    "LTOE": "LTOE",
    "RHEE": "RHEE",
    "RTOE": "RTOE",
    "LH": "LH",
    "RH": "RH",
    "LK": "LK",
    "RK": "RK",
}

@dataclass(frozen=True)
class CanonicalContactForceField:
    """Part-level force reference used by force-aware tracking policies."""

    part_order: tuple[str, ...]
    force_w: np.ndarray
    position_w: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        force = np.asarray(self.force_w, dtype=np.float64)
        position = np.asarray(self.position_w, dtype=np.float64)
        mask = np.asarray(self.mask, dtype=bool)
        if force.ndim != 3 or force.shape[2] != 3:
            raise ValueError(f"force_w must have shape [T, P, 3], got {force.shape}")
        if position.shape != force.shape:
            raise ValueError(f"position_w must have shape {force.shape}, got {position.shape}")
        if mask.shape != force.shape[:2]:
            raise ValueError(f"mask must have shape {force.shape[:2]}, got {mask.shape}")
        if len(self.part_order) != force.shape[1]:
            raise ValueError(f"part_order length {len(self.part_order)} does not match force width {force.shape[1]}")
        if not np.all(np.isfinite(force)):
            raise ValueError("force_w contains NaN or Inf")
        finite_positions = np.isfinite(position) | ~mask[..., None]
        if not bool(np.all(finite_positions)):
            raise ValueError("position_w contains NaN or Inf for active contact frames")

    def to_npz_arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "contact_force_part_order": np.asarray(self.part_order, dtype=np.str_),
            "contact_force_part_w": np.asarray(self.force_w, dtype=np.float64),
            "contact_force_part_force_w": np.asarray(self.force_w, dtype=np.float64),
            "contact_force_part_position_w": np.asarray(self.position_w, dtype=np.float64),
            "contact_force_part_mask": np.asarray(self.mask, dtype=bool),
        }

    def to_wbt_8part_npz_arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        source_index = {
            WBT_8PART_CANONICAL_TO_SHORT.get(str(part), str(part)): index
            for index, part in enumerate(self.part_order)
        }
        force = np.zeros((self.force_w.shape[0], len(WBT_8PART_CONTACT_FORCE_PART_ORDER), 3), dtype=np.float64)
        position = np.zeros_like(force)
        mask = np.zeros((self.force_w.shape[0], len(WBT_8PART_CONTACT_FORCE_PART_ORDER)), dtype=bool)
        for dst_i, part in enumerate(WBT_8PART_CONTACT_FORCE_PART_ORDER):
            src_i = source_index.get(part)
            if src_i is None:
                continue
            force[:, dst_i] = np.asarray(self.force_w[:, src_i], dtype=np.float64)
            position[:, dst_i] = np.asarray(self.position_w[:, src_i], dtype=np.float64)
            mask[:, dst_i] = np.asarray(self.mask[:, src_i], dtype=bool)
        return {
            "contact_force_part_order": np.asarray(WBT_8PART_CONTACT_FORCE_PART_ORDER, dtype=np.str_),
            "contact_force_part_w": force,
            "contact_force_part_force_w": force,
            "contact_force_part_position_w": position,
            "contact_force_part_mask": mask,
        }
