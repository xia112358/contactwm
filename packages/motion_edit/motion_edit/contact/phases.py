from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ContactPhase:
    """One contiguous contact interval expressed in local phase coordinates."""

    part: str
    start_frame: int
    end_frame: int
    position_w: np.ndarray | None = None
    normal_w: np.ndarray | None = None
    contact_strength: np.ndarray | None = None
    force_envelope_w: np.ndarray | None = None
    surface_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        start = int(self.start_frame)
        end = int(self.end_frame)
        if end <= start:
            raise ValueError(f"{self.part}: contact phase end_frame must be greater than start_frame")
        count = end - start
        for name in ("position_w", "normal_w", "force_envelope_w"):
            values = getattr(self, name)
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != (count, 3):
                raise ValueError(f"{self.part}: {name} must have shape {(count, 3)}, got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.part}: {name} contains NaN or Inf")
        if self.contact_strength is not None:
            strength = np.asarray(self.contact_strength, dtype=np.float64)
            if strength.shape != (count,):
                raise ValueError(f"{self.part}: contact_strength must have shape {(count,)}, got {strength.shape}")
            if not np.all(np.isfinite(strength)) or np.any(strength < 0.0):
                raise ValueError(f"{self.part}: contact_strength must be finite and nonnegative")

    @property
    def frame_count(self) -> int:
        return int(self.end_frame) - int(self.start_frame)

    @property
    def frames(self) -> np.ndarray:
        return np.arange(int(self.start_frame), int(self.end_frame), dtype=np.int64)

    @property
    def local_phase(self) -> np.ndarray:
        return contact_local_phase(self.frame_count)

    def slice_signal(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        return arr[int(self.start_frame) : int(self.end_frame)].copy()


def contact_local_phase(count: int) -> np.ndarray:
    n = int(count)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    if n == 1:
        return np.asarray([0.5], dtype=np.float64)
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def contiguous_true_ranges(mask: Sequence[bool] | np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(arr):
        if bool(active) and start is None:
            start = int(index)
        elif not bool(active) and start is not None:
            ranges.append((start, int(index)))
            start = None
    if start is not None:
        ranges.append((start, int(len(arr))))
    return ranges


def phases_from_part_mask(
    mask: Sequence[bool] | np.ndarray,
    *,
    part: str,
    position_w: np.ndarray | None = None,
    normal_w: np.ndarray | None = None,
    contact_strength: np.ndarray | None = None,
    force_envelope_w: np.ndarray | None = None,
    surface_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[ContactPhase]:
    phases: list[ContactPhase] = []
    meta = dict(metadata or {})
    for phase_index, (start, end) in enumerate(contiguous_true_ranges(mask)):
        phase = ContactPhase(
            part=str(part),
            start_frame=int(start),
            end_frame=int(end),
            position_w=_slice_optional(position_w, start, end),
            normal_w=_slice_optional(normal_w, start, end),
            contact_strength=_slice_optional(contact_strength, start, end),
            force_envelope_w=_slice_optional(force_envelope_w, start, end),
            surface_id=surface_id,
            metadata={**meta, "phase_index": int(phase_index)},
        )
        phase.validate()
        phases.append(phase)
    return phases


def phases_from_mask(
    mask: np.ndarray,
    *,
    parts: Sequence[str],
    position_w: np.ndarray | None = None,
    normal_w: np.ndarray | None = None,
    contact_strength: np.ndarray | None = None,
    force_envelope_w: np.ndarray | None = None,
    surface_ids: Sequence[str | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, list[ContactPhase]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError(f"mask must have shape [T, P], got {arr.shape}")
    if arr.shape[1] < len(parts):
        raise ValueError(f"mask has {arr.shape[1]} parts but {len(parts)} names were provided")
    if surface_ids is not None and len(surface_ids) < len(parts):
        raise ValueError(f"surface_ids has {len(surface_ids)} entries but {len(parts)} parts were provided")
    out: dict[str, list[ContactPhase]] = {}
    for part_index, part in enumerate(parts):
        surface_id = None if surface_ids is None else surface_ids[part_index]
        out[str(part)] = phases_from_part_mask(
            arr[:, part_index],
            part=str(part),
            position_w=_part_signal(position_w, part_index),
            normal_w=_part_signal(normal_w, part_index),
            contact_strength=_part_signal(contact_strength, part_index),
            force_envelope_w=_part_signal(force_envelope_w, part_index),
            surface_id=surface_id,
            metadata=metadata,
        )
    return out


def match_source_phase(
    source_phases: Sequence[ContactPhase],
    *,
    target_phase: ContactPhase,
    target_phase_index: int,
    target_frame_count: int,
    source_frame_count: int,
) -> ContactPhase | None:
    if not source_phases:
        return None
    if int(target_phase_index) < len(source_phases):
        return source_phases[int(target_phase_index)]
    target_center = phase_center(target_phase, frame_count=target_frame_count)
    source_centers = np.asarray([phase_center(phase, frame_count=source_frame_count) for phase in source_phases], dtype=np.float64)
    return source_phases[int(np.argmin(np.abs(source_centers - target_center)))]


def phase_center(phase: ContactPhase, *, frame_count: int) -> float:
    denom = max(1, int(frame_count) - 1)
    return ((int(phase.start_frame) + int(phase.end_frame) - 1) * 0.5) / float(denom)


def resample_phase_values(values: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = int(count)
    if arr.ndim < 1:
        raise ValueError(f"values must be at least one-dimensional, got {arr.shape}")
    if n <= 0:
        return np.zeros((0, *arr.shape[1:]), dtype=np.float64)
    if arr.shape[0] == n:
        return arr.copy()
    if arr.shape[0] == 0:
        return np.zeros((n, *arr.shape[1:]), dtype=np.float64)
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    src_phase = contact_local_phase(arr.shape[0])
    dst_phase = contact_local_phase(n)
    flat = arr.reshape(arr.shape[0], -1)
    resampled = np.stack([np.interp(dst_phase, src_phase, flat[:, axis]) for axis in range(flat.shape[1])], axis=1)
    return resampled.reshape((n, *arr.shape[1:]))


def _slice_optional(values: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values)[int(start) : int(end)].copy()


def _part_signal(values: np.ndarray | None, part_index: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim < 2 or int(part_index) >= arr.shape[1]:
        return None
    return arr[:, int(part_index)].copy()
