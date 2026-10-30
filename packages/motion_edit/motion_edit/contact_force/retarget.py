from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from somaforge_core.contact_schema import CONTACT_FORCE_PART_BODY_NAMES, canonical_contact_part_name

from motion_edit.contact.phases import (
    ContactPhase,
    contact_local_phase,
    contiguous_true_ranges,
    match_source_phase,
    phases_from_mask,
    resample_phase_values,
)

from .schema import DEFAULT_CONTACT_FORCE_PART_ORDER, CanonicalContactForceField

_PART_ALIASES = {
    "lhee": "left_heel",
    "ltoe": "left_toe",
    "rhee": "right_heel",
    "rtoe": "right_toe",
    "lh": "left_hand",
    "rh": "right_hand",
    "lk": "left_knee",
    "rk": "right_knee",
    "left_heel": "left_heel",
    "left_toe": "left_toe",
    "right_heel": "right_heel",
    "right_toe": "right_toe",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
}

_PART_BODY_CANDIDATES = {
    canonical_contact_part_name(part_id): body_names
    for part_id, body_names in CONTACT_FORCE_PART_BODY_NAMES.items()
}


@dataclass(frozen=True)
class RetargetContactForceConfig:
    part_order: tuple[str, ...] = DEFAULT_CONTACT_FORCE_PART_ORDER
    source_normal_w: Sequence[float] | np.ndarray | None = None
    target_normal_w: Sequence[float] | np.ndarray | None = None
    force_unit_scale: float = 1.0
    max_force_norm: float | None = 5000.0
    smoothing_window: int = 3
    force_norm_eps: float = 1.0e-8
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.force_unit_scale) < 0.0 or not np.isfinite(float(self.force_unit_scale)):
            raise ValueError("force_unit_scale must be finite and nonnegative")
        if self.max_force_norm is not None and (float(self.max_force_norm) <= 0.0 or not np.isfinite(float(self.max_force_norm))):
            raise ValueError("max_force_norm must be positive when provided")
        if int(self.smoothing_window) < 1:
            raise ValueError("smoothing_window must be >= 1")
        if float(self.force_norm_eps) < 0.0 or not np.isfinite(float(self.force_norm_eps)):
            raise ValueError("force_norm_eps must be finite and nonnegative")


def retarget_contact_forces(
    *,
    target_motion: Mapping[str, Any],
    source_force_ref: Mapping[str, Any],
    config: RetargetContactForceConfig | None = None,
) -> CanonicalContactForceField:
    cfg = config or RetargetContactForceConfig()
    parts = tuple(_canonical_part_name(part) or str(part) for part in cfg.part_order)
    n_frames = _motion_frame_count(target_motion)
    source_force, source_mask, _source_position, source_meta = _aligned_force_arrays(
        source_force_ref,
        parts=parts,
        frame_count=None,
        force_required=True,
        force_norm_eps=float(cfg.force_norm_eps),
    )
    target_mask, target_mask_meta = _target_mask(
        target_motion=target_motion,
        source_mask=source_mask,
        parts=parts,
        n_frames=n_frames,
    )
    target_position, target_position_meta = _target_positions(target_motion, parts=parts, n_frames=n_frames)

    source_normal_input = _prepare_normal_input(cfg.source_normal_w)
    target_normal_input = _prepare_normal_input(cfg.target_normal_w)
    source_normal_source = str(cfg.metadata.get("source_normal_source") or ("provided" if cfg.source_normal_w is not None else "default_up"))
    target_normal_source = str(cfg.metadata.get("target_normal_source") or ("provided" if cfg.target_normal_w is not None else "default_up"))

    out_force = np.zeros((n_frames, len(parts), 3), dtype=np.float64)
    source_phases_by_part = phases_from_mask(
        source_mask,
        parts=parts,
        force_envelope_w=source_force,
        metadata={"phase_source": "source_force_ref_contact_mask"},
    )
    target_phases_by_part = phases_from_mask(
        target_mask,
        parts=parts,
        position_w=target_position,
        metadata={"phase_source": "target_motion_contact_mask"},
    )
    phase_counts: dict[str, int] = {}
    matched_phase_count = 0
    unmatched_target_phase_count = 0
    for part_index, part in enumerate(parts):
        source_phases = source_phases_by_part[str(part)]
        target_phases = target_phases_by_part[str(part)]
        phase_counts[str(part)] = int(len(target_phases))
        for target_phase_index, target_phase in enumerate(target_phases):
            source_phase = match_source_phase(
                source_phases,
                target_phase=target_phase,
                target_phase_index=target_phase_index,
                target_frame_count=n_frames,
                source_frame_count=source_force.shape[0],
            )
            if source_phase is None:
                unmatched_target_phase_count += 1
                continue
            matched_phase_count += 1
            src_start, src_end = int(source_phase.start_frame), int(source_phase.end_frame)
            dst_start, dst_end = int(target_phase.start_frame), int(target_phase.end_frame)
            src_force = _phase_force(source_phase, fallback_force=source_force[:, part_index])
            source_normal = _phase_normal(source_normal_input, start=src_start, end=src_end, part_index=part_index)
            target_normal = _phase_normal(target_normal_input, start=dst_start, end=dst_end, part_index=part_index)
            transformed = _retarget_force_vectors(src_force, source_normal=source_normal, target_normal=target_normal)
            transformed = _resample_phase(transformed, dst_end - dst_start)
            transformed = _smooth_phase(transformed, int(cfg.smoothing_window))
            out_force[dst_start:dst_end, part_index] = transformed * float(cfg.force_unit_scale)

    out_force = _clip_force_norm(out_force, max_force_norm=cfg.max_force_norm)
    out_force[~target_mask] = 0.0
    metadata = {
        **dict(cfg.metadata),
        "force_source": "retargeted_contact_force",
        "state_policy": "contact_phase_force_retarget",
        "force_applied_to_body": False,
        "integrated": False,
        "retarget_method": "contact_phase_local_force_resampling",
        "part_order": list(parts),
        "source_phase_count": int(sum(len(source_phases_by_part[str(part)]) for part in parts)),
        "retarget_phase_count": int(sum(phase_counts.values())),
        "retarget_phase_count_by_part": phase_counts,
        "matched_phase_count": int(matched_phase_count),
        "unmatched_target_phase_count": int(unmatched_target_phase_count),
        "target_contact_frame_count": int(np.count_nonzero(target_mask)),
        "source_contact_frame_count": int(np.count_nonzero(source_mask)),
        "force_unit_scale": float(cfg.force_unit_scale),
        "max_force_norm": None if cfg.max_force_norm is None else float(cfg.max_force_norm),
        "force_norm_max": float(np.linalg.norm(out_force, axis=2).max(initial=0.0)),
        "normal_source": f"source:{source_normal_source};target:{target_normal_source}",
        "source_normal_source": source_normal_source,
        "target_normal_source": target_normal_source,
        "source_force": source_meta,
        "target_mask": target_mask_meta,
        "target_position": target_position_meta,
    }
    field = CanonicalContactForceField(
        part_order=parts,
        force_w=out_force,
        position_w=target_position,
        mask=target_mask,
        metadata=metadata,
    )
    field.validate()
    return field


def _aligned_force_arrays(
    motion: Mapping[str, Any],
    *,
    parts: tuple[str, ...],
    frame_count: int | None,
    force_required: bool,
    force_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    force_key = "contact_force_part_w" if "contact_force_part_w" in motion else "contact_force_part_force_w"
    if force_key not in motion:
        if force_required:
            raise KeyError("source force reference missing contact_force_part_w/contact_force_part_force_w")
        n = int(frame_count or 0)
        return np.zeros((n, len(parts), 3), dtype=np.float64), np.zeros((n, len(parts)), dtype=bool), None, {"force_key": "none"}
    raw_force = np.asarray(motion[force_key], dtype=np.float64)
    if raw_force.ndim != 3 or raw_force.shape[2] != 3:
        raise ValueError(f"{force_key} must have shape [T, P, 3], got {raw_force.shape}")
    n = raw_force.shape[0] if frame_count is None else int(frame_count)
    indices, order_meta = _part_indices(motion, width=raw_force.shape[1], parts=parts)
    force = np.zeros((n, len(parts), 3), dtype=np.float64)
    source_count = min(n, raw_force.shape[0])
    for dst, src in enumerate(indices):
        if src is not None and src < raw_force.shape[1]:
            force[:source_count, dst] = raw_force[:source_count, src]
    mask = _aligned_mask(motion, parts=parts, indices=indices, n_frames=n, fallback_force=force, force_norm_eps=force_norm_eps)
    position = _aligned_position_array(motion, parts=parts, indices=indices, n_frames=n)
    return force, mask, position, {"force_key": force_key, **order_meta}


def _target_mask(
    *,
    target_motion: Mapping[str, Any],
    source_mask: np.ndarray,
    parts: tuple[str, ...],
    n_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_key = _first_key(target_motion, ("contact_force_part_mask", "contact_part_mask"))
    if mask_key is not None:
        raw = np.asarray(target_motion[mask_key], dtype=bool)
        if raw.ndim == 2:
            indices, order_meta = _part_indices(target_motion, width=raw.shape[1], parts=parts)
            mask = np.zeros((n_frames, len(parts)), dtype=bool)
            count = min(n_frames, raw.shape[0])
            for dst, src in enumerate(indices):
                if src is not None and src < raw.shape[1]:
                    mask[:count, dst] = raw[:count, src]
            return mask, {"source": mask_key, **order_meta}
    if source_mask.shape[0] == n_frames:
        return source_mask.copy(), {"source": "source_mask_same_length_fallback"}
    resampled = np.zeros((n_frames, source_mask.shape[1]), dtype=bool)
    source_phase = _local_phase(source_mask.shape[0])
    target_phase = _local_phase(n_frames)
    for part_index in range(source_mask.shape[1]):
        values = np.interp(target_phase, source_phase, source_mask[:, part_index].astype(np.float64))
        resampled[:, part_index] = values >= 0.5
    return resampled, {"source": "source_mask_global_phase_fallback"}


def _target_positions(motion: Mapping[str, Any], *, parts: tuple[str, ...], n_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    if "contact_force_part_position_w" in motion:
        raw = np.asarray(motion["contact_force_part_position_w"])
        width = int(raw.shape[1]) if raw.ndim >= 2 else None
        indices, order_meta = _part_indices(motion, width=width, parts=parts)
        position = _aligned_position_array(motion, parts=parts, indices=indices, n_frames=n_frames)
        if position is not None:
            return _finite_or_zero(position), {"source": "contact_force_part_position_w", **order_meta}
    body_position = _positions_from_body_arrays(motion, parts=parts, n_frames=n_frames)
    if body_position is not None:
        return _finite_or_zero(body_position), {"source": "body_pos_w_body_names"}
    return np.zeros((n_frames, len(parts), 3), dtype=np.float64), {"source": "zeros_no_position_available"}


def _aligned_mask(
    motion: Mapping[str, Any],
    *,
    parts: tuple[str, ...],
    indices: Sequence[int | None],
    n_frames: int,
    fallback_force: np.ndarray,
    force_norm_eps: float,
) -> np.ndarray:
    mask_key = _first_key(motion, ("contact_force_part_mask", "contact_part_mask"))
    if mask_key is not None:
        raw = np.asarray(motion[mask_key], dtype=bool)
        if raw.ndim == 2:
            out = np.zeros((n_frames, len(parts)), dtype=bool)
            count = min(n_frames, raw.shape[0])
            for dst, src in enumerate(indices):
                if src is not None and src < raw.shape[1]:
                    out[:count, dst] = raw[:count, src]
            return out
    return np.linalg.norm(fallback_force, axis=2) > float(force_norm_eps)


def _aligned_position_array(
    motion: Mapping[str, Any],
    *,
    parts: tuple[str, ...],
    indices: Sequence[int | None],
    n_frames: int,
) -> np.ndarray | None:
    if "contact_force_part_position_w" not in motion:
        return None
    raw = np.asarray(motion["contact_force_part_position_w"], dtype=np.float64)
    if raw.ndim != 3 or raw.shape[2] != 3:
        return None
    out = np.zeros((n_frames, len(parts), 3), dtype=np.float64)
    count = min(n_frames, raw.shape[0])
    any_part = False
    for dst, src in enumerate(indices):
        if src is not None and src < raw.shape[1]:
            out[:count, dst] = raw[:count, src]
            any_part = True
    return out if any_part else None


def _positions_from_body_arrays(motion: Mapping[str, Any], *, parts: tuple[str, ...], n_frames: int) -> np.ndarray | None:
    if "body_pos_w" not in motion:
        return None
    body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        return None
    body_names = _motion_strings(motion, ("body_names", "body_name", "body_pos_w_names", "body_pos_names"))
    if not body_names:
        return body_pos[:n_frames, : len(parts), :3] if body_pos.shape[1] >= len(parts) else None
    name_to_index = {str(name): index for index, name in enumerate(body_names)}
    out = np.zeros((n_frames, len(parts), 3), dtype=np.float64)
    any_part = False
    count = min(n_frames, body_pos.shape[0])
    for part_index, part in enumerate(parts):
        indices = [name_to_index[name] for name in _PART_BODY_CANDIDATES.get(part, (part,)) if name in name_to_index]
        if not indices:
            continue
        out[:count, part_index] = np.mean(body_pos[:count, indices, :3], axis=1)
        any_part = True
    return out if any_part else None


def _part_indices(motion: Mapping[str, Any], *, width: int | None, parts: tuple[str, ...]) -> tuple[list[int | None], dict[str, Any]]:
    raw_order = _motion_strings(
        motion,
        ("contact_force_part_order", "contact_part_order", "part_order", "contact_part_names", "contact_force_part_names"),
    )
    if not raw_order:
        if width is None:
            return [None for _ in parts], {"part_order_source": "none"}
        return [index if index < int(width) else None for index in range(len(parts))], {"part_order_source": "default_width_order"}
    canonical_order = [_canonical_part_name(raw) for raw in raw_order]
    indices: list[int | None] = []
    for part in parts:
        try:
            indices.append(canonical_order.index(part))
        except ValueError:
            indices.append(None)
    return indices, {"part_order_source": "motion", "raw_part_order": raw_order}


def _retarget_force_vectors(force_w: np.ndarray, *, source_normal: np.ndarray, target_normal: np.ndarray) -> np.ndarray:
    normal_mag = force_w @ source_normal
    source_tangent = force_w - normal_mag[:, None] * source_normal[None, :]
    target_tangent = source_tangent - (source_tangent @ target_normal)[:, None] * target_normal[None, :]
    src_tangent_norm = np.linalg.norm(source_tangent, axis=1)
    dst_tangent_norm = np.linalg.norm(target_tangent, axis=1)
    scale = np.divide(src_tangent_norm, dst_tangent_norm, out=np.zeros_like(src_tangent_norm), where=dst_tangent_norm > 1.0e-12)
    target_tangent = target_tangent * scale[:, None]
    return np.maximum(0.0, normal_mag)[:, None] * target_normal[None, :] + target_tangent


def _phase_force(phase: ContactPhase, *, fallback_force: np.ndarray) -> np.ndarray:
    if phase.force_envelope_w is not None:
        return np.asarray(phase.force_envelope_w, dtype=np.float64)
    return np.asarray(fallback_force, dtype=np.float64)[int(phase.start_frame) : int(phase.end_frame)].copy()


def _resample_phase(values: np.ndarray, count: int) -> np.ndarray:
    return resample_phase_values(values, count)


def _smooth_phase(values: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w <= 1 or values.shape[0] < 3:
        return values
    if w % 2 == 0:
        w += 1
    w = min(w, values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1)
    if w <= 1:
        return values
    kernel = np.ones(w, dtype=np.float64) / float(w)
    pad = w // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])], axis=1)


def _clip_force_norm(force: np.ndarray, *, max_force_norm: float | None) -> np.ndarray:
    if max_force_norm is None:
        return force
    norm = np.linalg.norm(force, axis=2)
    scale = np.minimum(1.0, float(max_force_norm) / np.maximum(norm, 1.0e-12))
    return force * scale[..., None]


def _match_source_phase(
    source_phases: list[tuple[int, int]],
    *,
    target_range: tuple[int, int],
    target_phase_index: int,
    target_frame_count: int,
    source_frame_count: int,
) -> tuple[int, int] | None:
    source_phase_objs = [ContactPhase(part="unknown", start_frame=start, end_frame=end) for start, end in source_phases]
    target_phase = ContactPhase(part="unknown", start_frame=target_range[0], end_frame=target_range[1])
    matched = match_source_phase(
        source_phase_objs,
        target_phase=target_phase,
        target_phase_index=target_phase_index,
        target_frame_count=target_frame_count,
        source_frame_count=source_frame_count,
    )
    if matched is None:
        return None
    return int(matched.start_frame), int(matched.end_frame)


def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    return contiguous_true_ranges(mask)


def _motion_frame_count(motion: Mapping[str, Any]) -> int:
    for key in ("joint_pos", "body_pos_w", "contact_force_part_mask", "contact_part_mask", "contact_force_part_w"):
        if key in motion:
            arr = np.asarray(motion[key])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise ValueError("target motion has no frame-indexed arrays")


def _motion_strings(motion: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        if key not in motion:
            continue
        arr = np.asarray(motion[key], dtype=object)
        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, str):
                return [value.strip() for value in item.split(",") if value.strip()]
            if isinstance(item, (list, tuple)):
                return [str(value) for value in item]
            return [str(item)]
        return [str(value) for value in arr.reshape(-1).tolist()]
    return []


def _first_key(motion: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in motion:
            return key
    return None


def _canonical_part_name(name: str) -> str | None:
    key = str(name).strip().lower()
    if key in _PART_ALIASES:
        return _PART_ALIASES[key]
    for token, canonical in _PART_ALIASES.items():
        if token and token in key:
            return canonical
    return None


def _prepare_normal_input(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    normals = np.asarray(value, dtype=np.float64)
    if normals.shape == (3,):
        return _unit_normal(normals)
    if normals.ndim == 3 and normals.shape[2] == 3:
        return normals
    raise ValueError(f"normal input must have shape (3,) or [T, P, 3], got {normals.shape}")


def _phase_normal(normal_input: np.ndarray | None, *, start: int, end: int, part_index: int) -> np.ndarray:
    if normal_input is None:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normals = np.asarray(normal_input, dtype=np.float64)
    if normals.shape == (3,):
        return normals
    frame_start = max(0, min(normals.shape[0], int(start)))
    frame_end = max(frame_start, min(normals.shape[0], int(end)))
    if frame_end <= frame_start or int(part_index) >= normals.shape[1]:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    segment = normals[frame_start:frame_end, int(part_index)]
    finite = np.all(np.isfinite(segment), axis=1) & (np.linalg.norm(segment, axis=1) > 1.0e-12)
    if not np.any(finite):
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return _unit_normal(np.mean(segment[finite], axis=0))


def _unit_normal(value: Sequence[float] | np.ndarray) -> np.ndarray:
    normal = np.asarray(value, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError(f"normal must have shape (3,), got {normal.shape}")
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise ValueError("normal must be nonzero")
    return normal / norm


def _local_phase(count: int) -> np.ndarray:
    return contact_local_phase(count)


def _finite_or_zero(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 0.0
    return out
