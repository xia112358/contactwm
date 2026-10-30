from __future__ import annotations

from typing import Iterable

import numpy as np

from motion_edit.contact.events import body_names_for_mask
from motion_edit.contact.schema import ContactAnchorRecord, ContactAnchorRole


def _as_2d_bool(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim != 2:
        raise ValueError(f"contact mask must be 1D or 2D, got shape={arr.shape}")
    return arr


def _role_for_interval(
    *,
    body_index: int,
    start: int,
    end: int,
    active_mask: np.ndarray | None,
    support_mask: np.ndarray | None,
) -> ContactAnchorRole:
    active_any = bool(active_mask is not None and np.any(active_mask[start:end, body_index]))
    support_any = bool(support_mask is not None and np.any(support_mask[start:end, body_index]))
    if active_any and support_any:
        return "transition"
    if active_any:
        return "active"
    if support_any:
        return "support"
    return "unknown"


def _position_stats(
    *,
    body_index: int,
    start: int,
    end: int,
    body_pos_w: np.ndarray | None,
) -> tuple[list[float] | None, dict, str | None]:
    if body_pos_w is None:
        return None, {}, None
    positions = np.asarray(body_pos_w, dtype=float)
    if positions.ndim != 3 or positions.shape[2] < 3:
        return None, {}, None
    if body_index >= positions.shape[1]:
        return None, {}, None
    samples = positions[start:end, body_index, :3]
    if samples.size == 0:
        return None, {}, None
    mean = samples.mean(axis=0)
    first = samples[0]
    last = samples[-1]
    drift_xy = np.linalg.norm(samples[:, :2] - first[:2], axis=1)
    stats = {
        "mean_world_position": mean.tolist(),
        "first_world_position": first.tolist(),
        "last_world_position": last.tolist(),
        "max_drift_xy": float(drift_xy.max()) if drift_xy.size else 0.0,
        "mean_drift_xy": float(drift_xy.mean()) if drift_xy.size else 0.0,
        "position_source": "body_pos_w_mean",
    }
    return mean.tolist(), stats, "body_pos_w_mean"


def anchors_from_contact_mask(
    *,
    motion_id: str,
    contact_mask: np.ndarray,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_pos_w: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "contact_mask",
) -> list[ContactAnchorRecord]:
    contact = _as_2d_bool(contact_mask)
    if contact is None:
        return []
    active = _as_2d_bool(active_mask)
    support = _as_2d_bool(support_mask)
    names = body_names_for_mask(contact, body_names)
    anchors: list[ContactAnchorRecord] = []
    for body_index, body in enumerate(names):
        start: int | None = None
        for frame in range(contact.shape[0] + 1):
            in_contact = frame < contact.shape[0] and bool(contact[frame, body_index])
            if in_contact and start is None:
                start = frame
            if (not in_contact or frame == contact.shape[0]) and start is not None:
                end = frame
                role = _role_for_interval(
                    body_index=body_index,
                    start=start,
                    end=end,
                    active_mask=active,
                    support_mask=support,
                )
                world_position, position_stats, position_source = _position_stats(
                    body_index=body_index,
                    start=start,
                    end=end,
                    body_pos_w=body_pos_w,
                )
                anchor = ContactAnchorRecord(
                    motion_id=motion_id,
                    anchor_id=f"{motion_id}_anchor_{body}_{start:06d}_{end:06d}",
                    body=body,
                    start_frame=start,
                    end_frame=end,
                    role=role,
                    world_position=world_position,
                    position_source=position_source,
                    source=source,
                    metadata=position_stats,
                )
                anchor.validate()
                anchors.append(anchor)
                start = None
    return sorted(anchors, key=lambda anchor: (anchor.start_frame, anchor.end_frame, anchor.body, anchor.anchor_id))
