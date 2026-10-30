from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from motion_edit.contact.events import bodies_from_mask, body_names_for_mask
from motion_edit.contact.schema import ContactAnchorRecord, ContactEventRecord, ContactTransitionRecord
from motion_edit.contact.transitions import mask_string


STABLE_PROTO_KIND = "stable_contact_anchor"
STABLE_PROTO_ENDPOINT_POLICY = "global_stable_contact_cluster"
ACTIVE_MOTION_PARTS = ("left_foot", "right_foot", "left_hand", "right_hand")


@dataclass(frozen=True)
class StableContactProtoConfig:
    """Configuration for force/contact-centered A2A proto extraction.

    The contact mask is the authority for contact state. Optional part positions
    are only used to delay a touchdown anchor until the newly contacting part is
    kinematically stable.
    """

    merge_transition_window: int = 2
    stable_touchdown_window: int = 3
    stable_touchdown_speed_thresh: float = 0.12
    stable_touchdown_cluster_window: int = 6
    min_contact_island: int = 3
    fps: float = 50.0
    active_motion_parts: tuple[str, ...] = ACTIVE_MOTION_PARTS


def _as_2d_bool(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim != 2:
        raise ValueError(f"contact mask must be 1D or 2D, got shape={arr.shape}")
    return arr


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    values = np.asarray(mask, dtype=bool).reshape(-1)
    for frame, value in enumerate(np.r_[values, False]):
        if bool(value) and start is None:
            start = frame
        elif not bool(value) and start is not None:
            out.append((start, frame))
            start = None
    return out


def clean_contact_mask(mask: np.ndarray, *, min_island: int = 3) -> np.ndarray:
    """Fill one-frame holes and drop tiny contact islands."""

    cleaned = np.asarray(mask, dtype=bool).copy().reshape(-1)
    for frame in range(1, len(cleaned) - 1):
        if not cleaned[frame] and cleaned[frame - 1] and cleaned[frame + 1]:
            cleaned[frame] = True
    out = np.zeros_like(cleaned)
    for start, end in _intervals(cleaned):
        if end - start >= int(min_island):
            out[start:end] = True
    return out


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy().reshape(-1)
    if int(max_gap) <= 0:
        return filled
    for start, end in _intervals(~filled):
        if start == 0 or end == len(filled):
            continue
        if end - start <= int(max_gap):
            filled[start:end] = True
    return filled


def debounced_contact_mask(contact_mask: np.ndarray, *, max_gap: int = 2, min_island: int = 3) -> np.ndarray:
    contact = _as_2d_bool(contact_mask)
    if contact is None:
        return np.zeros((0, 0), dtype=bool)
    out = np.zeros_like(contact, dtype=bool)
    for part_i in range(contact.shape[1]):
        out[:, part_i] = clean_contact_mask(
            _fill_short_gaps(contact[:, part_i], int(max_gap)),
            min_island=int(min_island),
        )
    return out


def _part_speed(body_pos_w: np.ndarray | None, part_i: int, n_frames: int, fps: float) -> np.ndarray:
    if body_pos_w is None:
        return np.zeros(n_frames, dtype=np.float64)
    pos = np.asarray(body_pos_w, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[0] < n_frames or pos.shape[1] <= part_i or pos.shape[2] < 3:
        return np.zeros(n_frames, dtype=np.float64)
    vel = np.zeros((n_frames, 3), dtype=np.float64)
    vel[1:] = np.diff(pos[:n_frames, part_i, :3], axis=0) * float(fps)
    return np.linalg.norm(vel, axis=1)


def _merge_anchor_events(
    num_frames: int,
    num_parts: int,
    events: list[tuple[int, int]],
    cluster_window: int,
) -> tuple[list[int], dict[int, np.ndarray]]:
    touchdown_by_anchor: dict[int, np.ndarray] = {}
    cluster_frame: int | None = None
    cluster_mask = np.zeros(num_parts, dtype=bool)
    for frame, part_i in sorted(events):
        frame_i = int(frame)
        if cluster_frame is None:
            cluster_frame = frame_i
            cluster_mask = np.zeros(num_parts, dtype=bool)
            cluster_mask[int(part_i)] = True
            continue
        if frame_i - cluster_frame <= int(cluster_window):
            cluster_frame = frame_i
            cluster_mask[int(part_i)] = True
            continue
        touchdown_by_anchor[cluster_frame] = cluster_mask.copy()
        cluster_frame = frame_i
        cluster_mask = np.zeros(num_parts, dtype=bool)
        cluster_mask[int(part_i)] = True
    if cluster_frame is not None:
        touchdown_by_anchor[cluster_frame] = cluster_mask.copy()

    anchors = sorted(touchdown_by_anchor)
    if not anchors or anchors[0] != 0:
        anchors.insert(0, 0)
    if anchors[-1] != int(num_frames):
        anchors.append(int(num_frames))
    return anchors, touchdown_by_anchor


def stable_contact_anchor_maps(
    *,
    contact_mask: np.ndarray,
    body_pos_w: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    config: StableContactProtoConfig | None = None,
) -> tuple[list[int], dict[int, np.ndarray], dict[str, object]]:
    """Extract global stable-contact anchor frames from force/contact masks."""

    cfg = config or StableContactProtoConfig()
    contact = debounced_contact_mask(
        contact_mask,
        max_gap=int(cfg.merge_transition_window),
        min_island=int(cfg.min_contact_island),
    )
    if contact.size == 0:
        return [], {}, {"segmentation_kind": STABLE_PROTO_KIND, "reason": "empty_contact_mask"}
    names = body_names_for_mask(contact, body_names)
    active_parts = set(cfg.active_motion_parts)
    role_indices = [index for index, name in enumerate(names) if name in active_parts]
    if not role_indices:
        role_indices = list(range(contact.shape[1]))

    n_frames = int(contact.shape[0])
    contact_window = max(1, int(cfg.stable_touchdown_window))
    free_window = max(1, int(cfg.stable_touchdown_window))
    events: list[tuple[int, int]] = []
    for part_i in role_indices:
        part_contact = contact[:, part_i]
        speed = _part_speed(body_pos_w, part_i, n_frames, float(cfg.fps))
        for frame in range(free_window, n_frames - contact_window + 1):
            if part_contact[frame - 1] or not part_contact[frame]:
                continue
            if part_contact[frame - free_window : frame].any():
                continue
            for candidate in range(frame, n_frames - contact_window + 1):
                if not part_contact[candidate]:
                    break
                stable_contact = bool(part_contact[candidate : candidate + contact_window].all())
                stable_speed = bool(np.all(speed[candidate : candidate + contact_window] <= float(cfg.stable_touchdown_speed_thresh)))
                if stable_contact and stable_speed:
                    events.append((candidate, part_i))
                    break

    anchors, touchdown_by_anchor = _merge_anchor_events(
        n_frames,
        contact.shape[1],
        events,
        int(cfg.stable_touchdown_cluster_window),
    )
    metadata: dict[str, object] = {
        "segmentation_kind": STABLE_PROTO_KIND,
        "endpoint_policy": STABLE_PROTO_ENDPOINT_POLICY,
        "contact_source": "force_or_contact_mask",
        "body_names": names,
        "stable_touchdown_frames": sorted(int(frame) for frame, _part_i in events),
        "stable_anchor_frames": [int(frame) for frame in anchors],
        "stable_touchdown_count": len(events),
        "internal_anchor_count": max(0, len(anchors) - 2),
        "config": {
            "merge_transition_window": int(cfg.merge_transition_window),
            "stable_touchdown_window": int(cfg.stable_touchdown_window),
            "stable_touchdown_speed_thresh": float(cfg.stable_touchdown_speed_thresh),
            "stable_touchdown_cluster_window": int(cfg.stable_touchdown_cluster_window),
            "min_contact_island": int(cfg.min_contact_island),
            "fps": float(cfg.fps),
            "active_motion_parts": list(cfg.active_motion_parts),
        },
        "debounced_contact_mask": contact,
    }
    return anchors, touchdown_by_anchor, metadata


def _mask_at(mask: np.ndarray | None, frame: int, names: list[str]) -> list[str]:
    if mask is None or mask.shape[0] == 0:
        return []
    frame_i = max(0, min(int(frame), int(mask.shape[0]) - 1))
    return bodies_from_mask(mask[frame_i], names)


def _mask_string_at(mask: np.ndarray | None, frame: int) -> str | None:
    if mask is None or mask.shape[0] == 0:
        return None
    frame_i = max(0, min(int(frame), int(mask.shape[0]) - 1))
    return mask_string(mask[frame_i])


def _anchor_for_body_at(
    anchors: Sequence[ContactAnchorRecord],
    *,
    body: str | None,
    frame: int,
) -> ContactAnchorRecord | None:
    if body is None:
        return None
    return next(
        (
            anchor
            for anchor in anchors
            if anchor.body == body and anchor.start_frame <= frame < anchor.end_frame
        ),
        None,
    )


def _event_near(
    events: Sequence[ContactEventRecord],
    *,
    body: str,
    event_type: str,
    frame: int,
    window: int = 1,
) -> ContactEventRecord | None:
    candidates = [
        event
        for event in events
        if event.body == body and event.event_type == event_type and abs(int(event.frame) - int(frame)) <= int(window)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda event: (abs(int(event.frame) - int(frame)), int(event.frame), event.event_id))


def transitions_from_stable_contact_anchors(
    *,
    motion_id: str,
    anchors: list[ContactAnchorRecord],
    events: list[ContactEventRecord],
    contact_mask: np.ndarray | None,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_pos_w: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "contact_mask",
    config: StableContactProtoConfig | None = None,
) -> list[ContactTransitionRecord]:
    contact = _as_2d_bool(contact_mask)
    if contact is None or contact.size == 0:
        return []
    active = _as_2d_bool(active_mask)
    support = _as_2d_bool(support_mask)
    cfg = config or StableContactProtoConfig()
    names = body_names_for_mask(contact, body_names)
    stable_anchors, touchdown_by_anchor, stable_metadata = stable_contact_anchor_maps(
        contact_mask=contact,
        body_pos_w=body_pos_w,
        body_names=names,
        config=cfg,
    )
    debounced_contact = np.asarray(stable_metadata.get("debounced_contact_mask"), dtype=bool)
    if len(stable_anchors) < 2 or not touchdown_by_anchor:
        return []

    transitions: list[ContactTransitionRecord] = []
    for proto_id, (start, end) in enumerate(zip(stable_anchors[:-1], stable_anchors[1:])):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i or end_i not in touchdown_by_anchor:
            continue
        end_sample = max(start_i, min(end_i - 1, contact.shape[0] - 1))
        endpoint_sample = max(start_i, min(end_i, contact.shape[0] - 1))
        touchdown_mask = touchdown_by_anchor[end_i]
        active_bodies = bodies_from_mask(touchdown_mask, names)
        if not active_bodies and active is not None:
            active_bodies = _mask_at(active, start_i, names)
        active_body = active_bodies[0] if active_bodies else None
        support_bodies = _mask_at(support, start_i, names) if support is not None else []
        if not support_bodies:
            support_bodies = [body for body in _mask_at(debounced_contact, start_i, names) if body not in set(active_bodies)]
        free_bodies = [body for body in names if body not in set(active_bodies) and body not in set(support_bodies)]

        source_anchor = _anchor_for_body_at(anchors, body=active_body, frame=start_i)
        target_anchor = _anchor_for_body_at(anchors, body=active_body, frame=endpoint_sample)
        start_event = _event_near(events, body=active_body, event_type="liftoff", frame=start_i) if active_body else None
        end_event = _event_near(events, body=active_body, event_type="touchdown", frame=end_i) if active_body else None
        transition = ContactTransitionRecord(
            motion_id=motion_id,
            transition_id=f"{motion_id}_{source}_stable_proto_{proto_id:04d}",
            start_frame=start_i,
            end_frame=end_i,
            active_body=active_body,
            support_bodies=support_bodies,
            start_event_id=start_event.event_id if start_event else None,
            end_event_id=end_event.event_id if end_event else None,
            source_anchor_id=source_anchor.anchor_id if source_anchor else None,
            target_anchor_id=target_anchor.anchor_id if target_anchor else None,
            transition_type="support_transfer" if active_bodies else "unknown",
            source=source,
            metadata={
                "segmentation_kind": STABLE_PROTO_KIND,
                "endpoint_policy": STABLE_PROTO_ENDPOINT_POLICY,
                "outside_policy": "copy_original",
                "proto_index": int(proto_id),
                "body_names": names,
                "anchor_start": int(start_i),
                "anchor_end": int(end_i),
                "contact_start": _mask_string_at(debounced_contact, start_i),
                "contact_end": _mask_string_at(debounced_contact, endpoint_sample),
                "touchdown_part": mask_string(touchdown_mask),
                "active_bodies": active_bodies,
                "support_bodies": support_bodies,
                "free_bodies": free_bodies,
                "owned_bodies": active_bodies,
                "stable_anchor_frames": stable_metadata.get("stable_anchor_frames"),
                "stable_touchdown_count": stable_metadata.get("stable_touchdown_count"),
                "config": stable_metadata.get("config"),
            },
        )
        transition.validate()
        transitions.append(transition)
    return transitions
