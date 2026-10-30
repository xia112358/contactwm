from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from motion_edit.contact.anchors import anchors_from_contact_mask
from motion_edit.contact.events import bodies_from_mask, body_names_for_mask, detect_contact_events
from motion_edit.contact.patches import patches_from_anchors
from motion_edit.contact.schema import ContactAnchorRecord, ContactEventRecord, ContactTransitionRecord
from motion_edit.schema import SegmentRecord


ANCHOR_PAIR_ENDPOINT_POLICY = "contact_point_inclusive"


def mask_string(mask: np.ndarray) -> str:
    return "".join("1" if bool(value) else "0" for value in np.asarray(mask).reshape(-1))


def _as_2d_bool(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 1D or 2D, got shape={arr.shape}")
    return arr


def _frame_count(*masks: np.ndarray | None, anchors: Sequence[ContactAnchorRecord] | None = None) -> int:
    for mask in masks:
        if mask is not None:
            return int(mask.shape[0])
    if anchors:
        return max(int(anchor.end_frame) for anchor in anchors)
    return 0


def _mask_bodies_at(mask: np.ndarray | None, frame: int, names: list[str]) -> list[str]:
    if mask is None or mask.shape[0] == 0:
        return []
    clamped = max(0, min(int(frame), int(mask.shape[0]) - 1))
    return bodies_from_mask(mask[clamped], names)


def _mask_string_at(mask: np.ndarray | None, frame: int) -> str | None:
    if mask is None or mask.shape[0] == 0:
        return None
    clamped = max(0, min(int(frame), int(mask.shape[0]) - 1))
    return mask_string(mask[clamped])


def _event_at_or_after(events: list[ContactEventRecord], start: int, end: int) -> ContactEventRecord | None:
    return next((event for event in events if start <= event.frame < end), None)


def _event_at_or_before(events: list[ContactEventRecord], start: int, end: int) -> ContactEventRecord | None:
    return next((event for event in reversed(events) if start < event.frame <= end), None)


def _event_near(
    events: list[ContactEventRecord],
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


def _anchor_for_body_at(
    anchors: list[ContactAnchorRecord],
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


def _first_body(mask: np.ndarray | None, frame: int, names: list[str]) -> str | None:
    bodies = _mask_bodies_at(mask, frame, names)
    return bodies[0] if bodies else None


def _body_sort_key(body: str, names: list[str]) -> tuple[int, str]:
    return (names.index(body), body) if body in names else (len(names), body)


def _anchor_pair_window(
    source_anchor: ContactAnchorRecord,
    target_anchor: ContactAnchorRecord,
    *,
    n_frames: int,
) -> tuple[int, int, int, int]:
    """Return a half-open contact-point segment window.

    The segment owns the source contact point, the intervening transfer, and the
    target contact point, but not the full neighboring contact holds. This keeps
    an edit local to the incoming A2A transfer instead of leaking into the next
    segment.
    """

    source_point = max(int(source_anchor.start_frame), int(source_anchor.end_frame) - 1)
    target_point = int(target_anchor.start_frame)
    start = source_point
    end = target_point + 1
    if n_frames > 0:
        start = max(0, min(n_frames - 1, start))
        end = max(start + 1, min(n_frames, end))
    if end <= start:
        start = max(0, int(source_anchor.start_frame))
        end = max(start + 1, int(target_anchor.end_frame))
        if n_frames > 0:
            end = max(start + 1, min(n_frames, end))
    return start, end, source_point, target_point


def _transition_type_for_anchor_pair(source_anchor: ContactAnchorRecord, target_anchor: ContactAnchorRecord) -> str:
    body = target_anchor.body.lower()
    if "hand" in body or "wrist" in body or "palm" in body:
        return "reach"
    if "foot" in body or "ankle" in body or "toe" in body or "heel" in body:
        if source_anchor.world_position is not None and target_anchor.world_position is not None:
            delta = np.asarray(target_anchor.world_position, dtype=np.float64) - np.asarray(source_anchor.world_position, dtype=np.float64)
            if float(np.linalg.norm(delta[:2])) < 0.08 and abs(float(delta[2])) < 0.08:
                return "micro_adjust"
        return "step"
    return "support_transfer"


def transitions_from_anchor_pairs(
    *,
    motion_id: str,
    anchors: list[ContactAnchorRecord],
    events: list[ContactEventRecord],
    contact_mask: np.ndarray | None,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "contact_mask",
) -> list[ContactTransitionRecord]:
    """Build contact-to-contact A2A segments from consecutive anchors.

    Each transition is local to one body lane: source contact point -> next
    contact point of the same body. The resulting frame range is intentionally
    the source endpoint, transfer interior, and target endpoint only. Neighboring
    contact holds are exposed in metadata but not made part of the owned segment
    window.
    """

    contact = _as_2d_bool(contact_mask)
    active = _as_2d_bool(active_mask)
    support = _as_2d_bool(support_mask)
    reference = next((mask for mask in (contact, active, support) if mask is not None), None)
    names = body_names_for_mask(reference, body_names) if reference is not None else list(body_names or [])
    n_frames = _frame_count(contact, active, support, anchors=anchors)

    by_body: dict[str, list[ContactAnchorRecord]] = defaultdict(list)
    for anchor in anchors:
        by_body[anchor.body].append(anchor)
    ordered_bodies = sorted(by_body, key=lambda body: _body_sort_key(body, names))

    transitions: list[ContactTransitionRecord] = []
    for body in ordered_bodies:
        body_anchors = sorted(by_body[body], key=lambda anchor: (anchor.start_frame, anchor.end_frame, anchor.anchor_id))
        for local_index, (source_anchor, target_anchor) in enumerate(zip(body_anchors, body_anchors[1:])):
            start, end, source_point, target_point = _anchor_pair_window(
                source_anchor,
                target_anchor,
                n_frames=n_frames,
            )
            if end <= start:
                continue
            source_event = _event_near(
                events,
                body=body,
                event_type="liftoff",
                frame=int(source_anchor.end_frame),
            )
            target_event = _event_near(
                events,
                body=body,
                event_type="touchdown",
                frame=int(target_anchor.start_frame),
            )
            support_bodies = _mask_bodies_at(support, start, names)
            if not support_bodies:
                support_bodies = [item for item in _mask_bodies_at(contact, start, names) if item != body]
            transition = ContactTransitionRecord(
                motion_id=motion_id,
                transition_id=f"{motion_id}_{source}_anchor_pair_{len(transitions):04d}",
                start_frame=start,
                end_frame=end,
                active_body=body,
                support_bodies=support_bodies,
                start_event_id=source_event.event_id if source_event else None,
                end_event_id=target_event.event_id if target_event else None,
                source_anchor_id=source_anchor.anchor_id,
                target_anchor_id=target_anchor.anchor_id,
                transition_type=_transition_type_for_anchor_pair(source_anchor, target_anchor),
                source=source,
                metadata={
                    "segmentation_kind": "anchor_pair",
                    "endpoint_policy": ANCHOR_PAIR_ENDPOINT_POLICY,
                    "body_lane_index": local_index,
                    "body_names": names,
                    "contact_start": _mask_string_at(contact, start),
                    "contact_end": _mask_string_at(contact, max(start, end - 1)),
                    "source_anchor_frame": int(source_point),
                    "target_anchor_frame": int(target_point),
                    "source_anchor_interval": [int(source_anchor.start_frame), int(source_anchor.end_frame)],
                    "target_anchor_interval": [int(target_anchor.start_frame), int(target_anchor.end_frame)],
                    "owned_bodies": [body],
                    "outside_policy": "copy_original",
                },
            )
            transition.validate()
            transitions.append(transition)
    return transitions


def transitions_from_proto_indices(
    *,
    motion_id: str,
    starts: Iterable[int],
    ends: Iterable[int],
    contact_mask: np.ndarray | None,
    active_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    body_pos_w: np.ndarray | None = None,
    body_names: Iterable[str] | None = None,
    source: str = "force_contact",
) -> tuple[list[ContactEventRecord], list[ContactAnchorRecord], list[ContactTransitionRecord]]:
    contact = _as_2d_bool(contact_mask)
    active = _as_2d_bool(active_mask)
    support = _as_2d_bool(support_mask)
    reference = next((mask for mask in (contact, active, support) if mask is not None), None)
    names = body_names_for_mask(reference, body_names) if reference is not None else list(body_names or [])
    events = (
        detect_contact_events(
            motion_id=motion_id,
            contact_mask=contact,
            active_mask=active,
            support_mask=support,
            body_names=names,
            source=source,
        )
        if reference is not None
        else []
    )
    anchors = (
        anchors_from_contact_mask(
            motion_id=motion_id,
            contact_mask=contact,
            active_mask=active,
            support_mask=support,
            body_pos_w=body_pos_w,
            body_names=names,
            source=source,
        )
        if contact is not None
        else []
    )
    transitions: list[ContactTransitionRecord] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        end_sample = max(start_i, end_i - 1)
        active_body = _first_body(active, start_i, names)
        support_bodies = _mask_bodies_at(support, start_i, names)
        start_event = _event_at_or_after(events, start_i, end_i)
        end_event = _event_at_or_before(events, start_i, end_i)
        source_anchor = _anchor_for_body_at(anchors, body=active_body, frame=start_i)
        target_anchor = _anchor_for_body_at(anchors, body=active_body, frame=end_sample)
        transition = ContactTransitionRecord(
            motion_id=motion_id,
            transition_id=f"{motion_id}_{source}_transition_{index:04d}",
            start_frame=start_i,
            end_frame=end_i,
            active_body=active_body,
            support_bodies=support_bodies,
            start_event_id=start_event.event_id if start_event else None,
            end_event_id=end_event.event_id if end_event else None,
            source_anchor_id=source_anchor.anchor_id if source_anchor else None,
            target_anchor_id=target_anchor.anchor_id if target_anchor else None,
            transition_type="support_transfer" if start_event or end_event else "unknown",
            source=source,
            metadata={
                "segmentation_kind": "proto_index",
                "body_names": names,
                "contact_start": _mask_string_at(contact, start_i),
                "contact_end": _mask_string_at(contact, end_sample),
            },
        )
        transition.validate()
        transitions.append(transition)
    return events, anchors, transitions


def transitions_from_event_pairs(
    *,
    motion_id: str,
    events: list[ContactEventRecord],
    anchors: list[ContactAnchorRecord],
    source: str = "contact_mask",
) -> list[ContactTransitionRecord]:
    transitions: list[ContactTransitionRecord] = []
    liftoffs = [event for event in events if event.event_type == "liftoff"]
    touchdowns = [event for event in events if event.event_type == "touchdown"]
    for index, start_event in enumerate(liftoffs):
        end_event = next(
            (
                event
                for event in touchdowns
                if event.body == start_event.body and event.frame > start_event.frame
            ),
            None,
        )
        if end_event is None:
            continue
        source_anchor = _anchor_for_body_at(anchors, body=start_event.body, frame=max(0, start_event.frame - 1))
        target_anchor = _anchor_for_body_at(anchors, body=start_event.body, frame=end_event.frame)
        transition = ContactTransitionRecord(
            motion_id=motion_id,
            transition_id=f"{motion_id}_{source}_event_transition_{index:04d}",
            start_frame=start_event.frame,
            end_frame=end_event.frame,
            active_body=start_event.body,
            support_bodies=end_event.contact_after,
            start_event_id=start_event.event_id,
            end_event_id=end_event.event_id,
            source_anchor_id=source_anchor.anchor_id if source_anchor else None,
            target_anchor_id=target_anchor.anchor_id if target_anchor else None,
            transition_type="step",
            source=source,
            metadata={
                "segmentation_kind": "event_pair",
                "source_event_frame": int(start_event.frame),
                "target_event_frame": int(end_event.frame),
                "outside_policy": "copy_original",
            },
        )
        transition.validate()
        transitions.append(transition)
    return transitions


def segment_from_contact_transition(
    *,
    transition: ContactTransitionRecord,
    segment_id: str,
    source: str,
    status: str,
    track: str,
    motion_path: str | None,
    clip_npz: str | None,
    clip_output_dir: str | None,
    clip_file_name: str | None,
    atom_label: str,
    contact_start: str | None,
    contact_end: str | None,
    active: str | None,
    support: str | None,
    events: list[ContactEventRecord],
    anchors: list[ContactAnchorRecord],
    metadata: dict | None = None,
) -> SegmentRecord:
    transition_events = [
        event.to_dict()
        for event in events
        if transition.start_frame <= event.frame < transition.end_frame
    ]
    overlapping_anchors = [
        anchor
        for anchor in anchors
        if anchor.start_frame < transition.end_frame and anchor.end_frame > transition.start_frame
    ]
    meta = dict(metadata or {})
    transition_metadata = dict(transition.metadata or {})
    contact_start_value = contact_start if contact_start is not None else transition_metadata.get("contact_start")
    contact_end_value = contact_end if contact_end is not None else transition_metadata.get("contact_end")
    meta.update(
        {
            "contact_transition": transition.to_dict(),
            "contact_events": transition_events,
            "contact_anchors": [anchor.to_dict() for anchor in overlapping_anchors],
            "contact_patches": [patch.to_dict() for patch in patches_from_anchors(overlapping_anchors)],
            "source_anchor_id": transition.source_anchor_id,
            "target_anchor_id": transition.target_anchor_id,
            "active_body": transition.active_body,
            "support_bodies": transition.support_bodies,
            "transition_type": transition.transition_type,
            "segmentation_kind": transition_metadata.get("segmentation_kind"),
            "endpoint_policy": transition_metadata.get("endpoint_policy"),
            "outside_policy": transition_metadata.get("outside_policy", "copy_original"),
            "owned_bodies": transition_metadata.get("owned_bodies"),
        }
    )
    segment = SegmentRecord(
        motion_id=transition.motion_id,
        segment_id=segment_id,
        start_frame=transition.start_frame,
        end_frame=transition.end_frame,
        source=source,
        status=status,  # type: ignore[arg-type]
        track=track,
        motion_path=motion_path,
        clip_npz=clip_npz,
        clip_output_dir=clip_output_dir,
        clip_file_name=clip_file_name,
        atom_label=atom_label,
        contact_start=contact_start_value,
        contact_end=contact_end_value,
        active=active,
        support=support,
        metadata=meta,
    )
    segment.validate()
    return segment
