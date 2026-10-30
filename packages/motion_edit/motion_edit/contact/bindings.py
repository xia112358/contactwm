from __future__ import annotations

from dataclasses import replace

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.schema import ContactAnchorRecord, ContactTransitionRecord
from motion_edit.schema import SegmentRecord


def _overlaps(start: int, end: int, item_start: int, item_end: int) -> bool:
    return item_start < end and item_end > start


def _transition_for_bounds(graph: ContactGraph, start_frame: int, end_frame: int) -> ContactTransitionRecord | None:
    containing = [
        transition
        for transition in graph.transitions
        if transition.start_frame <= start_frame and transition.end_frame >= end_frame
    ]
    if containing:
        return min(containing, key=lambda item: (item.end_frame - item.start_frame, item.transition_id))
    overlapping = [
        transition
        for transition in graph.transitions
        if _overlaps(start_frame, end_frame, transition.start_frame, transition.end_frame)
    ]
    if not overlapping:
        return None
    return max(
        overlapping,
        key=lambda item: (
            min(end_frame, item.end_frame) - max(start_frame, item.start_frame),
            -item.start_frame,
            item.transition_id,
        ),
    )


def _anchor_ids(anchors: list[ContactAnchorRecord]) -> tuple[str | None, str | None]:
    if not anchors:
        return None, None
    ordered = sorted(anchors, key=lambda item: (item.start_frame, item.end_frame, item.anchor_id))
    return ordered[0].anchor_id, ordered[-1].anchor_id


def contact_metadata_for_bounds(graph: ContactGraph, *, start_frame: int, end_frame: int) -> dict:
    events = [event for event in graph.events if start_frame <= event.frame < end_frame]
    transition = _transition_for_bounds(graph, start_frame, end_frame)
    transition_anchor_ids = {
        anchor_id
        for anchor_id in (
            transition.source_anchor_id if transition else None,
            transition.target_anchor_id if transition else None,
        )
        if anchor_id
    }
    anchors = [
        anchor
        for anchor in graph.anchors
        if _overlaps(start_frame, end_frame, anchor.start_frame, anchor.end_frame) or anchor.anchor_id in transition_anchor_ids
    ]
    anchor_ids = {anchor.anchor_id for anchor in anchors}
    patches = [
        patch
        for patch in graph.patches
        if (patch.anchor_id in anchor_ids if patch.anchor_id else _overlaps(start_frame, end_frame, patch.start_frame, patch.end_frame))
    ]
    source_anchor_id, target_anchor_id = _anchor_ids(anchors)
    support_bodies = sorted({anchor.body for anchor in anchors if anchor.role in {"support", "transition"}})
    active_body = transition.active_body if transition else next((anchor.body for anchor in anchors if anchor.role == "active"), None)
    return {
        "contact_transition": transition.to_dict() if transition else None,
        "contact_events": [event.to_dict() for event in events],
        "contact_anchors": [anchor.to_dict() for anchor in anchors],
        "contact_patches": [patch.to_dict() for patch in patches],
        "source_anchor_id": transition.source_anchor_id if transition and transition.source_anchor_id else source_anchor_id,
        "target_anchor_id": transition.target_anchor_id if transition and transition.target_anchor_id else target_anchor_id,
        "active_body": active_body,
        "support_bodies": transition.support_bodies if transition and transition.support_bodies else support_bodies,
        "transition_type": transition.transition_type if transition else "unknown",
        "contact_binding": {
            "source": "contact_graph",
            "motion_id": graph.motion_id,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "event_count": len(events),
            "anchor_count": len(anchors),
            "patch_count": len(patches),
            "transition_id": transition.transition_id if transition else None,
        },
    }


def bind_segment_to_contact_graph(segment: SegmentRecord, graph: ContactGraph) -> SegmentRecord:
    metadata = dict(segment.metadata)
    metadata.update(contact_metadata_for_bounds(graph, start_frame=segment.start_frame, end_frame=segment.end_frame))
    return replace(segment, metadata=metadata)
