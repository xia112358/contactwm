from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from motion_edit.contact.bindings import bind_segment_to_contact_graph
from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.transitions import segment_from_contact_transition
from motion_edit.layers import read_layer
from motion_edit.paths import LAYERS_ROOT
from motion_edit.schema import SegmentRecord
from motion_edit.storage.io import (
    canonical_segment_path,
    read_canonical_segments,
    replace_canonical_segments,
    upsert_motion_version_canonical_path,
    write_canonical_segments,
    write_motion_version,
)
from motion_edit.storage.segments import canonical_segment_id, with_segment_motion_version_id
from motion_edit.storage.schema import MotionVersionRecord


def _parent_transition_id(segment: SegmentRecord) -> str | None:
    transition = segment.metadata.get("contact_transition")
    if isinstance(transition, dict):
        return transition.get("transition_id")
    return segment.metadata.get("parent_transition_id")


def canonicalize_segment(
    segment: SegmentRecord,
    *,
    motion_version_id: str,
    motion_path: str,
    cut_source: str,
) -> SegmentRecord:
    metadata = dict(segment.metadata)
    metadata.setdefault("base_motion_id", segment.motion_id)
    metadata["cut_source"] = cut_source
    parent_transition_id = _parent_transition_id(segment)
    if parent_transition_id:
        metadata["parent_transition_id"] = parent_transition_id
    return with_segment_motion_version_id(
        replace(segment, motion_path=segment.motion_path or motion_path, clip_npz=segment.clip_npz or motion_path, metadata=metadata),
        motion_version_id,
    )


def segments_from_contact_transitions(
    graph: ContactGraph,
    *,
    motion_version_id: str,
    motion_path: str,
    source: str = "canonical",
    status: str = "candidate",
    cut_source: str = "contact_auto",
) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    for index, transition in enumerate(graph.transitions):
        segment = segment_from_contact_transition(
            transition=transition,
            segment_id=canonical_segment_id(motion_version_id, index=index),
            source=source,
            status=status,
            track="contact",
            motion_path=motion_path,
            clip_npz=motion_path,
            clip_output_dir=None,
            clip_file_name=None,
            atom_label="",
            contact_start=transition.metadata.get("contact_start"),
            contact_end=transition.metadata.get("contact_end"),
            active=transition.active_body,
            support=",".join(transition.support_bodies),
            events=graph.events,
            anchors=graph.anchors,
            metadata={"parent_transition_id": transition.transition_id, "cut_source": cut_source},
        )
        segments.append(canonicalize_segment(segment, motion_version_id=motion_version_id, motion_path=motion_path, cut_source=cut_source))
    return segments


def load_source_segments_for_motion(source: str | None, motion_id: str) -> list[SegmentRecord]:
    if not source:
        return []
    path = LAYERS_ROOT / source / f"{motion_id}.jsonl"
    if not path.exists():
        return []
    default_status = "candidate" if source.startswith("candidates/") else "manual"
    return read_layer(path, default_source=source.split("/", 1)[-1], default_status=default_status)


def build_canonical_segments(
    *,
    motion_version_id: str,
    motion_path: str,
    graph: ContactGraph,
    source_segments: list[SegmentRecord] | None = None,
    cut_source: str = "contact_auto",
) -> list[SegmentRecord]:
    source_items = source_segments or []
    if source_items:
        rebound = [bind_segment_to_contact_graph(segment, graph) for segment in source_items]
        return [
            canonicalize_segment(segment, motion_version_id=motion_version_id, motion_path=motion_path, cut_source=cut_source)
            for segment in rebound
        ]
    return segments_from_contact_transitions(
        graph,
        motion_version_id=motion_version_id,
        motion_path=motion_path,
        cut_source=cut_source,
    )


def write_motion_version_with_canonical_segments(
    *,
    motion_version_id: str,
    motion_path: str,
    contact_layer: str,
    segments: list[SegmentRecord],
    kind: str = "raw",
    base_motion_id: str | None = None,
    reset_canonical: bool = False,
    reason: str | None = None,
    source: str = "contact_auto",
) -> tuple[MotionVersionRecord, Path]:
    active_path = canonical_segment_path(motion_version_id)
    if active_path.exists() and not reset_canonical:
        raise ValueError(
            f"canonical segmentation already exists for {motion_version_id}; "
            "use cutter/update/mark-status to refine it or pass --reset-canonical"
        )
    if reset_canonical:
        segment_path = replace_canonical_segments(
            motion_version_id,
            segments,
            reason=reason or "reset canonical segmentation",
            source=source,
            kind="reset_canonical_segmentation",
            backup_existing=True,
        )
    else:
        segment_path = write_canonical_segments(
            motion_version_id,
            segments,
            reason=reason or "build canonical segmentation",
            source=source,
        )
    record = upsert_motion_version_canonical_path(
        motion_version_id,
        segment_path,
        contact_layer=contact_layer,
        motion_path=motion_path,
    )
    if record.base_motion_id is None and base_motion_id is not None:
        record = MotionVersionRecord(
            motion_version_id=record.motion_version_id,
            motion_path=record.motion_path,
            kind=record.kind,
            base_motion_id=base_motion_id,
            motion_asset_id=record.motion_asset_id,
            parent_motion_version_id=record.parent_motion_version_id,
            contact_layer=record.contact_layer,
            canonical_segment_path=record.canonical_segment_path,
            token_catalog_path=record.token_catalog_path,
            edit_plan_id=record.edit_plan_id,
            metadata=dict(record.metadata),
        )
        write_motion_version(record)
    return record, segment_path


def mark_canonical_segment_status(
    *,
    motion_version_id: str,
    segment_id: str,
    status: str,
    reason: str | None = None,
    source: str = "mark_segment_status",
) -> list[SegmentRecord]:
    return mark_canonical_segment_statuses(
        motion_version_id=motion_version_id,
        updates=[{"segment_id": segment_id, "status": status, "reason": reason, "source": source}],
    )


def mark_canonical_segment_statuses(
    *,
    motion_version_id: str,
    updates: list[dict],
) -> list[SegmentRecord]:
    segments = read_canonical_segments(motion_version_id)
    update_by_id = {str(update["segment_id"]): update for update in updates}
    updated: list[SegmentRecord] = []
    found: set[str] = set()
    for segment in segments:
        if segment.segment_id in update_by_id:
            update = update_by_id[segment.segment_id]
            status = str(update["status"])
            metadata = dict(segment.metadata)
            history = list(metadata.get("status_history") or [])
            event = {
                "old_status": segment.status,
                "new_status": status,
                "source": str(update.get("source") or "mark_segment_status"),
            }
            if update.get("reason") is not None:
                event["reason"] = str(update["reason"])
            history.append(event)
            metadata["status_history"] = history
            updated.append(replace(segment, status=status, metadata=metadata))  # type: ignore[arg-type]
            found.add(segment.segment_id)
        else:
            updated.append(segment)
    missing = sorted(set(update_by_id) - found)
    if missing:
        raise ValueError(f"segments not found in canonical segmentation: {', '.join(missing)}")
    sources = {str(update.get("source") or "mark_segment_status") for update in updates}
    reasons = [str(update["reason"]) for update in updates if update.get("reason") is not None]
    source = sources.pop() if len(sources) == 1 else "mark_segment_status"
    reason = reasons[0] if len(reasons) == 1 else f"status update for {len(updates)} segment(s)"
    replace_canonical_segments(
        motion_version_id,
        updated,
        reason=reason,
        source=source,
        kind="mark_status",
    )
    return updated
