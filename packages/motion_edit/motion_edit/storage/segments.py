from __future__ import annotations

from dataclasses import replace

from motion_edit.schema import SegmentRecord


def get_segment_motion_version_id(segment: SegmentRecord) -> str | None:
    value = segment.metadata.get("motion_version_id")
    return str(value) if value else None


def with_segment_motion_version_id(segment: SegmentRecord, motion_version_id: str) -> SegmentRecord:
    metadata = dict(segment.metadata)
    metadata["motion_version_id"] = motion_version_id
    return replace(segment, metadata=metadata)


def canonical_segment_id(motion_version_id: str, index: int | None = None, transition_id: str | None = None) -> str:
    if transition_id:
        return f"{motion_version_id}_{transition_id}"
    if index is None:
        raise ValueError("canonical_segment_id requires index or transition_id")
    return f"{motion_version_id}_seg_{index:04d}"
