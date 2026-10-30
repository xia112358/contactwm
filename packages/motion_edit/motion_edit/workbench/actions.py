from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from motion_edit.schema import SegmentRecord

WorkbenchStatus = Literal["candidate", "accepted", "rejected", "manual"]


def _with_provenance(segment: SegmentRecord, kind: str, params: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(segment.metadata)
    edits = list(metadata.get("motion_edit_edits") or [])
    edits.append(
        {
            "kind": kind,
            "source": "workbench",
            "parent_segment_id": segment.segment_id,
            "params": params,
        }
    )
    metadata["motion_edit_edits"] = edits
    return metadata


def trim_segment(
    segment: SegmentRecord,
    *,
    start_frame: int,
    end_frame: int,
    allow_extend: bool = False,
) -> SegmentRecord:
    """Return a segment with edited bounds while preserving its identity."""
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if not allow_extend and not (segment.start_frame <= start_frame < end_frame <= segment.end_frame):
        raise ValueError(
            f"{segment.segment_id}: trim must stay inside "
            f"[{segment.start_frame}, {segment.end_frame}], got [{start_frame}, {end_frame}]"
        )
    trimmed = replace(
        segment,
        start_frame=start_frame,
        end_frame=end_frame,
        metadata=_with_provenance(
            segment,
            "trim",
            {
                "old_start_frame": segment.start_frame,
                "old_end_frame": segment.end_frame,
                "new_start_frame": start_frame,
                "new_end_frame": end_frame,
                "allow_extend": bool(allow_extend),
            },
        ),
    )
    trimmed.validate()
    return trimmed


def split_segment(segment: SegmentRecord, *, frame: int) -> tuple[SegmentRecord, SegmentRecord]:
    """Split a segment at an interior frame into left/right child records."""
    frame = int(frame)
    if frame <= segment.start_frame or frame >= segment.end_frame:
        raise ValueError(
            f"{segment.segment_id}: split frame must be inside "
            f"[{segment.start_frame}, {segment.end_frame}), got {frame}"
        )
    left = replace(
        segment,
        segment_id=f"{segment.segment_id}_split0_{segment.start_frame}_{frame}",
        end_frame=frame,
        metadata=_with_provenance(
            segment,
            "split",
            {
                "side": "left",
                "split_frame": frame,
                "old_start_frame": segment.start_frame,
                "old_end_frame": segment.end_frame,
            },
        ),
    )
    right = replace(
        segment,
        segment_id=f"{segment.segment_id}_split1_{frame}_{segment.end_frame}",
        start_frame=frame,
        metadata=_with_provenance(
            segment,
            "split",
            {
                "side": "right",
                "split_frame": frame,
                "old_start_frame": segment.start_frame,
                "old_end_frame": segment.end_frame,
            },
        ),
    )
    left.validate()
    right.validate()
    return left, right


def curate_segment(segment: SegmentRecord, *, status: WorkbenchStatus) -> SegmentRecord:
    if status not in {"candidate", "accepted", "rejected", "manual"}:
        raise ValueError(f"unsupported segment status: {status}")
    curated = replace(
        segment,
        status=status,
        metadata=_with_provenance(segment, "curate", {"old_status": segment.status, "new_status": status}),
    )
    curated.validate()
    return curated
