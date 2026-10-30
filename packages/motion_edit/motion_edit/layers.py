from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, segment_from_dict, segment_to_dict, write_jsonl
from .schema import SegmentRecord


def read_layer(path: Path, *, default_source: str, default_status: str) -> list[SegmentRecord]:
    return [segment_from_dict(item, default_source=default_source, default_status=default_status) for item in read_jsonl(path)]


def write_layer(path: Path, segments: list[SegmentRecord]) -> None:
    write_jsonl(path, (segment_to_dict(segment) for segment in segments))


def iter_layer_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*.jsonl"))


def group_by_motion(segments: list[SegmentRecord]) -> dict[str, list[SegmentRecord]]:
    grouped: dict[str, list[SegmentRecord]] = {}
    for segment in segments:
        grouped.setdefault(segment.motion_id, []).append(segment)
    for items in grouped.values():
        items.sort(key=lambda s: (s.start_frame, s.end_frame, s.segment_id))
    return grouped
