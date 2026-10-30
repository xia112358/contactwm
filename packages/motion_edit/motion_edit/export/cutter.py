from __future__ import annotations

from pathlib import Path

from motion_edit.io import write_jsonl
from motion_edit.layers import group_by_motion
from motion_edit.schema import SegmentRecord


def export_cutter_segments(output_dir: Path, segments: list[SegmentRecord]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for motion_id, motion_segments in group_by_motion(segments).items():
        path = output_dir / f"{motion_id}.segments.jsonl"
        write_jsonl(path, (segment.to_cutter_json() for segment in motion_segments))
        written.append(path)
    return written
