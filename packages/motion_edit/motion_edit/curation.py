from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from motion_edit.layers import group_by_motion, iter_layer_files, read_layer, write_layer
from motion_edit.paths import LAYERS_ROOT
from motion_edit.schema import SegmentRecord


def _default_status_for(path: Path) -> str:
    if "candidates" in path.parts:
        return "candidate"
    if "accepted" in path.parts:
        return "accepted"
    if "rejected" in path.parts:
        return "rejected"
    if "manual" in path.parts:
        return "manual"
    return "candidate"


def load_layer_segments(source: str) -> list[SegmentRecord]:
    source_path = LAYERS_ROOT / source
    if source_path.is_file():
        return read_layer(source_path, default_source=source_path.stem, default_status=_default_status_for(source_path))
    segments: list[SegmentRecord] = []
    for path in iter_layer_files(source_path):
        segments.extend(read_layer(path, default_source=source_path.name, default_status=_default_status_for(source_path)))
    return segments


def filter_segments(
    segments: list[SegmentRecord],
    *,
    motion_id: str | None = None,
    segment_ids: set[str] | None = None,
    indices: set[int] | None = None,
) -> list[SegmentRecord]:
    by_motion = group_by_motion(segments)
    selected: list[SegmentRecord] = []
    for mid, items in by_motion.items():
        if motion_id is not None and mid != motion_id:
            continue
        for index, segment in enumerate(items):
            if segment_ids is not None and segment.segment_id not in segment_ids:
                continue
            if indices is not None and index not in indices:
                continue
            selected.append(segment)
    return selected


def _with_curation_edit(segment: SegmentRecord, *, status: str, layer_name: str) -> SegmentRecord:
    metadata = dict(segment.metadata)
    edits = list(metadata.get("motion_edit_edits") or [])
    edits.append(
        {
            "kind": "curate",
            "source": "cli",
            "parent_segment_id": segment.segment_id,
            "params": {
                "old_status": segment.status,
                "new_status": status,
                "layer_name": layer_name,
            },
        }
    )
    metadata["motion_edit_edits"] = edits
    return replace(segment, status=status, metadata=metadata)


def write_status_layer(status: str, layer_name: str, segments: list[SegmentRecord]) -> Path:
    root = LAYERS_ROOT / status / layer_name
    root.mkdir(parents=True, exist_ok=True)
    changed = [_with_curation_edit(segment, status=status, layer_name=layer_name) for segment in segments]
    by_motion = group_by_motion(changed)
    for motion_id, items in by_motion.items():
        path = root / f"{motion_id}.jsonl"
        existing = read_layer(path, default_source=layer_name, default_status=status) if path.exists() else []
        merged = {segment.segment_id: segment for segment in existing}
        for segment in items:
            merged[segment.segment_id] = segment
        write_layer(
            path,
            sorted(
                merged.values(),
                key=lambda segment: (segment.motion_id, segment.start_frame, segment.end_frame, segment.segment_id),
            ),
        )
    return root
