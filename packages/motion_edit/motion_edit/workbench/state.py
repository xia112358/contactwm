from __future__ import annotations

from pathlib import Path

from motion_edit.layers import group_by_motion, iter_layer_files, read_layer, write_layer
from motion_edit.paths import LAYERS_ROOT
from motion_edit.schema import SegmentRecord


def _source_path(source: str | Path, *, layers_root: Path = LAYERS_ROOT) -> Path:
    path = Path(source).expanduser()
    if path.is_absolute():
        return path
    return layers_root / path


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


def load_workbench_segments(source: str | Path, *, layers_root: Path = LAYERS_ROOT) -> list[SegmentRecord]:
    source_path = _source_path(source, layers_root=layers_root)
    if source_path.is_file():
        return read_layer(
            source_path,
            default_source=source_path.stem,
            default_status=_default_status_for(source_path),
        )
    segments: list[SegmentRecord] = []
    for path in iter_layer_files(source_path):
        segments.extend(
            read_layer(
                path,
                default_source=source_path.name,
                default_status=_default_status_for(source_path),
            )
        )
    return segments


def select_segment(
    segments: list[SegmentRecord],
    *,
    motion_id: str | None = None,
    segment_id: str | None = None,
    index: int | None = None,
) -> SegmentRecord:
    if segment_id is not None:
        matches = [segment for segment in segments if segment.segment_id == segment_id]
    else:
        if motion_id is None:
            raise ValueError("motion_id is required when selecting by index")
        matches = group_by_motion(segments).get(motion_id, [])
        if index is None:
            if len(matches) != 1:
                raise ValueError(f"motion_id={motion_id} matched {len(matches)} segments; pass --index or --segment-id")
            index = 0
        matches = [matches[index]] if 0 <= index < len(matches) else []
    if not matches:
        raise ValueError("no segment matched the workbench selection")
    if len(matches) > 1:
        raise ValueError(f"selection matched {len(matches)} segments; use motion_id/index to disambiguate")
    return matches[0]


def replace_segment(
    segments: list[SegmentRecord],
    old_segment_id: str,
    replacements: list[SegmentRecord],
) -> list[SegmentRecord]:
    updated: list[SegmentRecord] = []
    replaced = False
    for segment in segments:
        if segment.segment_id == old_segment_id:
            updated.extend(replacements)
            replaced = True
        else:
            updated.append(segment)
    if not replaced:
        raise ValueError(f"segment not found for replacement: {old_segment_id}")
    return updated


def validate_workbench_segments(segments: list[SegmentRecord], *, destination: str | Path | None = None) -> list[str]:
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for segment in segments:
        segment.validate()
        if segment.segment_id in seen_ids:
            warnings.append(f"duplicate segment_id: {segment.segment_id}")
        seen_ids.add(segment.segment_id)
        if not segment.motion_path and not segment.clip_npz:
            warnings.append(f"{segment.segment_id}: missing motion_path/clip_npz")

    destination_text = str(destination or "")
    if destination_text.startswith("accepted/") or "/accepted/" in destination_text:
        by_motion = group_by_motion(sorted(segments, key=lambda item: (item.motion_id, item.start_frame, item.end_frame)))
        for motion_id, items in by_motion.items():
            previous: SegmentRecord | None = None
            for segment in items:
                if previous is not None and segment.start_frame < previous.end_frame:
                    warnings.append(
                        f"{motion_id}: accepted overlap {previous.segment_id} "
                        f"[{previous.start_frame}, {previous.end_frame}] and "
                        f"{segment.segment_id} [{segment.start_frame}, {segment.end_frame}]"
                    )
                previous = segment
    return warnings


def write_workbench_segments(
    destination: str | Path,
    segments: list[SegmentRecord],
    *,
    layers_root: Path = LAYERS_ROOT,
) -> Path:
    validate_workbench_segments(segments, destination=destination)
    root = _source_path(destination, layers_root=layers_root)
    root.mkdir(parents=True, exist_ok=True)
    for motion_id, items in group_by_motion(segments).items():
        write_layer(root / f"{motion_id}.jsonl", items)
    return root


def upsert_workbench_segments(
    destination: str | Path,
    segments: list[SegmentRecord],
    *,
    layers_root: Path = LAYERS_ROOT,
) -> Path:
    root = _source_path(destination, layers_root=layers_root)
    existing = load_workbench_segments(root, layers_root=layers_root) if root.exists() else []
    by_id = {segment.segment_id: segment for segment in existing}
    for segment in segments:
        by_id[segment.segment_id] = segment
    merged = sorted(
        by_id.values(),
        key=lambda segment: (segment.motion_id, segment.start_frame, segment.end_frame, segment.segment_id),
    )
    validate_workbench_segments(merged, destination=destination)
    return write_workbench_segments(root, merged, layers_root=layers_root)
