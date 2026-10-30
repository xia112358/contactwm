from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from somaforge_core.published_data import relocate_published_values, resolve_published_path

from .schema import SegmentRecord


def read_jsonl(path: Path) -> list[dict]:
    path = resolve_published_path(path)
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return relocate_published_values(records)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def segment_from_dict(data: dict, *, default_source: str, default_status: str) -> SegmentRecord:
    motion_id = str(data.get("motion_id") or Path(str(data.get("clip_npz") or "")).stem)
    segment_id = str(data.get("segment_id") or f"{motion_id}_{int(data['start_frame']):04d}_{int(data['end_frame']):04d}")
    segment = SegmentRecord(
        motion_id=motion_id,
        segment_id=segment_id,
        start_frame=int(data["start_frame"]),
        end_frame=int(data["end_frame"]),
        source=str(data.get("source") or default_source),
        status=str(data.get("status") or default_status),  # type: ignore[arg-type]
        track=str(data.get("track") or "proto"),
        motion_path=data.get("motion_path"),
        clip_npz=data.get("clip_npz"),
        clip_output_dir=data.get("clip_output_dir"),
        clip_file_name=data.get("clip_file_name"),
        atom_label=str(data.get("atom_label") or ""),
        score=data.get("score"),
        contact_start=data.get("contact_start"),
        contact_end=data.get("contact_end"),
        active=data.get("active"),
        support=data.get("support"),
        metadata=dict(data.get("metadata") or {}),
    )
    segment.validate()
    return segment


def segment_to_dict(segment: SegmentRecord) -> dict:
    data = asdict(segment)
    return data
