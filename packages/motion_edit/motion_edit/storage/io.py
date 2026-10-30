from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from somaforge_core.published_data import load_published_json, resolve_published_path

from motion_edit.io import read_jsonl, segment_from_dict, segment_to_dict, write_jsonl
from motion_edit.paths import MOTION_ASSETS_ROOT, MOTION_VERSIONS_ROOT, MOTIONS_ROOT, SEGMENTS_ROOT, TOKENS_ROOT
from motion_edit.schema import SegmentRecord
from motion_edit.storage.segments import with_segment_motion_version_id
from motion_edit.storage.schema import MotionAssetRecord, MotionVersionRecord, TokenRecord


def motion_asset_path(motion_asset_id: str) -> Path:
    return MOTIONS_ROOT / f"{motion_asset_id}.json"


def legacy_motion_asset_path(motion_asset_id: str) -> Path:
    return MOTION_ASSETS_ROOT / f"{motion_asset_id}.json"


def motion_version_path(motion_version_id: str) -> Path:
    return MOTION_VERSIONS_ROOT / f"{motion_version_id}.json"


def canonical_segment_path(motion_version_id: str) -> Path:
    return SEGMENTS_ROOT / f"{motion_version_id}.jsonl"


def canonical_segmentation_exists(motion_version_id: str) -> bool:
    return canonical_segment_path(motion_version_id).exists()


def canonical_history_dir(motion_version_id: str) -> Path:
    return SEGMENTS_ROOT / "history" / motion_version_id


def canonical_history_event_path(motion_version_id: str) -> Path:
    return SEGMENTS_ROOT / "history" / f"{motion_version_id}.events.jsonl"


def token_catalog_path(motion_version_id: str) -> Path:
    return TOKENS_ROOT / f"{motion_version_id}.jsonl"


def write_motion_asset(record: MotionAssetRecord, path: str | Path | None = None) -> Path:
    out = Path(path).expanduser() if path is not None else motion_asset_path(record.motion_asset_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_motion_asset(motion_asset_id: str, path: str | Path | None = None) -> MotionAssetRecord:
    if path is not None:
        source = Path(path).expanduser()
    else:
        source = motion_asset_path(motion_asset_id)
        if not source.exists():
            source = legacy_motion_asset_path(motion_asset_id)
    data = load_published_json(source)
    return MotionAssetRecord(**data)


def list_motion_assets(root: str | Path | None = None) -> list[MotionAssetRecord]:
    if root is not None:
        source = resolve_published_path(root)
        if not source.exists():
            return []
        return [read_motion_asset(path.stem, path) for path in sorted(source.glob("*.json"))]
    records: dict[str, MotionAssetRecord] = {}
    for source in (legacy_motion_asset_path("__dummy__").parent, MOTIONS_ROOT):
        if not source.exists():
            continue
        for path in sorted(source.glob("*.json")):
            record = read_motion_asset(path.stem, path)
            records[record.motion_asset_id] = record
    return [records[key] for key in sorted(records)]


def write_motion_version(record: MotionVersionRecord, path: str | Path | None = None) -> Path:
    out = Path(path).expanduser() if path is not None else motion_version_path(record.motion_version_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_motion_version(motion_version_id: str, path: str | Path | None = None) -> MotionVersionRecord:
    source = Path(path).expanduser() if path is not None else motion_version_path(motion_version_id)
    data = load_published_json(source)
    return MotionVersionRecord(**data)


def list_motion_versions(root: str | Path | None = None) -> list[MotionVersionRecord]:
    source = resolve_published_path(root) if root is not None else MOTION_VERSIONS_ROOT
    if not source.exists():
        return []
    return [read_motion_version(path.stem, path) for path in sorted(source.glob("*.json"))]


def _mark_token_catalog_stale(motion_version_id: str, *, reason: str) -> None:
    try:
        version = read_motion_version(motion_version_id)
    except FileNotFoundError:
        return
    metadata = dict(version.metadata)
    metadata["token_catalog_status"] = "stale"
    metadata["token_catalog_stale_reason"] = reason
    write_motion_version(
        MotionVersionRecord(
            motion_version_id=version.motion_version_id,
            motion_path=version.motion_path,
            kind=version.kind,
            base_motion_id=version.base_motion_id,
            motion_asset_id=version.motion_asset_id,
            parent_motion_version_id=version.parent_motion_version_id,
            contact_layer=version.contact_layer,
            canonical_segment_path=version.canonical_segment_path,
            token_catalog_path=version.token_catalog_path,
            edit_plan_id=version.edit_plan_id,
            metadata=metadata,
        )
    )


def upsert_motion_version_canonical_path(
    motion_version_id: str,
    canonical_path: str | Path,
    *,
    contact_layer: str | None = None,
    motion_path: str | None = None,
) -> MotionVersionRecord:
    try:
        existing = read_motion_version(motion_version_id)
    except FileNotFoundError:
        if not motion_path:
            raise ValueError(f"{motion_version_id}: motion_path is required to create a MotionVersionRecord")
        existing = MotionVersionRecord(motion_version_id=motion_version_id, motion_path=motion_path)
    record = MotionVersionRecord(
        motion_version_id=existing.motion_version_id,
        motion_path=motion_path or existing.motion_path,
        kind=existing.kind,
        base_motion_id=existing.base_motion_id,
        motion_asset_id=existing.motion_asset_id,
        parent_motion_version_id=existing.parent_motion_version_id,
        contact_layer=contact_layer or existing.contact_layer,
        canonical_segment_path=str(canonical_path),
        token_catalog_path=existing.token_catalog_path,
        edit_plan_id=existing.edit_plan_id,
        metadata=dict(existing.metadata),
    )
    write_motion_version(record)
    return record


def _canonical_segment_records(motion_version_id: str, segments: Iterable[SegmentRecord]) -> list[dict]:
    records = []
    for segment in segments:
        updated = with_segment_motion_version_id(segment, motion_version_id)
        updated.validate()
        records.append(segment_to_dict(updated))
    return records


def write_canonical_history_event(
    motion_version_id: str,
    *,
    kind: str,
    source: str,
    reason: str | None = None,
    affected_segment_ids: list[str] | None = None,
) -> Path:
    path = canonical_history_event_path(motion_version_id)
    existing = read_jsonl(path)
    event = {
        "event_id": f"{motion_version_id}_{kind}_{len(existing):06d}_{uuid4().hex[:8]}",
        "motion_version_id": motion_version_id,
        "kind": kind,
        "source": source,
        "reason": reason,
        "affected_segment_ids": affected_segment_ids or [],
    }
    write_jsonl(path, [*existing, event])
    return path


def backup_canonical_segments(motion_version_id: str, *, reason: str | None = None, source: str = "canonical") -> Path | None:
    active = canonical_segment_path(motion_version_id)
    if not active.exists():
        return None
    events = read_jsonl(canonical_history_event_path(motion_version_id))
    backup = canonical_history_dir(motion_version_id) / f"{len(events):06d}.jsonl"
    records = read_jsonl(active)
    write_jsonl(backup, records)
    write_canonical_history_event(
        motion_version_id,
        kind="backup_canonical_segmentation",
        source=source,
        reason=reason,
        affected_segment_ids=[str(record.get("segment_id")) for record in records if record.get("segment_id")],
    )
    return backup


def write_canonical_segments(
    motion_version_id: str,
    segments: Iterable[SegmentRecord],
    path: str | Path | None = None,
    *,
    reason: str | None = None,
    source: str | None = None,
) -> Path:
    out = Path(path).expanduser() if path is not None else canonical_segment_path(motion_version_id)
    records = _canonical_segment_records(motion_version_id, segments)
    write_jsonl(out, records)
    if path is None and source is not None:
        write_canonical_history_event(
            motion_version_id,
            kind="write_canonical_segments",
            source=source,
            reason=reason,
            affected_segment_ids=[str(record["segment_id"]) for record in records],
        )
        _mark_token_catalog_stale(motion_version_id, reason="canonical_segmentation_updated")
    return out


def replace_canonical_segments(
    motion_version_id: str,
    segments: Iterable[SegmentRecord],
    *,
    reason: str,
    source: str,
    kind: str = "replace_canonical_segmentation",
    backup_existing: bool = True,
) -> Path:
    if backup_existing:
        backup_canonical_segments(motion_version_id, reason=reason, source=source)
    records = _canonical_segment_records(motion_version_id, segments)
    out = canonical_segment_path(motion_version_id)
    write_jsonl(out, records)
    write_canonical_history_event(
        motion_version_id,
        kind=kind,
        source=source,
        reason=reason,
        affected_segment_ids=[str(record["segment_id"]) for record in records],
    )
    _mark_token_catalog_stale(motion_version_id, reason="canonical_segmentation_updated")
    return out


def update_canonical_segments(
    motion_version_id: str,
    update_fn,
    *,
    reason: str,
    source: str,
    kind: str = "update_canonical_segmentation",
) -> list[SegmentRecord]:
    current = read_canonical_segments(motion_version_id)
    updated = list(update_fn(current))
    replace_canonical_segments(motion_version_id, updated, reason=reason, source=source, kind=kind)
    return updated


def read_canonical_segments(motion_version_id: str, path: str | Path | None = None) -> list[SegmentRecord]:
    source = Path(path).expanduser() if path is not None else canonical_segment_path(motion_version_id)
    return [segment_from_dict(record, default_source="canonical", default_status="candidate") for record in read_jsonl(source)]


def write_token_catalog(motion_version_id: str, tokens: Iterable[TokenRecord], path: str | Path | None = None) -> Path:
    out = Path(path).expanduser() if path is not None else token_catalog_path(motion_version_id)
    write_jsonl(out, (token.to_dict() for token in tokens))
    return out


def read_token_catalog(motion_version_id: str, path: str | Path | None = None) -> list[TokenRecord]:
    source = Path(path).expanduser() if path is not None else token_catalog_path(motion_version_id)
    return [TokenRecord(**record) for record in read_jsonl(source)]


def resolve_motion_path(motion_version_id: str) -> Path:
    version = read_motion_version(motion_version_id)
    return Path(version.motion_path).expanduser()
