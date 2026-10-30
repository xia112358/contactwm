from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from motion_edit.io import read_jsonl, segment_from_dict, write_jsonl
from motion_edit.paths import LAYERS_ROOT, WORKBENCH_ROOT
from motion_edit.schema import SegmentRecord
from motion_edit.contact import bind_segment_to_contact_graph, read_contact_graph
from motion_edit.export import export_contact_overlay
from motion_edit.workbench.state import load_workbench_segments, upsert_workbench_segments, write_workbench_segments


@dataclass(frozen=True)
class CutterSession:
    session_name: str
    motion_id: str
    source_layer: str
    segment_path: Path
    destination_layer: str
    manifest_path: Path | None = None
    contact_overlay_path: Path | None = None


def _session_dir(session_name: str, *, workbench_root: Path = WORKBENCH_ROOT) -> Path:
    return workbench_root / "sessions" / session_name


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_session_manifest(
    *,
    session_name: str,
    motion_id: str,
    source_layer: str,
    destination_layer: str,
    segment_path: Path,
    manifest_path: Path,
    motion_path: str | None = None,
    contact_overlay_path: Path | None = None,
    ui_port: int | None = None,
    sync_status: str = "prepared",
) -> None:
    existing: dict = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    now = _utc_now()
    manifest = {
        **existing,
        "session_name": session_name,
        "motion_id": motion_id,
        "motion_path": motion_path or existing.get("motion_path"),
        "source_layer": source_layer,
        "working_cutter_file": str(segment_path),
        "contact_overlay_file": str(contact_overlay_path) if contact_overlay_path is not None else existing.get("contact_overlay_file"),
        "output_layer": destination_layer,
        "ui_port": ui_port if ui_port is not None else existing.get("ui_port"),
        "sync_status": sync_status,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _cutter_metadata(segment: SegmentRecord, *, source_layer: str, session_name: str) -> dict:
    metadata = dict(segment.metadata)
    metadata["motion_edit_cutter_session"] = {
        "source_layer": source_layer,
        "session_name": session_name,
        "original_segment_id": segment.segment_id,
        "edit_source": "motion_edit_export",
    }
    return metadata


def _source_contact_layer(source_layer: str, *, layers_root: Path) -> Path:
    return layers_root / "contact" / Path(source_layer).name


def _export_session_contact_overlay(
    *,
    motion_id: str,
    source_layer: str,
    session_dir: Path,
    layers_root: Path,
) -> Path | None:
    contact_layer = _source_contact_layer(source_layer, layers_root=layers_root)
    if not contact_layer.exists():
        return None
    graph = read_contact_graph(contact_layer, motion_id)
    out = session_dir / f"{motion_id}.contact_overlay.json"
    return export_contact_overlay(out, graph)


def _read_source_contact_graph(
    *,
    motion_id: str,
    source_layer: str,
    layers_root: Path,
):
    contact_layer = _source_contact_layer(source_layer, layers_root=layers_root)
    if not contact_layer.exists():
        return None
    return read_contact_graph(contact_layer, motion_id)


def export_cutter_session_file(
    *,
    motion_id: str,
    source_layer: str,
    session_name: str,
    destination_layer: str | None = None,
    motion_path: str | Path | None = None,
    ui_port: int | None = None,
    layers_root: Path = LAYERS_ROOT,
    workbench_root: Path = WORKBENCH_ROOT,
) -> CutterSession:
    source_segments = [
        segment for segment in load_workbench_segments(source_layer, layers_root=layers_root) if segment.motion_id == motion_id
    ]
    session_dir = _session_dir(session_name, workbench_root=workbench_root)
    segment_path = session_dir / f"{motion_id}.segments.jsonl"
    manifest_path = session_dir / "session.json"
    contact_overlay_path = _export_session_contact_overlay(
        motion_id=motion_id,
        source_layer=source_layer,
        session_dir=session_dir,
        layers_root=layers_root,
    )
    records = []
    for segment in source_segments:
        record = segment.to_cutter_json()
        record["metadata"] = _cutter_metadata(segment, source_layer=source_layer, session_name=session_name)
        records.append(record)
    write_jsonl(segment_path, records)
    source_motion_path = str(Path(motion_path).expanduser()) if motion_path is not None else None
    if source_segments:
        source_motion_path = source_motion_path or source_segments[0].motion_path or source_segments[0].clip_npz
    _write_session_manifest(
        session_name=session_name,
        motion_id=motion_id,
        source_layer=source_layer,
        destination_layer=destination_layer or f"manual/{session_name}",
        segment_path=segment_path,
        manifest_path=manifest_path,
        motion_path=source_motion_path,
        contact_overlay_path=contact_overlay_path,
        ui_port=ui_port,
        sync_status="prepared",
    )
    return CutterSession(
        session_name=session_name,
        motion_id=motion_id,
        source_layer=source_layer,
        segment_path=segment_path,
        destination_layer=destination_layer or f"manual/{session_name}",
        manifest_path=manifest_path,
        contact_overlay_path=contact_overlay_path,
    )


def _source_segment_ids(source_layer: str, *, motion_id: str, layers_root: Path) -> set[str]:
    return {
        segment.segment_id
        for segment in load_workbench_segments(source_layer, layers_root=layers_root)
        if segment.motion_id == motion_id
    }


def segments_from_cutter_file(
    segment_path: str | Path,
    *,
    source_layer: str,
    session_name: str,
    layers_root: Path = LAYERS_ROOT,
) -> list[SegmentRecord]:
    path = Path(segment_path).expanduser()
    records = read_jsonl(path)
    motion_id = path.name.removesuffix(".segments.jsonl")
    source_ids = _source_segment_ids(source_layer, motion_id=motion_id, layers_root=layers_root)
    contact_graph = _read_source_contact_graph(motion_id=motion_id, source_layer=source_layer, layers_root=layers_root)
    segments: list[SegmentRecord] = []
    for item in records:
        parsed = segment_from_dict(item, default_source="motion_edit_workbench", default_status="manual")
        existing_meta = dict(parsed.metadata)
        session_meta = dict(existing_meta.get("motion_edit_cutter_session") or {})
        original_segment_id = session_meta.get("original_segment_id")
        if original_segment_id is None and parsed.segment_id in source_ids:
            original_segment_id = parsed.segment_id
        existing_meta["motion_edit_cutter_session"] = {
            "source_layer": source_layer,
            "session_name": session_name,
            "original_segment_id": original_segment_id,
            "edit_source": "motion_edit_workbench",
        }
        parsed = replace(parsed, metadata=existing_meta)
        if contact_graph is not None:
            parsed = bind_segment_to_contact_graph(parsed, contact_graph)
        segments.append(
            replace(
                parsed,
                source="motion_edit_workbench",
                status="manual",
            )
        )
    return sorted(segments, key=lambda segment: (segment.motion_id, segment.start_frame, segment.end_frame, segment.segment_id))


def _with_import_edit(
    segment: SegmentRecord,
    *,
    source_layer: str,
    session_name: str,
    destination_layer: str,
    status: str,
) -> SegmentRecord:
    metadata = dict(segment.metadata)
    session_meta = dict(metadata.get("motion_edit_cutter_session") or {})
    edits = list(metadata.get("motion_edit_edits") or [])
    edits.append(
        {
            "kind": "import_from_cutter",
            "source": "motion_edit_workbench",
            "parent_segment_id": session_meta.get("original_segment_id"),
            "params": {
                "source_layer": source_layer,
                "session_name": session_name,
                "destination_layer": destination_layer,
                "old_status": segment.status,
                "new_status": status,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
            },
        }
    )
    metadata["motion_edit_edits"] = edits
    return replace(segment, status=status, metadata=metadata)


def sync_cutter_session_file(
    segment_path: str | Path,
    *,
    source_layer: str,
    session_name: str,
    destination_layer: str | None = None,
    layers_root: Path = LAYERS_ROOT,
    workbench_root: Path | None = None,
) -> Path:
    destination = destination_layer or f"manual/{session_name}"
    segments = segments_from_cutter_file(
        segment_path,
        source_layer=source_layer,
        session_name=session_name,
        layers_root=layers_root,
    )
    if destination.startswith("accepted/"):
        out = upsert_workbench_segments(
            destination,
            [
                _with_import_edit(
                    segment,
                    source_layer=source_layer,
                    session_name=session_name,
                    destination_layer=destination,
                    status="accepted",
                )
                for segment in segments
            ],
            layers_root=layers_root,
        )
    elif destination.startswith("rejected/"):
        out = upsert_workbench_segments(
            destination,
            [
                _with_import_edit(
                    segment,
                    source_layer=source_layer,
                    session_name=session_name,
                    destination_layer=destination,
                    status="rejected",
                )
                for segment in segments
            ],
            layers_root=layers_root,
        )
    else:
        out = write_workbench_segments(
            destination,
            [
                _with_import_edit(
                    segment,
                    source_layer=source_layer,
                    session_name=session_name,
                    destination_layer=destination,
                    status="manual",
                )
                for segment in segments
            ],
            layers_root=layers_root,
        )

    path = Path(segment_path).expanduser()
    motion_id = path.name.removesuffix(".segments.jsonl")
    _write_session_manifest(
        session_name=session_name,
        motion_id=motion_id,
        source_layer=source_layer,
        destination_layer=destination,
        segment_path=path,
        manifest_path=(_session_dir(session_name, workbench_root=workbench_root) if workbench_root else path.parent)
        / "session.json",
        sync_status="synced",
    )
    return out
