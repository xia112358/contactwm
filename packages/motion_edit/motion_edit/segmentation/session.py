from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal
from uuid import uuid4

from motion_edit.io import read_jsonl, segment_from_dict, segment_to_dict, write_jsonl
from motion_edit.paths import WORKBENCH_ROOT
from motion_edit.schema import SegmentRecord
from motion_edit.storage.io import (
    canonical_segment_path,
    read_canonical_segments,
    read_motion_version,
    replace_canonical_segments,
)
from motion_edit.workbench.actions import trim_segment
from motion_edit.workbench.state import replace_segment, select_segment

SessionState = Literal["open", "saved", "discarded"]
SegmentStatus = Literal["candidate", "accepted", "rejected", "manual"]


@dataclass(frozen=True)
class SegmentationEditSession:
    session_id: str
    motion_version_id: str
    motion_path: str
    base_segment_path: str
    draft_segment_path: str
    edits_path: str
    state: SessionState = "open"
    selected_segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.motion_version_id:
            raise ValueError(f"{self.session_id}: motion_version_id is required")
        return asdict(self)


def _sessions_root(workbench_root: str | Path = WORKBENCH_ROOT) -> Path:
    return Path(workbench_root).expanduser() / "segmentation_sessions"


def _session_dir(session_id: str, *, workbench_root: str | Path = WORKBENCH_ROOT) -> Path:
    return _sessions_root(workbench_root) / session_id


def _resolve_session_path(session: str | Path, *, workbench_root: str | Path = WORKBENCH_ROOT) -> Path:
    path = Path(session).expanduser()
    if path.suffix == ".json":
        return path
    if path.is_absolute() or path.parent != Path("."):
        return path / "session.json" if path.is_dir() or path.suffix == "" else path
    return _session_dir(str(session), workbench_root=workbench_root) / "session.json"


def _write_session(session: SegmentationEditSession, path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(session.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_segmentation_edit_session(session: str | Path, *, workbench_root: str | Path = WORKBENCH_ROOT) -> SegmentationEditSession:
    path = _resolve_session_path(session, workbench_root=workbench_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    return SegmentationEditSession(**data)


def _ensure_open(session: SegmentationEditSession) -> None:
    if session.state != "open":
        raise ValueError(f"segmentation edit session {session.session_id} is {session.state}, not open")


def _segments_to_records(segments: Iterable[SegmentRecord]) -> list[dict[str, Any]]:
    records = []
    for segment in segments:
        segment.validate()
        records.append(segment_to_dict(segment))
    return records


def read_draft_segments(session: SegmentationEditSession | str | Path, *, workbench_root: str | Path = WORKBENCH_ROOT) -> list[SegmentRecord]:
    if not isinstance(session, SegmentationEditSession):
        session = read_segmentation_edit_session(session, workbench_root=workbench_root)
    return [
        segment_from_dict(record, default_source="segmentation_editor", default_status="candidate")
        for record in read_jsonl(Path(session.draft_segment_path))
    ]


def _write_draft_segments(session: SegmentationEditSession, segments: Iterable[SegmentRecord], *, allow_overlap: bool = False) -> list[str]:
    items = list(segments)
    warnings = validate_segmentation_segments(items, allow_overlap=allow_overlap)
    write_jsonl(Path(session.draft_segment_path), _segments_to_records(items))
    return warnings


def _append_session_edit(
    session: SegmentationEditSession,
    *,
    kind: str,
    affected_segment_ids: list[str],
    params: dict[str, Any],
    reason: str | None = None,
) -> Path:
    path = Path(session.edits_path)
    existing = read_jsonl(path)
    event = {
        "edit_id": f"{session.session_id}_{len(existing):06d}_{uuid4().hex[:8]}",
        "session_id": session.session_id,
        "motion_version_id": session.motion_version_id,
        "kind": kind,
        "source": "segmentation_editor",
        "affected_segment_ids": affected_segment_ids,
        "params": params,
    }
    if reason is not None:
        event["reason"] = reason
    write_jsonl(path, [*existing, event])
    return path


def _with_segment_edit_metadata(
    segment: SegmentRecord,
    *,
    kind: str,
    params: dict[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    metadata = dict(segment.metadata)
    edits = list(metadata.get("motion_edit_edits") or [])
    edit = {
        "kind": kind,
        "source": "segmentation_editor",
        "parent_segment_id": segment.segment_id,
        "params": params,
    }
    if reason is not None:
        edit["reason"] = reason
    edits.append(edit)
    metadata["motion_edit_edits"] = edits
    metadata["cut_source"] = "manual_refined"
    review = dict(metadata.get("review") or {})
    review["state"] = "edited"
    if reason is not None:
        review["note"] = reason
    metadata["review"] = review
    return metadata


def _single_motion_id(segments: list[SegmentRecord]) -> str:
    motion_ids = sorted({segment.motion_id for segment in segments})
    if len(motion_ids) != 1:
        raise ValueError("motion_id is required when the draft contains multiple motions")
    return motion_ids[0]


def _unique_segment_id(base: str, segments: list[SegmentRecord]) -> str:
    existing = {segment.segment_id for segment in segments}
    if base not in existing:
        return base
    index = 1
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def create_segmentation_edit_session(
    motion_version_id: str,
    *,
    session_id: str | None = None,
    overwrite: bool = False,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentationEditSession:
    version = read_motion_version(motion_version_id)
    segments = read_canonical_segments(motion_version_id)
    session_id = session_id or f"{motion_version_id}_seg_{uuid4().hex[:8]}"
    root = _session_dir(session_id, workbench_root=workbench_root)
    if root.exists() and not overwrite:
        raise FileExistsError(f"segmentation edit session already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    draft_segment_path = root / "draft_segments.jsonl"
    edits_path = root / "edits.jsonl"
    write_jsonl(draft_segment_path, _segments_to_records(segments))
    write_jsonl(edits_path, [])
    session = SegmentationEditSession(
        session_id=session_id,
        motion_version_id=motion_version_id,
        motion_path=version.motion_path,
        base_segment_path=str(canonical_segment_path(motion_version_id)),
        draft_segment_path=str(draft_segment_path),
        edits_path=str(edits_path),
        metadata={"base_segment_count": len(segments)},
    )
    _write_session(session, root / "session.json")
    return session


def validate_segmentation_segments(segments: list[SegmentRecord], *, allow_overlap: bool = False) -> list[str]:
    warnings: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        segment.validate()
        if segment.segment_id in seen:
            raise ValueError(f"duplicate segment_id: {segment.segment_id}")
        seen.add(segment.segment_id)
        if segment.status == "accepted":
            metadata = segment.metadata or {}
            if not (metadata.get("active_body") or segment.active):
                warnings.append(f"{segment.segment_id}: accepted segment has no active body")
            if not (metadata.get("support_bodies") or segment.support):
                warnings.append(f"{segment.segment_id}: accepted segment has no support bodies")
    by_motion: dict[str, list[SegmentRecord]] = {}
    for segment in segments:
        by_motion.setdefault(segment.motion_id, []).append(segment)
    for motion_id, items in by_motion.items():
        ordered = sorted(items, key=lambda item: (item.start_frame, item.end_frame, item.segment_id))
        previous: SegmentRecord | None = None
        for segment in ordered:
            if previous is not None:
                if segment.start_frame < previous.end_frame:
                    message = (
                        f"{motion_id}: overlap {previous.segment_id} "
                        f"[{previous.start_frame}, {previous.end_frame}] and "
                        f"{segment.segment_id} [{segment.start_frame}, {segment.end_frame}]"
                    )
                    if allow_overlap:
                        warnings.append(message)
                    else:
                        raise ValueError(message)
                elif segment.start_frame > previous.end_frame:
                    warnings.append(
                        f"{motion_id}: gap [{previous.end_frame}, {segment.start_frame}] "
                        f"between {previous.segment_id} and {segment.segment_id}"
                    )
            previous = segment
    return warnings


def list_draft_segments(
    session: SegmentationEditSession | str | Path,
    *,
    motion_id: str | None = None,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> list[SegmentRecord]:
    segments = read_draft_segments(session, workbench_root=workbench_root)
    if motion_id is not None:
        segments = [segment for segment in segments if segment.motion_id == motion_id]
    return sorted(segments, key=lambda item: (item.motion_id, item.start_frame, item.end_frame, item.segment_id))


def trim_draft_segment(
    session_ref: SegmentationEditSession | str | Path,
    *,
    segment_id: str | None = None,
    motion_id: str | None = None,
    index: int | None = None,
    start_frame: int,
    end_frame: int,
    reason: str | None = None,
    allow_overlap: bool = False,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentRecord:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    _ensure_open(session)
    segments = read_draft_segments(session)
    selected = select_segment(segments, motion_id=motion_id, segment_id=segment_id, index=index)
    trimmed = trim_segment(selected, start_frame=start_frame, end_frame=end_frame, allow_extend=True)
    metadata = _with_segment_edit_metadata(
        trimmed,
        kind="boundary_edit",
        reason=reason,
        params={
            "old_start_frame": selected.start_frame,
            "old_end_frame": selected.end_frame,
            "new_start_frame": int(start_frame),
            "new_end_frame": int(end_frame),
        },
    )
    trimmed = replace(trimmed, source="segmentation_editor", status="manual", metadata=metadata)
    updated = replace_segment(segments, selected.segment_id, [trimmed])
    _write_draft_segments(session, updated, allow_overlap=allow_overlap)
    _append_session_edit(
        session,
        kind="boundary_edit",
        affected_segment_ids=[trimmed.segment_id],
        reason=reason,
        params={
            "old_segment_id": selected.segment_id,
            "old_start_frame": selected.start_frame,
            "old_end_frame": selected.end_frame,
            "new_start_frame": int(start_frame),
            "new_end_frame": int(end_frame),
        },
    )
    return trimmed


def add_draft_segment(
    session_ref: SegmentationEditSession | str | Path,
    *,
    motion_id: str | None = None,
    segment_id: str | None = None,
    start_frame: int,
    end_frame: int,
    active: str | None = None,
    support: str | None = None,
    active_body: str | None = None,
    support_bodies: list[str] | None = None,
    transition_type: str | None = None,
    status: SegmentStatus = "manual",
    reason: str | None = None,
    allow_overlap: bool = False,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentRecord:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    _ensure_open(session)
    segments = read_draft_segments(session)
    motion_id = motion_id or _single_motion_id(segments)
    base_id = segment_id or f"{motion_id}_manual_{int(start_frame):04d}_{int(end_frame):04d}"
    segment_id = _unique_segment_id(base_id, segments)
    support_bodies = list(support_bodies or [])
    metadata: dict[str, Any] = {
        "motion_version_id": session.motion_version_id,
        "cut_source": "manual_add",
        "review": {"state": "edited", "note": reason or "manual add"},
        "motion_edit_edits": [
            {
                "kind": "add_segment",
                "source": "segmentation_editor",
                "parent_segment_id": None,
                "params": {
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "active_body": active_body,
                    "support_bodies": support_bodies,
                    "transition_type": transition_type,
                },
                **({"reason": reason} if reason is not None else {}),
            }
        ],
    }
    if active_body is not None:
        metadata["active_body"] = active_body
    if support_bodies:
        metadata["support_bodies"] = support_bodies
    if transition_type is not None:
        metadata["transition_type"] = transition_type
    segment = SegmentRecord(
        motion_id=motion_id,
        segment_id=segment_id,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        source="segmentation_editor",
        status=status,
        track="manual",
        motion_path=session.motion_path,
        clip_npz=session.motion_path,
        active=active or active_body,
        support=support or (",".join(support_bodies) if support_bodies else None),
        metadata=metadata,
    )
    segment.validate()
    updated = sorted([*segments, segment], key=lambda item: (item.motion_id, item.start_frame, item.end_frame, item.segment_id))
    _write_draft_segments(session, updated, allow_overlap=allow_overlap)
    _append_session_edit(
        session,
        kind="add_segment",
        affected_segment_ids=[segment.segment_id],
        reason=reason,
        params={
            "segment_id": segment.segment_id,
            "motion_id": motion_id,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "active_body": active_body,
            "support_bodies": support_bodies,
            "transition_type": transition_type,
        },
    )
    return segment


def delete_draft_segment(
    session_ref: SegmentationEditSession | str | Path,
    *,
    segment_id: str | None = None,
    motion_id: str | None = None,
    index: int | None = None,
    reason: str | None = None,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentRecord:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    _ensure_open(session)
    segments = read_draft_segments(session)
    selected = select_segment(segments, motion_id=motion_id, segment_id=segment_id, index=index)
    updated = [segment for segment in segments if segment.segment_id != selected.segment_id]
    _write_draft_segments(session, updated)
    _append_session_edit(
        session,
        kind="delete_segment",
        affected_segment_ids=[selected.segment_id],
        reason=reason,
        params={"deleted_segment": segment_to_dict(selected)},
    )
    return selected


def relabel_draft_segment(
    session_ref: SegmentationEditSession | str | Path,
    *,
    segment_id: str | None = None,
    motion_id: str | None = None,
    index: int | None = None,
    active: str | None = None,
    support: str | None = None,
    active_body: str | None = None,
    support_bodies: list[str] | None = None,
    transition_type: str | None = None,
    status: SegmentStatus | None = None,
    reason: str | None = None,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentRecord:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    _ensure_open(session)
    segments = read_draft_segments(session)
    selected = select_segment(segments, motion_id=motion_id, segment_id=segment_id, index=index)
    support_bodies = list(support_bodies or [])
    metadata = _with_segment_edit_metadata(
        selected,
        kind="relabel_segment",
        reason=reason,
        params={
            "old_active": selected.active,
            "old_support": selected.support,
            "old_active_body": selected.metadata.get("active_body"),
            "old_support_bodies": selected.metadata.get("support_bodies"),
            "old_transition_type": selected.metadata.get("transition_type"),
            "new_active": active,
            "new_support": support,
            "new_active_body": active_body,
            "new_support_bodies": support_bodies,
            "new_transition_type": transition_type,
        },
    )
    if active_body is not None:
        metadata["active_body"] = active_body
    if support_bodies:
        metadata["support_bodies"] = support_bodies
    if transition_type is not None:
        metadata["transition_type"] = transition_type
    relabeled = replace(
        selected,
        source="segmentation_editor",
        status=status or "manual",
        active=active if active is not None else active_body if active_body is not None else selected.active,
        support=support if support is not None else ",".join(support_bodies) if support_bodies else selected.support,
        metadata=metadata,
    )
    relabeled.validate()
    updated = replace_segment(segments, selected.segment_id, [relabeled])
    _write_draft_segments(session, updated)
    _append_session_edit(
        session,
        kind="relabel_segment",
        affected_segment_ids=[relabeled.segment_id],
        reason=reason,
        params={
            "active": relabeled.active,
            "support": relabeled.support,
            "active_body": metadata.get("active_body"),
            "support_bodies": metadata.get("support_bodies"),
            "transition_type": metadata.get("transition_type"),
            "status": relabeled.status,
        },
    )
    return relabeled


def save_segmentation_edit_session(
    session_ref: SegmentationEditSession | str | Path,
    *,
    reason: str | None = None,
    allow_overlap: bool = False,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> Path:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    _ensure_open(session)
    segments = read_draft_segments(session)
    warnings = validate_segmentation_segments(segments, allow_overlap=allow_overlap)
    out = replace_canonical_segments(
        session.motion_version_id,
        segments,
        reason=reason or f"save segmentation edit session {session.session_id}",
        source="segmentation_editor",
        kind="segmentation_edit_session_save",
    )
    updated = replace(
        session,
        state="saved",
        metadata={**dict(session.metadata), "save_warnings": warnings, "saved_canonical_path": str(out)},
    )
    _write_session(updated, _resolve_session_path(session.session_id, workbench_root=workbench_root))
    _append_session_edit(
        updated,
        kind="save_session",
        affected_segment_ids=[segment.segment_id for segment in segments],
        reason=reason,
        params={"canonical_path": str(out), "warnings": warnings},
    )
    return out


def discard_segmentation_edit_session(
    session_ref: SegmentationEditSession | str | Path,
    *,
    reason: str | None = None,
    workbench_root: str | Path = WORKBENCH_ROOT,
) -> SegmentationEditSession:
    session = session_ref if isinstance(session_ref, SegmentationEditSession) else read_segmentation_edit_session(session_ref, workbench_root=workbench_root)
    if session.state != "open":
        raise ValueError(f"segmentation edit session {session.session_id} is {session.state}, cannot discard")
    updated = replace(session, state="discarded", metadata={**dict(session.metadata), "discard_reason": reason})
    _write_session(updated, _resolve_session_path(session.session_id, workbench_root=workbench_root))
    _append_session_edit(
        updated,
        kind="discard_session",
        affected_segment_ids=[],
        reason=reason,
        params={},
    )
    return updated
