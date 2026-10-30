from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motion_edit.paths import WORKBENCH_ROOT


RECENT_MOTIONS_PATH = WORKBENCH_ROOT / "recent_motions.json"


@dataclass(frozen=True)
class RecentMotionEntry:
    label: str
    motion_path: str
    motion_id: str
    motion_ref_id: str | None = None
    motion_asset_id: str | None = None
    motion_asset_path: str | None = None
    motion_version_id: str | None = None
    terrain_urdf: str | None = None
    contact_layer: str | None = None
    surface_catalog: str | None = None
    edit_plan_path: str | None = None
    output_contact_layer: str | None = None
    output_segment_layer: str | None = None
    last_opened_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        if self.motion_ref_id:
            return self.motion_ref_id
        if self.motion_version_id:
            return f"version:{self.motion_version_id}"
        return self.motion_asset_path or self.motion_asset_id or self.motion_path

    def exists(self) -> bool:
        if self.motion_asset_path and Path(self.motion_asset_path).expanduser().exists():
            return True
        return Path(self.motion_path).expanduser().exists()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_recent_motions(path: str | Path = RECENT_MOTIONS_PATH, *, prune_missing: bool = True) -> list[RecentMotionEntry]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    data = json.loads(source.read_text(encoding="utf-8"))
    items = []
    for item in data.get("items", []):
        payload = dict(item)
        payload.setdefault(
            "motion_ref_id",
            payload.get("motion_version_id")
            or payload.get("motion_asset_id")
            or payload.get("motion_id"),
        )
        items.append(RecentMotionEntry(**payload))
    if prune_missing:
        items = [item for item in items if item.exists()]
    return items


def write_recent_motions(
    entries: list[RecentMotionEntry],
    path: str | Path = RECENT_MOTIONS_PATH,
    *,
    limit: int = 20,
) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    deduped: dict[str, RecentMotionEntry] = {}
    for entry in entries:
        deduped[entry.key()] = entry
    ordered = sorted(
        deduped.values(),
        key=lambda item: item.last_opened_at or "",
        reverse=True,
    )[:limit]
    out.write_text(
        json.dumps({"schema_version": 2, "items": [item.to_dict() for item in ordered]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def clear_recent_motions(
    path: str | Path = RECENT_MOTIONS_PATH,
) -> Path:
    """Start a new editor-session history without deleting the cache file."""

    return write_recent_motions([], path)


def upsert_recent_motion(
    entry: RecentMotionEntry,
    path: str | Path = RECENT_MOTIONS_PATH,
    *,
    limit: int = 20,
) -> list[RecentMotionEntry]:
    current = read_recent_motions(path, prune_missing=True)
    stamped = RecentMotionEntry(**{**entry.to_dict(), "last_opened_at": _now_iso()})
    current = [item for item in current if item.key() != stamped.key()]
    current.insert(0, stamped)
    write_recent_motions(current, path, limit=limit)
    return read_recent_motions(path, prune_missing=True)


def recent_entry_labels(entries: list[RecentMotionEntry]) -> list[str]:
    labels: list[str] = []
    used: set[str] = set()
    for entry in entries:
        label = entry.label or Path(entry.motion_path).name
        if label in used:
            label = f"{label} [{entry.motion_id}]"
        used.add(label)
        labels.append(label)
    return labels
