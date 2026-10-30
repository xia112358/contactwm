from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LayerStatus = Literal["candidate", "accepted", "rejected", "manual"]


@dataclass(frozen=True)
class MotionRef:
    motion_id: str
    path: str
    fps: float | None = None
    terrain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditRecord:
    edit_id: str
    kind: str
    source: str
    params: dict[str, Any] = field(default_factory=dict)
    parent_motion_id: str | None = None
    output_motion_id: str | None = None


@dataclass(frozen=True)
class SegmentRecord:
    motion_id: str
    segment_id: str
    start_frame: int
    end_frame: int
    source: str
    status: LayerStatus = "candidate"
    track: str = "proto"
    motion_path: str | None = None
    clip_npz: str | None = None
    clip_output_dir: str | None = None
    clip_file_name: str | None = None
    atom_label: str = ""
    score: float | None = None
    contact_start: str | None = None
    contact_end: str | None = None
    active: str | None = None
    support: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"{self.segment_id}: start_frame must be >= 0")
        if self.end_frame <= self.start_frame:
            raise ValueError(f"{self.segment_id}: end_frame must be > start_frame")

    def to_cutter_json(self) -> dict[str, Any]:
        clip_npz = self.clip_npz or self.motion_path or ""
        clip_path = Path(clip_npz) if clip_npz else None
        return {
            "motion_id": self.motion_id,
            "segment_id": self.segment_id,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "atom_label": self.atom_label,
            "clip_npz": str(clip_npz),
            "clip_output_dir": str(self.clip_output_dir or ""),
            "clip_file_name": self.clip_file_name or (clip_path.name if clip_path else ""),
            "source": self.source,
            "status": self.status,
            "track": self.track,
            "score": self.score,
            "contact_start": self.contact_start,
            "contact_end": self.contact_end,
            "active": self.active,
            "support": self.support,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SegmentLayer:
    name: str
    source: str
    status: LayerStatus
    segments: list[SegmentRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
