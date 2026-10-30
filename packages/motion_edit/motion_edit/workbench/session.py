from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from motion_edit.io import segment_to_dict
from motion_edit.schema import SegmentRecord
from motion_edit.workbench.actions import curate_segment, split_segment, trim_segment
from motion_edit.workbench.state import (
    load_workbench_segments,
    replace_segment,
    select_segment,
    upsert_workbench_segments,
    write_workbench_segments,
)


@dataclass
class WorkbenchSession:
    source: str
    motion_path: str | None = None
    motion_id: str | None = None
    selected_segment_id: str | None = None
    selected_index: int | None = None
    current_frame: int = 0
    output_source: str = "manual/workbench_tmp"
    layer_name: str = "workbench_tmp"
    dry_run: bool = False
    layers_root: Path | None = None
    segments: list[SegmentRecord] = field(default_factory=list)
    last_write: str | None = None

    def __post_init__(self) -> None:
        root = self.layers_root if self.layers_root is not None else None
        if root is None:
            self.segments = load_workbench_segments(self.source)
        else:
            self.segments = load_workbench_segments(self.source, layers_root=root)
        if self.motion_id is None and self.motion_path:
            self.motion_id = Path(self.motion_path).stem
        selected = self.selected()
        self.current_frame = selected.start_frame if self.current_frame < 0 else self.current_frame

    def selected(self) -> SegmentRecord:
        return select_segment(
            self.segments,
            motion_id=self.motion_id,
            segment_id=self.selected_segment_id,
            index=self.selected_index,
        )

    def state(self) -> dict[str, Any]:
        selected = self.selected()
        motion_segments = [segment for segment in self.segments if segment.motion_id == selected.motion_id]
        return {
            "source": self.source,
            "motion_path": self.motion_path,
            "motion_id": selected.motion_id,
            "current_frame": self.current_frame,
            "selected_segment_id": selected.segment_id,
            "selected_index": self.selected_index,
            "output_source": self.output_source,
            "dry_run": self.dry_run,
            "last_write": self.last_write,
            "selected_segment": segment_to_dict(selected),
            "segments": [segment_to_dict(segment) for segment in motion_segments],
        }

    def set_selection(
        self,
        *,
        motion_id: str | None = None,
        segment_id: str | None = None,
        index: int | None = None,
        current_frame: int | None = None,
    ) -> dict[str, Any]:
        if motion_id is not None:
            self.motion_id = motion_id
        if segment_id is not None:
            self.selected_segment_id = segment_id
            self.selected_index = None
        if index is not None:
            self.selected_index = index
            self.selected_segment_id = None
        if current_frame is not None:
            self.current_frame = int(current_frame)
        self.selected()
        return self.state()

    def trim(self, *, start_frame: int, end_frame: int, output_source: str | None = None) -> dict[str, Any]:
        selected = self.selected()
        trimmed = trim_segment(selected, start_frame=start_frame, end_frame=end_frame)
        self.segments = replace_segment(self.segments, selected.segment_id, [trimmed])
        self.selected_segment_id = trimmed.segment_id
        self.selected_index = None
        self.current_frame = trimmed.start_frame
        self._write(output_source or self.output_source, self.segments)
        return self.state()

    def split(self, *, frame: int, output_source: str | None = None) -> dict[str, Any]:
        selected = self.selected()
        left, right = split_segment(selected, frame=frame)
        self.segments = replace_segment(self.segments, selected.segment_id, [left, right])
        self.selected_segment_id = left.segment_id
        self.selected_index = None
        self.current_frame = left.start_frame
        self._write(output_source or self.output_source, self.segments)
        return self.state()

    def accept(self, *, output_source: str | None = None) -> dict[str, Any]:
        return self._curate("accepted", output_source=output_source)

    def reject(self, *, output_source: str | None = None) -> dict[str, Any]:
        return self._curate("rejected", output_source=output_source)

    def _curate(self, status: str, *, output_source: str | None = None) -> dict[str, Any]:
        selected = self.selected()
        curated = curate_segment(selected, status=status)  # type: ignore[arg-type]
        destination = output_source or f"{status}/{self.layer_name}"
        self.segments = replace_segment(self.segments, selected.segment_id, [curated])
        self.selected_segment_id = curated.segment_id
        self.selected_index = None
        self.last_write = None
        self._upsert(destination, [curated])
        return self.state()

    def _write(self, destination: str, segments: list[SegmentRecord]) -> None:
        if self.dry_run:
            self.last_write = f"dry-run:{destination}"
            return
        if self.layers_root is None:
            out = write_workbench_segments(destination, segments)
        else:
            out = write_workbench_segments(destination, segments, layers_root=self.layers_root)
        self.last_write = str(out)

    def _upsert(self, destination: str, segments: list[SegmentRecord]) -> None:
        if self.dry_run:
            self.last_write = f"dry-run:{destination}"
            return
        if self.layers_root is None:
            out = upsert_workbench_segments(destination, segments)
        else:
            out = upsert_workbench_segments(destination, segments, layers_root=self.layers_root)
        self.last_write = str(out)
