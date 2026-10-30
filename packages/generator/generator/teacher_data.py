"""Load full generated teacher trajectories and boundary-aligned events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from somaforge_core.motion_contracts import BODY_NAMES, CONTACT_PARTS, SparseKeyframe, TeacherSegment
from somaforge_core.motion_trajectory import FullBodyTeacherSegment, FullBodyTrajectory

EVENT_FIELDS = (
    "start_position",
    "start_rotation",
    "end_position",
    "end_rotation",
    "start_contact",
    "end_contact",
    "touchdown",
    "target_contact",
    "target_surface",
    "duration",
    "geometry",
    "action",
    "trajectory_id",
    "event_index",
)


class TeacherDataset:
    """Canonical access to generated trajectories, not the legacy 207 sources."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory).resolve()
        report_path = self.directory / "report.json"
        event_path = self.directory / "teacher_events.npz"
        if not report_path.is_file() or not event_path.is_file():
            raise FileNotFoundError(f"expected report.json and teacher_events.npz in {self.directory}")
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        if self.report.get("schema") != "climb00_privileged_wide_event_teacher_v1":
            raise ValueError(f"unsupported teacher schema: {self.report.get('schema')}")
        with np.load(event_path, allow_pickle=False) as loaded:
            missing = sorted(set(EVENT_FIELDS) - set(loaded.files))
            if missing:
                raise ValueError(f"teacher event cache is missing fields: {missing}")
            self.events = {name: np.asarray(loaded[name]) for name in EVENT_FIELDS}
        self._rows = tuple(self.report["samples"])
        trajectory_id = self.events["trajectory_id"].astype(np.int64)
        event_index = self.events["event_index"].astype(np.int64)
        self._event_rows: list[np.ndarray] = []
        for index, row in enumerate(self._rows):
            selected = np.flatnonzero(trajectory_id == index)
            selected = selected[np.argsort(event_index[selected])]
            expected = np.arange(len(selected), dtype=np.int64)
            if not np.array_equal(event_index[selected], expected):
                raise ValueError(f"trajectory {index} has a non-contiguous event sequence")
            if Path(row["motion_file"]).resolve().parent != self.directory:
                raise ValueError(f"trajectory {index} points outside the teacher directory")
            self._event_rows.append(selected)

    def __len__(self) -> int:
        return len(self.events["action"])

    @property
    def trajectory_count(self) -> int:
        return len(self._rows)

    def trajectory_path(self, trajectory_id: int) -> Path:
        return Path(self._rows[trajectory_id]["motion_file"]).resolve()

    def _load_trajectory(self, trajectory_id: int) -> dict[str, np.ndarray]:
        path = self.trajectory_path(trajectory_id)
        with np.load(path, allow_pickle=False) as loaded:
            result = {name: np.asarray(loaded[name]) for name in loaded.files}
        if tuple(result["body_names"].astype(str)) != BODY_NAMES:
            raise ValueError(f"unexpected sparse body order in {path}")
        if tuple(result["contact_force_part_order"].astype(str)) != CONTACT_PARTS:
            raise ValueError(f"unexpected contact part order in {path}")
        boundary = result["boundary_frames"].astype(np.int64)
        if len(boundary) != len(self._event_rows[trajectory_id]) + 1:
            raise ValueError(f"boundary/event count mismatch in {path}")
        if boundary[0] != 0 or boundary[-1] != len(result["body_pos_w"]) - 1:
            raise ValueError(f"invalid boundary extent in {path}")
        return result

    def segment(self, trajectory_id: int, event_index: int) -> TeacherSegment:
        trajectory = self._load_trajectory(trajectory_id)
        event_row = int(self._event_rows[trajectory_id][event_index])
        boundary = trajectory["boundary_frames"].astype(np.int64)
        first, last = int(boundary[event_index]), int(boundary[event_index + 1])
        position = trajectory["body_pos_w"][first : last + 1].astype(np.float32)
        quaternion = trajectory["body_quat_w"][first : last + 1].astype(np.float32)
        contact = trajectory["contact_force_part_mask"][first : last + 1].astype(bool)
        start = SparseKeyframe(
            self.events["start_position"][event_row].astype(np.float32),
            self.events["start_rotation"][event_row].astype(np.float32),
            self.events["start_contact"][event_row].astype(bool),
        )
        end = SparseKeyframe(
            self.events["end_position"][event_row].astype(np.float32),
            self.events["end_rotation"][event_row].astype(np.float32),
            self.events["end_contact"][event_row].astype(bool),
        )
        return TeacherSegment(
            trajectory_id=trajectory_id,
            event_index=event_index,
            action=int(self.events["action"][event_row]),
            fps=float(np.asarray(trajectory["fps"]).item()),
            position=position,
            quaternion=quaternion,
            contact=contact,
            start=start,
            end=end,
            touchdown=self.events["touchdown"][event_row].astype(bool),
            target_contact=self.events["target_contact"][event_row].astype(np.float32),
            target_surface=self.events["target_surface"][event_row].astype(np.int64),
            duration=float(self.events["duration"][event_row]),
            geometry=self.events["geometry"][event_row].astype(np.float32),
        )

    def full_body_segment(self, trajectory_id: int, event_index: int) -> FullBodyTeacherSegment:
        """Load private mechanical supervision; sparse-only teachers are rejected."""
        trajectory = self._load_trajectory(trajectory_id)
        missing = sorted({"joint_pos", "joint_names", "robot_asset_json"} - set(trajectory))
        if missing:
            path = Path(self._rows[trajectory_id]["motion_file"])
            raise ValueError(
                f"{path} is a sparse-only teacher and cannot supervise a mechanically "
                f"constrained trajectory; missing {missing}"
            )
        boundary = trajectory["boundary_frames"].astype(np.int64)
        first, last = int(boundary[event_index]), int(boundary[event_index + 1])
        full_body = FullBodyTrajectory(
            qpos=trajectory["joint_pos"][first : last + 1].astype(np.float32),
            fps=float(np.asarray(trajectory["fps"]).item()),
            robot_asset_json=str(np.asarray(trajectory["robot_asset_json"]).item()),
            joint_names=tuple(trajectory["joint_names"].astype(str)),
        )
        return FullBodyTeacherSegment(
            trajectory=full_body,
            contact=trajectory["contact_force_part_mask"][first : last + 1].astype(bool),
        )
