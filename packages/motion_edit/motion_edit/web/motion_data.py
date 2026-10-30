from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from somaforge_core.contact_schema import CONTACT_FORCE_PART_NAMES


def load_motion_sequence(path: str | Path) -> tuple[np.ndarray, float, tuple[str, ...]]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path} requires finite positive fps")
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
        elif "joint_pos" in data:
            qpos = np.asarray(data["joint_pos"], dtype=np.float32)
        else:
            raise KeyError(f"{path} has neither qpos nor joint_pos")
        if qpos.ndim != 2:
            raise ValueError(f"{path} motion array must be [T,D], got {qpos.shape}")
        if "joint_names" not in data:
            raise ValueError(f"{path} has no joint_names; canonical playback requires named joints")
        joint_names = tuple(str(item) for item in np.asarray(data["joint_names"]).reshape(-1).tolist())
    if qpos.shape[1] != 7 + len(joint_names):
        raise ValueError(f"{path} motion width {qpos.shape[1]} does not match 7 + {len(joint_names)} joint_names")
    return qpos, fps, joint_names


def load_contact_force_payload(path: str | Path, *, frame_count: int, motion_path: str | Path | None = None) -> dict:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        names = (
            [str(item) for item in np.asarray(data["contact_force_part_order"]).reshape(-1).tolist()]
            if "contact_force_part_order" in data
            else list(CONTACT_FORCE_PART_NAMES)
        )
        forces = (
            np.asarray(data["contact_force_part_w"], dtype=np.float32)
            if "contact_force_part_w" in data
            else np.zeros((frame_count, len(names), 3), dtype=np.float32)
        )
        masks_known = "contact_force_part_mask" in data
        masks = (
            np.asarray(data["contact_force_part_mask"], dtype=bool)
            if "contact_force_part_mask" in data
            else np.zeros(forces.shape[:2], dtype=bool)
        )
        positions = (
            np.asarray(data["contact_force_part_position_w"], dtype=np.float32)
            if "contact_force_part_position_w" in data
            else np.zeros_like(forces)
        )
        position_valid = (
            np.asarray(data["contact_force_part_position_valid"], dtype=bool)
            if "contact_force_part_position_valid" in data
            else masks.copy()
        )
        raw_counts = np.asarray(data["raw_contact_count"], dtype=np.int64) if "raw_contact_count" in data else None
        raw_points = np.asarray(data["raw_contact_point0_w"], dtype=np.float32) if "raw_contact_point0_w" in data else None
        raw_forces = np.asarray(data["raw_contact_force_w"], dtype=np.float32) if "raw_contact_force_w" in data else None
        has_solver_evidence = "contact_semantics_json" in data and "contact_pairs_json" in data
        semantics = json.loads(data["contact_semantics_json"].item()) if has_solver_evidence else None
        pairs = json.loads(data["contact_pairs_json"].item()) if has_solver_evidence else None
    count = min(frame_count, forces.shape[0], masks.shape[0], positions.shape[0], position_valid.shape[0])
    sample_points: list[list[list[float]]] = []
    sample_forces: list[list[list[float]]] = []
    if raw_counts is not None and raw_points is not None and raw_forces is not None:
        raw_frame_count = min(count, raw_counts.shape[0], raw_points.shape[0], raw_forces.shape[0])
        for frame in range(raw_frame_count):
            sample_count = max(0, min(int(raw_counts[frame]), raw_points.shape[1], raw_forces.shape[1]))
            sample_points.append(raw_points[frame, :sample_count].tolist())
            sample_forces.append(raw_forces[frame, :sample_count].tolist())
        for _ in range(raw_frame_count, count):
            sample_points.append([])
            sample_forces.append([])
    else:
        for frame in range(count):
            active = np.asarray(masks[frame], dtype=bool)
            sample_points.append(np.asarray(positions[frame])[active].tolist())
            sample_forces.append(np.asarray(forces[frame])[active].tolist())
    payload = {
        "mask_status": "recorded_force_mask_unverified" if masks_known else "unknown_missing_contact_mask",
        "part_order": names,
        "forces": forces[:count].tolist(),
        "masks": [],
        "recorded_force_masks": masks[:count].tolist() if masks_known else [],
        "positions": positions[:count].tolist(),
        "position_valid": position_valid[:count].tolist(),
        "sample_points": sample_points,
        "sample_forces": sample_forces,
    }
    if has_solver_evidence:
        from somaforge_core.newton_contact_data import load_raw_contact_labels
        from somaforge_core.contact_face_selection import FACE_POLICY, select_contact_pairs
        from somaforge_core.newton_contacts import PARTS
        source = Path(motion_path or path)
        separate = Path(path) if source.resolve() != Path(path).resolve() else None
        load_raw_contact_labels(source, separate, frame_count=frame_count)
        if len(pairs) != frame_count:
            raise ValueError('Newton pair timeline mismatch')
        selected = select_contact_pairs(pairs, semantics['scene']['surface_catalog'])
        payload.update(part_order=list(PARTS), masks=selected['contact_part_mask'].tolist(),
                       positions=selected['contact_position_w'].tolist(),
                       position_valid=selected['contact_part_mask'].tolist(),
                       forces=np.zeros((frame_count, len(PARTS), 3)).tolist(),
                       mask_status='verified_newton_primary_faces', force_status='unknown_for_selected_pairs',
                       contact_semantics=semantics, face_selection=FACE_POLICY,
                       raw_contact_pairs=pairs, abnormal_contact_pairs=selected['abnormal_contact_pairs'])
    return payload
