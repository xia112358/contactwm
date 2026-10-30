from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import ContactAnchorRecord, ContactPatchRecord


_BODY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "left_heel",
        "lhee",
        "left_ankle_roll_link",
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
    ),
    (
        "left_toe",
        "ltoe",
        "left_ankle_roll_link",
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "left_ankle_roll_sphere_5_link",
    ),
    (
        "right_heel",
        "rhee",
        "right_ankle_roll_link",
        "right_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_2_link",
    ),
    (
        "right_toe",
        "rtoe",
        "right_ankle_roll_link",
        "right_ankle_roll_sphere_1_link",
        "right_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_4_link",
        "right_ankle_roll_sphere_5_link",
    ),
    ("left_foot", "lf", "left_ankle", "left_ankle_roll_link", "left_ankle_pitch_link"),
    ("right_foot", "rf", "right_ankle", "right_ankle_roll_link", "right_ankle_pitch_link"),
    (
        "left_hand",
        "lh",
        "left_wrist",
        "left_wrist_yaw_link",
        "left_rubber_hand_link",
        "left_sphere_hand_link",
        "left_sphere_hand_tip_link",
    ),
    (
        "right_hand",
        "rh",
        "right_wrist",
        "right_wrist_yaw_link",
        "right_rubber_hand_link",
        "right_sphere_hand_link",
        "right_sphere_hand_tip_link",
    ),
    ("left_knee", "lk", "left_knee_link"),
    ("right_knee", "rk", "right_knee_link"),
)


def world_point_to_body_local(point_w: np.ndarray, body_pos_w: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    """Transform one world-space point into a Newton-recorded body frame."""

    rotation = _quat_wxyz_to_matrix(body_quat_wxyz)
    return rotation.T @ (np.asarray(point_w, dtype=np.float64) - np.asarray(body_pos_w, dtype=np.float64))


def body_local_point_to_world(point_b: np.ndarray, body_pos_w: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    """Transform one body-local point back to world space."""

    rotation = _quat_wxyz_to_matrix(body_quat_wxyz)
    return np.asarray(body_pos_w, dtype=np.float64) + rotation @ np.asarray(point_b, dtype=np.float64)


def world_vector_to_body_local(vector_w: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = _quat_wxyz_to_matrix(body_quat_wxyz)
    return rotation.T @ np.asarray(vector_w, dtype=np.float64)


def bind_newton_contact_patches(
    anchors: Sequence[ContactAnchorRecord],
    motion: Mapping[str, Any],
    *,
    source_recording_path: str | Path | None = None,
    min_force_norm: float = 0.0,
) -> tuple[list[ContactPatchRecord], dict[str, Any]]:
    """Build robot-local contact patches from Newton/MJWarp raw contacts.

    Runtime Newton body/shape integer IDs are used only to resolve stable labels.
    Persisted bindings use body/shape labels plus robot-local points so they remain
    valid when the scene is rebuilt with different runtime IDs.

    The persisted robot-local patch uses the point owned by the matched robot
    body. The paired counterpart point is retained only as a world-space surface
    target: ``body0 -> point0_w`` and ``body1 -> point1_w``.
    """

    arrays = _required_motion_arrays(motion)
    metadata = _load_newton_metadata(motion, source_recording_path=source_recording_path)
    body_labels = _string_list(motion.get("newton_body_labels")) or [str(item) for item in metadata.get("newton_body_labels", [])]
    shape_labels = _string_list(motion.get("newton_shape_labels")) or [str(item) for item in metadata.get("newton_shape_labels", [])]
    body_names = _string_list(motion.get("body_names"))
    selected_env_id = _scalar_int(
        motion.get("raw_contact_selected_env_id", motion.get("rollout_ref_source_env_id", metadata.get("selected_env_id", metadata.get("env_id", 0)))),
        default=0,
    )

    if not body_labels:
        raise ValueError(
            "Newton contact binding requires newton_body_labels in the motion or in the source recording metadata"
        )
    if not body_names:
        raise ValueError("Newton contact binding requires body_names in the motion")

    patches: list[ContactPatchRecord] = []
    warnings: list[str] = []
    bound_count = 0
    sample_count = 0

    for anchor in anchors:
        patch, patch_warnings, used_samples = _bind_anchor(
            anchor,
            arrays=arrays,
            body_labels=body_labels,
            shape_labels=shape_labels,
            body_names=body_names,
            selected_env_id=selected_env_id,
            min_force_norm=float(min_force_norm),
        )
        patches.append(patch)
        warnings.extend(patch_warnings)
        sample_count += used_samples
        if patch.robot_points_local:
            bound_count += 1

    summary = {
        "backend": "newton_mjwarp",
        "point_body_pairing": "body0->point0,body1->point1",
        "selected_env_id": int(selected_env_id),
        "anchor_count": len(anchors),
        "bound_patch_count": int(bound_count),
        "fallback_patch_count": int(len(anchors) - bound_count),
        "raw_sample_count": int(sample_count),
        "warnings": warnings,
    }
    return sorted(patches, key=lambda item: (item.start_frame, item.end_frame, item.body, item.patch_id)), summary


def _bind_anchor(
    anchor: ContactAnchorRecord,
    *,
    arrays: dict[str, np.ndarray],
    body_labels: list[str],
    shape_labels: list[str],
    body_names: list[str],
    selected_env_id: int,
    min_force_norm: float,
) -> tuple[ContactPatchRecord, list[str], int]:
    n_frames = arrays["body_pos_w"].shape[0]
    start = max(0, min(n_frames, int(anchor.start_frame)))
    end = max(start, min(n_frames, int(anchor.end_frame)))
    groups: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    for frame in range(start, end):
        count = max(0, min(int(arrays["raw_contact_count"][frame]), arrays["raw_contact_body0"].shape[1]))
        for contact_index in range(count):
            force = arrays["raw_contact_force_w"][frame, contact_index]
            if float(np.linalg.norm(force)) < min_force_norm:
                continue

            body0 = int(arrays["raw_contact_body0"][frame, contact_index])
            body1 = int(arrays["raw_contact_body1"][frame, contact_index])
            label0 = body_labels[body0] if 0 <= body0 < len(body_labels) else ""
            label1 = body_labels[body1] if 0 <= body1 < len(body_labels) else ""
            match0 = _label_matches_anchor(label0, anchor.body, selected_env_id)
            match1 = _label_matches_anchor(label1, anchor.body, selected_env_id)
            if match0 == match1:
                continue

            if match0:
                robot_body_id = body0
                robot_shape_id = int(arrays["raw_contact_shape0"][frame, contact_index])
                robot_point_w = arrays["raw_contact_point0_w"][frame, contact_index]
                counterpart_point_w = arrays["raw_contact_point1_w"][frame, contact_index]
                robot_to_counterpart_normal_w = arrays["raw_contact_normal_w"][frame, contact_index]
                robot_body_label = label0
            else:
                robot_body_id = body1
                robot_shape_id = int(arrays["raw_contact_shape1"][frame, contact_index])
                robot_point_w = arrays["raw_contact_point1_w"][frame, contact_index]
                counterpart_point_w = arrays["raw_contact_point0_w"][frame, contact_index]
                robot_to_counterpart_normal_w = -arrays["raw_contact_normal_w"][frame, contact_index]
                robot_body_label = label1

            motion_body_index = _resolve_motion_body_index(body_names, robot_body_label, anchor.body)
            if motion_body_index is None:
                continue

            body_pos_w = arrays["body_pos_w"][frame, motion_body_index]
            body_quat_w = arrays["body_quat_w"][frame, motion_body_index]
            point_local = world_point_to_body_local(robot_point_w, body_pos_w, body_quat_w)
            normal_local = world_vector_to_body_local(robot_to_counterpart_normal_w, body_quat_w)
            normal_norm = float(np.linalg.norm(normal_local))
            if normal_norm > 1.0e-12:
                normal_local = normal_local / normal_norm

            shape_label = shape_labels[robot_shape_id] if 0 <= robot_shape_id < len(shape_labels) else f"shape:{robot_shape_id}"
            groups.setdefault(shape_label, []).append(
                {
                    "frame": frame,
                    "body_index": motion_body_index,
                    "body_label": robot_body_label,
                    "body_id": robot_body_id,
                    "shape_id": robot_shape_id,
                    "point_local": point_local,
                    "normal_local": normal_local,
                    "point_world": np.asarray(robot_point_w, dtype=np.float64),
                    "target_point_world": np.asarray(
                        counterpart_point_w, dtype=np.float64
                    ),
                }
            )

    if not groups:
        warnings.append(f"{anchor.anchor_id}: no Newton raw-contact samples matched body {anchor.body!r}")
        return _fallback_patch(anchor), warnings, 0

    shape_names: list[str] = []
    local_points: list[list[float]] = []
    local_normals: list[list[float]] = []
    link_names: list[str] = []
    reconstruction_errors: list[float] = []
    runtime_body_ids: set[int] = set()
    runtime_shape_ids: set[int] = set()
    total_samples = 0

    for shape_label in sorted(groups):
        samples = groups[shape_label]
        total_samples += len(samples)
        point_local = np.median(np.stack([item["point_local"] for item in samples], axis=0), axis=0)
        normal_local = np.median(np.stack([item["normal_local"] for item in samples], axis=0), axis=0)
        normal_norm = float(np.linalg.norm(normal_local))
        if normal_norm > 1.0e-12:
            normal_local = normal_local / normal_norm

        for item in samples:
            frame = int(item["frame"])
            body_index = int(item["body_index"])
            reconstructed = body_local_point_to_world(
                point_local,
                arrays["body_pos_w"][frame, body_index],
                arrays["body_quat_w"][frame, body_index],
            )
            reconstruction_errors.append(float(np.linalg.norm(reconstructed - item["point_world"])))
            runtime_body_ids.add(int(item["body_id"]))
            runtime_shape_ids.add(int(item["shape_id"]))

        shape_names.append(shape_label)
        local_points.append(point_local.astype(float).tolist())
        local_normals.append(normal_local.astype(float).tolist())
        body_leaf = _label_leaf(str(samples[0]["body_label"]))
        if body_leaf and body_leaf not in link_names:
            link_names.append(body_leaf)

    target_frames = np.arange(start, end, dtype=np.int64)
    source_target_points = np.empty(
        (len(target_frames), len(shape_names), 3),
        dtype=np.float64,
    )
    for shape_index, shape_label in enumerate(sorted(groups)):
        samples_by_frame: dict[int, list[np.ndarray]] = {}
        for item in groups[shape_label]:
            samples_by_frame.setdefault(int(item["frame"]), []).append(
                np.asarray(item["target_point_world"], dtype=np.float64)
            )
        known_frames = np.asarray(sorted(samples_by_frame), dtype=np.int64)
        known_points = np.stack(
            [
                np.median(np.stack(samples_by_frame[int(frame)]), axis=0)
                for frame in known_frames
            ],
            axis=0,
        )
        for axis in range(3):
            source_target_points[:, shape_index, axis] = np.interp(
                target_frames,
                known_frames,
                known_points[:, axis],
            )

    center_world = anchor.world_position
    if center_world is None:
        all_world_points = np.concatenate(
            [
                np.stack(
                    [item["target_point_world"] for item in samples], axis=0
                )
                for samples in groups.values()
            ],
            axis=0,
        )
        center_world = np.median(all_world_points, axis=0).astype(float).tolist()

    metadata = dict(anchor.metadata)
    metadata["newton_robot_contact_binding"] = {
        "backend": "newton_mjwarp",
        "point_body_pairing": "body0->point0,body1->point1",
        "normal_convention": "robot_to_counterpart_from_newton_raw_normal",
        "sample_count": int(total_samples),
        "runtime_body_ids_source": sorted(runtime_body_ids),
        "runtime_shape_ids_source": sorted(runtime_shape_ids),
        "reconstruction_error_mean_m": float(np.mean(reconstruction_errors)) if reconstruction_errors else 0.0,
        "reconstruction_error_max_m": float(np.max(reconstruction_errors)) if reconstruction_errors else 0.0,
    }

    patch = ContactPatchRecord(
        motion_id=anchor.motion_id,
        patch_id=anchor.patch_id or f"{anchor.anchor_id}_patch",
        body=anchor.body,
        start_frame=anchor.start_frame,
        end_frame=anchor.end_frame,
        patch_type=_patch_type(anchor.body),
        patch_center_world=center_world,
        link_names=link_names or [anchor.body],
        sphere_ids=shape_names,
        anchor_id=anchor.anchor_id,
        slip_score=anchor.metadata.get("mean_drift_xy"),
        newton_body_label=str(next(iter(groups.values()))[0]["body_label"]),
        newton_shape_labels=shape_names,
        robot_points_local=local_points,
        robot_normals_local=local_normals,
        source_target_frames=target_frames.astype(int).tolist(),
        source_target_points_w=source_target_points.astype(float).tolist(),
        robot_binding_backend="newton_mjwarp",
        robot_binding_source="newton_raw_contact",
        metadata=metadata,
    )
    patch.validate()
    return patch, warnings, total_samples


def _fallback_patch(anchor: ContactAnchorRecord) -> ContactPatchRecord:
    patch = ContactPatchRecord(
        motion_id=anchor.motion_id,
        patch_id=anchor.patch_id or f"{anchor.anchor_id}_patch",
        body=anchor.body,
        start_frame=anchor.start_frame,
        end_frame=anchor.end_frame,
        patch_type=_patch_type(anchor.body),
        patch_center_world=anchor.world_position,
        link_names=[anchor.body],
        anchor_id=anchor.anchor_id,
        slip_score=anchor.metadata.get("mean_drift_xy"),
        robot_binding_backend="newton_mjwarp",
        robot_binding_source="legacy_center",
        metadata={**anchor.metadata, "role": anchor.role, "source_anchor_id": anchor.anchor_id},
    )
    patch.validate()
    return patch


def _required_motion_arrays(motion: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = (
        "body_pos_w",
        "body_quat_w",
        "raw_contact_count",
        "raw_contact_shape0",
        "raw_contact_shape1",
        "raw_contact_body0",
        "raw_contact_body1",
        "raw_contact_point0_w",
        "raw_contact_point1_w",
        "raw_contact_normal_w",
        "raw_contact_force_w",
    )
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"Newton contact binding is missing required motion arrays: {missing}")
    arrays = {name: np.asarray(motion[name]) for name in required}
    if arrays["body_pos_w"].ndim != 3 or arrays["body_pos_w"].shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T,B,3], got {arrays['body_pos_w'].shape}")
    if arrays["body_quat_w"].shape[:2] != arrays["body_pos_w"].shape[:2] or arrays["body_quat_w"].shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape [T,B,4], got {arrays['body_quat_w'].shape}")
    return arrays


def _load_newton_metadata(motion: Mapping[str, Any], *, source_recording_path: str | Path | None) -> dict[str, Any]:
    for key in ("newton_contact_metadata_json", "_metadata_json"):
        if key in motion:
            try:
                return json.loads(_scalar_string(motion[key]))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

    recording = Path(source_recording_path).expanduser() if source_recording_path is not None else None
    if recording is None and "rollout_ref_source_recording" in motion:
        candidate = Path(_scalar_string(motion["rollout_ref_source_recording"])).expanduser()
        if candidate.is_file():
            recording = candidate
    if recording is None or not recording.is_file():
        return {}
    with np.load(recording, allow_pickle=True) as data:
        if "_metadata_json" not in data.files:
            return {}
        return json.loads(str(np.asarray(data["_metadata_json"]).item()))


def _resolve_motion_body_index(body_names: Sequence[str], newton_body_label: str, anchor_body: str) -> int | None:
    leaf = _label_leaf(newton_body_label).lower()
    lowered = [str(name).lower() for name in body_names]
    if leaf in lowered:
        return lowered.index(leaf)
    aliases = _body_aliases(anchor_body)
    for index, name in enumerate(lowered):
        if name in aliases or any(alias in name or name in alias for alias in aliases):
            return index
    return None


def _label_matches_anchor(label: str, body: str, selected_env_id: int) -> bool:
    if not label:
        return False
    lower = label.lower()
    if "/envs/env_" in lower and f"/envs/env_{selected_env_id}/" not in lower:
        return False
    leaf = _label_leaf(label).lower()
    aliases = _body_aliases(body)
    return leaf in aliases or any(alias in leaf or leaf in alias for alias in aliases)


def _body_aliases(body: str) -> set[str]:
    lower = str(body).lower()
    aliases = {lower}
    for group in _BODY_ALIAS_GROUPS:
        if lower in group or any(token in lower or lower in token for token in group):
            aliases.update(group)
    return aliases


def _patch_type(body: str) -> str:
    lower = str(body).lower()
    if "foot" in lower or "ankle" in lower or lower in {"lf", "rf"}:
        return "foot"
    if "hand" in lower or "wrist" in lower or lower in {"lh", "rh"}:
        return "hand"
    if "knee" in lower or lower in {"lk", "rk"}:
        return "knee"
    return "unknown"


def _label_leaf(label: str) -> str:
    return str(label).rstrip("/").split("/")[-1]


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        raw = arr.item()
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [raw]
        if isinstance(raw, (list, tuple)):
            return [str(item) for item in raw]
        return [str(raw)]
    return [str(item) for item in arr.reshape(-1).tolist()]


def _scalar_string(value: Any) -> str:
    arr = np.asarray(value, dtype=object)
    raw = arr.item() if arr.ndim == 0 else arr.reshape(-1)[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _scalar_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return int(default)
