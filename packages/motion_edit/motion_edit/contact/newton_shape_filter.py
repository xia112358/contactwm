from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from motion_edit.contact.surface_geometry import point_in_polygon_uv
from . import newton_bindings as _legacy
from .schema import ContactAnchorRecord, ContactPatchRecord


_INSTALLED = False

# Keep physical foot subcontacts disjoint.  The ankle-roll body can own all sole
# collision shapes, so raw shape identity is the authoritative heel/toe split.
_STRICT_BODY_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
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


def _subcontact_kind(
    body: str,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    role = (metadata or {}).get("patch_role")
    if str(role).lower() in {"heel", "toe", "sole"}:
        return str(role).lower()
    value = str(body).lower()
    if "heel" in value or value in {"lhee", "rhee"}:
        return "heel"
    if "toe" in value or value in {"ltoe", "rtoe"}:
        return "toe"
    return None


def _shape_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    resolved: set[int] = set()
    for item in np.asarray(value, dtype=object).reshape(-1).tolist():
        try:
            resolved.add(int(item))
        except (TypeError, ValueError):
            continue
    return resolved


def _anchor_raw_shape_ids(anchor: ContactAnchorRecord) -> tuple[int, ...]:
    """Read direct and nested foot-subcontact shape identities.

    Current split-foot metadata stores IDs as
    ``foot_subcontact.heel.raw_shape_ids`` or
    ``foot_subcontact.toe.raw_shape_ids``. Direct
    ``foot_subcontact.raw_shape_ids`` remains supported for old layers.
    """

    metadata = anchor.metadata if isinstance(anchor.metadata, Mapping) else {}
    subcontact = metadata.get("foot_subcontact")
    if not isinstance(subcontact, Mapping):
        return ()

    resolved = _shape_ids(subcontact.get("raw_shape_ids"))
    kind = _subcontact_kind(anchor.body, metadata)
    if kind in {"heel", "toe"}:
        nested = subcontact.get(kind)
        if isinstance(nested, Mapping):
            resolved.update(_shape_ids(nested.get("raw_shape_ids")))
    elif kind == "sole" or not resolved:
        # Unsplit/legacy foot anchors may intentionally represent the whole sole.
        for name in ("heel", "toe"):
            nested = subcontact.get(name)
            if isinstance(nested, Mapping):
                resolved.update(_shape_ids(nested.get("raw_shape_ids")))

    return tuple(sorted(resolved))


def _foot_shape_role(label: str) -> str | None:
    value = str(label).lower()
    if "heel" in value or any(
        token in value for token in ("sphere_1", "sphere_2")
    ):
        return "heel"
    if "toe" in value or any(
        token in value for token in ("sphere_3", "sphere_4", "sphere_5")
    ):
        return "toe"
    return None


def _physical_foot_shape_ids(
    anchor: ContactAnchorRecord,
    shape_ids: Sequence[int],
    shape_labels: Sequence[str],
) -> tuple[int, ...]:
    """Drop generic ankle collision shapes from heel/toe patch identity."""

    metadata = anchor.metadata if isinstance(anchor.metadata, Mapping) else {}
    role = _subcontact_kind(anchor.body, metadata)
    if role not in {"heel", "toe", "sole"}:
        return tuple(sorted(int(value) for value in shape_ids))
    accepted_roles = {"heel", "toe"} if role == "sole" else {role}
    physical = tuple(
        sorted(
            int(shape_id)
            for shape_id in shape_ids
            if 0 <= int(shape_id) < len(shape_labels)
            and _foot_shape_role(shape_labels[int(shape_id)]) in accepted_roles
        )
    )
    if physical:
        return physical
    any_foot_sphere = tuple(
        sorted(
            int(shape_id)
            for shape_id in shape_ids
            if 0 <= int(shape_id) < len(shape_labels)
            and _foot_shape_role(shape_labels[int(shape_id)]) is not None
        )
    )
    if any_foot_sphere:
        return any_foot_sphere
    # Old recordings can use semantic labels without sphere/heel/toe identity.
    # Preserve their explicit IDs only when no physical role can be resolved.
    return tuple(sorted(int(value) for value in shape_ids))


def _frame_candidates(body: str) -> tuple[str, ...]:
    value = str(body).lower()
    side = "left" if value.startswith(("left", "l")) else "right" if value.startswith(("right", "r")) else ""

    if side and any(token in value for token in ("heel", "toe", "foot", "ankle")):
        # Express every sole patch in one actual rigid foot frame. The fixed
        # sphere links are geometry markers, not independent articulated bodies.
        return (
            f"{side}_ankle_roll_link",
            f"{side}_ankle_pitch_link",
            f"{side}_ankle_intermediate_1_link",
            f"{side}_ankle_roll_sphere_5_link",
            f"{side}_ankle_roll_sphere_1_link",
        )
    if side and any(token in value for token in ("hand", "wrist")):
        return (
            f"{side}_sphere_hand_link",
            f"{side}_wrist_yaw_link",
            f"{side}_rubber_hand_link",
            f"{side}_sphere_hand_tip_link",
        )
    if side and "knee" in value:
        return (f"{side}_knee_link",)
    return (str(body),)


def _resolve_canonical_frame(body_names: Sequence[str], anchor_body: str) -> tuple[int, str] | None:
    names = [str(item) for item in body_names]
    lowered = [name.lower() for name in names]
    for candidate in _frame_candidates(anchor_body):
        key = candidate.lower()
        if key in lowered:
            index = lowered.index(key)
            return index, names[index]
    for candidate in _frame_candidates(anchor_body):
        key = candidate.lower()
        for index, name in enumerate(lowered):
            if key and (key in name or name in key):
                return index, names[index]
    return None


def _uses_rolling_local_contact_point(anchor_body: str) -> bool:
    """Allow contact material points to migrate only on round hand/knee geometry."""

    value = str(anchor_body).lower()
    return any(token in value for token in ("hand", "wrist", "knee"))


def _shape_label(shape_labels: Sequence[str], shape_id: int) -> str:
    if 0 <= int(shape_id) < len(shape_labels):
        return str(shape_labels[int(shape_id)])
    return f"shape:{int(shape_id)}"


def _anchor_surface_polygon(anchor: ContactAnchorRecord) -> list[tuple[float, float]]:
    bindings = anchor.metadata.get("surface_bindings")
    if not isinstance(bindings, list) or not bindings:
        return []
    latest = bindings[-1]
    raw = (
        latest.get("polygon_surface_coordinates")
        if isinstance(latest, dict)
        else None
    )
    if not isinstance(raw, list) or len(raw) < 3:
        return []
    try:
        return [(float(item["u"]), float(item["v"])) for item in raw]
    except (KeyError, TypeError, ValueError):
        return []


def _counterpart_matches_anchor_surface(
    anchor: ContactAnchorRecord,
    point_w: np.ndarray,
    *,
    plane_tolerance_m: float = 2.0e-3,
) -> bool:
    if (
        anchor.surface_origin is None
        or anchor.surface_normal is None
        or anchor.surface_tangent_u is None
        or anchor.surface_tangent_v is None
    ):
        return True
    point = np.asarray(point_w, dtype=np.float64)
    origin = np.asarray(anchor.surface_origin, dtype=np.float64)
    normal = np.asarray(anchor.surface_normal, dtype=np.float64)
    tangent_u = np.asarray(anchor.surface_tangent_u, dtype=np.float64)
    tangent_v = np.asarray(anchor.surface_tangent_v, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    tangent_u /= max(float(np.linalg.norm(tangent_u)), 1.0e-12)
    tangent_v /= max(float(np.linalg.norm(tangent_v)), 1.0e-12)
    relative = point - origin
    if abs(float(np.dot(relative, normal))) > float(plane_tolerance_m):
        return False
    uv = (float(np.dot(relative, tangent_u)), float(np.dot(relative, tangent_v)))
    polygon = _anchor_surface_polygon(anchor)
    if polygon:
        return bool(point_in_polygon_uv(uv, polygon))
    bounds = anchor.surface_bounds
    if not isinstance(bounds, dict):
        return True
    for axis, value in (("u", uv[0]), ("v", uv[1])):
        interval = bounds.get(axis)
        if (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and not float(interval[0]) <= value <= float(interval[1])
        ):
            return False
    return True


def _bind_anchor_canonical(
    anchor: ContactAnchorRecord,
    *,
    arrays: dict[str, np.ndarray],
    body_labels: list[str],
    shape_labels: list[str],
    body_names: list[str],
    selected_env_id: int,
    min_force_norm: float,
) -> tuple[
    ContactPatchRecord,
    list[str],
    int,
    int,
    int,
    tuple[int, ...],
]:
    warnings: list[str] = []
    allowed_shape_ids = _physical_foot_shape_ids(
        anchor,
        _anchor_raw_shape_ids(anchor),
        shape_labels,
    )
    allowed = set(allowed_shape_ids)
    anchor_metadata = (
        anchor.metadata if isinstance(anchor.metadata, Mapping) else {}
    )
    fallback_foot_role = _subcontact_kind(anchor.body, anchor_metadata)

    frame_resolution = _resolve_canonical_frame(body_names, anchor.body)
    if frame_resolution is None:
        warnings.append(
            f"{anchor.anchor_id}: no canonical motion body frame resolves for {anchor.body!r}"
        )
        return (
            _legacy._fallback_patch(anchor),
            warnings,
            0,
            0,
            0,
            allowed_shape_ids,
        )
    canonical_body_index, canonical_frame_label = frame_resolution

    n_frames = arrays["body_pos_w"].shape[0]
    start = max(0, min(n_frames, int(anchor.start_frame)))
    end = max(start, min(n_frames, int(anchor.end_frame)))
    groups: dict[str, list[dict[str, Any]]] = {}
    filtered_sample_count = 0
    surface_filtered_sample_count = 0
    matched_unfiltered_count = 0

    for frame in range(start, end):
        count = max(
            0,
            min(
                int(arrays["raw_contact_count"][frame]),
                arrays["raw_contact_body0"].shape[1],
            ),
        )
        for contact_index in range(count):
            force = arrays["raw_contact_force_w"][frame, contact_index]
            if float(np.linalg.norm(force)) < float(min_force_norm):
                continue

            body0 = int(arrays["raw_contact_body0"][frame, contact_index])
            body1 = int(arrays["raw_contact_body1"][frame, contact_index])
            label0 = body_labels[body0] if 0 <= body0 < len(body_labels) else ""
            label1 = body_labels[body1] if 0 <= body1 < len(body_labels) else ""
            match0 = _legacy._label_matches_anchor(label0, anchor.body, selected_env_id)
            match1 = _legacy._label_matches_anchor(label1, anchor.body, selected_env_id)
            if match0 == match1:
                continue

            if match0:
                runtime_body_id = body0
                runtime_shape_id = int(arrays["raw_contact_shape0"][frame, contact_index])
                runtime_body_label = label0
                robot_point_w = arrays["raw_contact_point0_w"][frame, contact_index]
                counterpart_point_w = arrays["raw_contact_point1_w"][frame, contact_index]
                robot_normal_w = arrays["raw_contact_normal_w"][frame, contact_index]
            else:
                runtime_body_id = body1
                runtime_shape_id = int(arrays["raw_contact_shape1"][frame, contact_index])
                runtime_body_label = label1
                robot_point_w = arrays["raw_contact_point1_w"][frame, contact_index]
                counterpart_point_w = arrays["raw_contact_point0_w"][frame, contact_index]
                robot_normal_w = -arrays["raw_contact_normal_w"][frame, contact_index]

            matched_unfiltered_count += 1
            if not _counterpart_matches_anchor_surface(
                anchor,
                counterpart_point_w,
            ):
                surface_filtered_sample_count += 1
                continue
            if allowed and runtime_shape_id not in allowed:
                filtered_sample_count += 1
                continue

            label = _shape_label(shape_labels, runtime_shape_id)
            if not allowed and fallback_foot_role in {"heel", "toe"}:
                if _foot_shape_role(label) != fallback_foot_role:
                    filtered_sample_count += 1
                    continue

            body_pos_w = arrays["body_pos_w"][frame, canonical_body_index]
            body_quat_w = arrays["body_quat_w"][frame, canonical_body_index]
            point_local = _legacy.world_point_to_body_local(
                robot_point_w,
                body_pos_w,
                body_quat_w,
            )
            normal_local = _legacy.world_vector_to_body_local(
                robot_normal_w,
                body_quat_w,
            )
            normal_norm = float(np.linalg.norm(normal_local))
            if normal_norm > 1.0e-12:
                normal_local = normal_local / normal_norm

            groups.setdefault(label, []).append(
                {
                    "frame": int(frame),
                    "runtime_body_id": int(runtime_body_id),
                    "runtime_shape_id": int(runtime_shape_id),
                    "runtime_body_label": str(runtime_body_label),
                    "point_local": np.asarray(point_local, dtype=np.float64),
                    "normal_local": np.asarray(normal_local, dtype=np.float64),
                    "robot_point_world": np.asarray(
                        robot_point_w, dtype=np.float64
                    ),
                    "target_point_world": np.asarray(
                        counterpart_point_w, dtype=np.float64
                    ),
                }
            )

    if not groups:
        suffix = (
            f" after strict raw_shape_ids={list(allowed_shape_ids)}"
            if allowed_shape_ids
            else ""
        )
        warnings.append(
            f"{anchor.anchor_id}: no Newton raw-contact samples matched {anchor.body!r}{suffix}"
        )
        return (
            _legacy._fallback_patch(anchor),
            warnings,
            0,
            filtered_sample_count,
            surface_filtered_sample_count,
            allowed_shape_ids,
        )

    shape_names: list[str] = []
    local_points: list[list[float]] = []
    local_normals: list[list[float]] = []
    reconstruction_errors: list[float] = []
    runtime_body_ids: set[int] = set()
    runtime_shape_ids: set[int] = set()
    runtime_body_labels: set[str] = set()
    source_world_points: list[np.ndarray] = []
    total_samples = 0

    for label in sorted(groups):
        samples = groups[label]
        total_samples += len(samples)
        point_local = np.median(
            np.stack([item["point_local"] for item in samples], axis=0),
            axis=0,
        )
        normal_local = np.median(
            np.stack([item["normal_local"] for item in samples], axis=0),
            axis=0,
        )
        normal_norm = float(np.linalg.norm(normal_local))
        if normal_norm > 1.0e-12:
            normal_local = normal_local / normal_norm

        for item in samples:
            frame = int(item["frame"])
            reconstructed = _legacy.body_local_point_to_world(
                point_local,
                arrays["body_pos_w"][frame, canonical_body_index],
                arrays["body_quat_w"][frame, canonical_body_index],
            )
            reconstruction_errors.append(
                float(
                    np.linalg.norm(
                        reconstructed - item["robot_point_world"]
                    )
                )
            )
            runtime_body_ids.add(int(item["runtime_body_id"]))
            runtime_shape_ids.add(int(item["runtime_shape_id"]))
            runtime_body_labels.add(str(item["runtime_body_label"]))
            source_world_points.append(
                np.asarray(item["target_point_world"], dtype=np.float64)
            )

        shape_names.append(label)
        local_points.append(np.asarray(point_local, dtype=float).tolist())
        local_normals.append(np.asarray(normal_local, dtype=float).tolist())

    target_frames = np.arange(start, end, dtype=np.int64)
    source_target_points = np.empty(
        (len(target_frames), len(shape_names), 3),
        dtype=np.float64,
    )
    source_local_points = np.empty_like(source_target_points)
    for shape_index, label in enumerate(sorted(groups)):
        samples = groups[label]
        samples_by_frame: dict[int, list[np.ndarray]] = {}
        for item in samples:
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
        known_local_points = np.stack(
            [
                np.median(
                    np.stack(
                        [
                            np.asarray(item["point_local"], dtype=np.float64)
                            for item in samples
                            if int(item["frame"]) == int(frame)
                        ]
                    ),
                    axis=0,
                )
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
            source_local_points[:, shape_index, axis] = np.interp(
                target_frames,
                known_frames,
                known_local_points[:, axis],
            )

    center_world = anchor.world_position
    if center_world is None:
        center_world = np.median(
            np.stack(source_world_points, axis=0),
            axis=0,
        ).astype(float).tolist()

    metadata = dict(anchor.metadata)
    metadata["newton_robot_contact_binding"] = {
        "backend": "newton_mjwarp",
        "point_body_pairing": "body0->point0,body1->point1",
        "normal_convention": "robot_to_counterpart_from_newton_raw_normal",
        "sample_count": int(total_samples),
        "matched_sample_count_before_shape_filter": int(matched_unfiltered_count),
        "filtered_raw_contact_count": int(filtered_sample_count),
        "surface_filtered_raw_contact_count": int(
            surface_filtered_sample_count
        ),
        "raw_shape_ids_filter": list(allowed_shape_ids),
        "runtime_body_ids_source": sorted(runtime_body_ids),
        "runtime_shape_ids_source": sorted(runtime_shape_ids),
        "runtime_body_labels_source": sorted(runtime_body_labels),
        "local_point_frame_label": canonical_frame_label,
        "local_point_frame_body_index": int(canonical_body_index),
        "frame_contract": "newton_body_label_equals_robot_points_local_frame",
        "local_point_time_contract": (
            "rolling_per_frame"
            if _uses_rolling_local_contact_point(anchor.body)
            else "fixed_rigid_patch"
        ),
        "reconstruction_error_mean_m": (
            float(np.mean(reconstruction_errors)) if reconstruction_errors else 0.0
        ),
        "reconstruction_error_max_m": (
            float(np.max(reconstruction_errors)) if reconstruction_errors else 0.0
        ),
    }

    patch = ContactPatchRecord(
        motion_id=anchor.motion_id,
        patch_id=anchor.patch_id or f"{anchor.anchor_id}_patch",
        body=anchor.body,
        start_frame=anchor.start_frame,
        end_frame=anchor.end_frame,
        patch_type=_legacy._patch_type(anchor.body),
        patch_center_world=center_world,
        link_names=[canonical_frame_label],
        sphere_ids=shape_names,
        anchor_id=anchor.anchor_id,
        slip_score=anchor.metadata.get("mean_drift_xy"),
        # This label is the frame in which robot_points_local were computed.
        # Runtime contact body labels remain diagnostics in metadata.
        newton_body_label=canonical_frame_label,
        newton_shape_labels=shape_names,
        robot_points_local=local_points,
        robot_normals_local=local_normals,
        source_target_frames=target_frames.astype(int).tolist(),
        source_target_points_w=source_target_points.astype(float).tolist(),
        source_points_local_by_frame=(
            source_local_points.astype(float).tolist()
            if _uses_rolling_local_contact_point(anchor.body)
            else None
        ),
        robot_binding_backend="newton_mjwarp",
        robot_binding_source="newton_raw_contact",
        metadata=metadata,
    )
    patch.validate()
    return (
        patch,
        warnings,
        total_samples,
        filtered_sample_count,
        surface_filtered_sample_count,
        allowed_shape_ids,
    )


def bind_newton_contact_patches(
    anchors: Sequence[ContactAnchorRecord],
    motion: Mapping[str, Any],
    *,
    source_recording_path: str | Path | None = None,
    min_force_norm: float = 0.0,
) -> tuple[list[ContactPatchRecord], dict[str, Any]]:
    """Bind stable contact patches in one canonical robot-link frame.

    The persisted ``newton_body_label`` is exactly the body frame used to compute
    every ``robot_points_local`` value. Runtime Newton body/shape labels are kept
    only as diagnostics. Split foot anchors use their direct or nested
    ``raw_shape_ids`` as a strict robot-side shape filter.
    """

    arrays = _legacy._required_motion_arrays(motion)
    metadata = _legacy._load_newton_metadata(
        motion,
        source_recording_path=source_recording_path,
    )
    body_labels = _legacy._string_list(motion.get("newton_body_labels")) or [
        str(item) for item in metadata.get("newton_body_labels", [])
    ]
    shape_labels = _legacy._string_list(motion.get("newton_shape_labels")) or [
        str(item) for item in metadata.get("newton_shape_labels", [])
    ]
    body_names = _legacy._string_list(motion.get("body_names"))
    selected_env_id = _legacy._scalar_int(
        motion.get(
            "raw_contact_selected_env_id",
            motion.get(
                "rollout_ref_source_env_id",
                metadata.get("selected_env_id", metadata.get("env_id", 0)),
            ),
        ),
        default=0,
    )

    if not body_labels:
        raise ValueError(
            "Newton contact binding requires newton_body_labels in the motion "
            "or source recording metadata"
        )
    if not body_names:
        raise ValueError("Newton contact binding requires body_names in the motion")

    patches: list[ContactPatchRecord] = []
    warnings: list[str] = []
    bound_count = 0
    raw_sample_count = 0
    filtered_sample_count = 0
    surface_filtered_sample_count = 0
    strict_anchor_count = 0
    shape_filters: dict[str, list[int]] = {}
    frame_labels: dict[str, str] = {}

    for anchor in anchors:
        (
            patch,
            patch_warnings,
            used,
            filtered,
            surface_filtered,
            allowed,
        ) = _bind_anchor_canonical(
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
        raw_sample_count += int(used)
        filtered_sample_count += int(filtered)
        surface_filtered_sample_count += int(surface_filtered)
        if patch.robot_points_local:
            bound_count += 1
        if allowed:
            strict_anchor_count += 1
            shape_filters[anchor.anchor_id] = list(allowed)
        if patch.newton_body_label:
            frame_labels[anchor.anchor_id] = patch.newton_body_label

    summary = {
        "backend": "newton_mjwarp",
        "point_body_pairing": "body0->point0,body1->point1",
        "frame_contract": "newton_body_label_equals_robot_points_local_frame",
        "selected_env_id": int(selected_env_id),
        "anchor_count": len(anchors),
        "bound_patch_count": int(bound_count),
        "fallback_patch_count": int(len(anchors) - bound_count),
        "raw_sample_count": int(raw_sample_count),
        "strict_shape_filter_anchor_count": int(strict_anchor_count),
        "filtered_raw_contact_count": int(filtered_sample_count),
        "surface_filtered_raw_contact_count": int(
            surface_filtered_sample_count
        ),
        "raw_shape_ids_by_anchor": shape_filters,
        "local_point_frame_by_anchor": frame_labels,
        "warnings": warnings,
    }
    return (
        sorted(
            patches,
            key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.body,
                item.patch_id,
            ),
        ),
        summary,
    )


def install_strict_newton_shape_filter() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _legacy._BODY_ALIAS_GROUPS = _STRICT_BODY_ALIAS_GROUPS
    _legacy.bind_newton_contact_patches = bind_newton_contact_patches
    contact_package = sys.modules.get("motion_edit.contact")
    if contact_package is not None:
        setattr(
            contact_package,
            "bind_newton_contact_patches",
            bind_newton_contact_patches,
        )
    _INSTALLED = True
