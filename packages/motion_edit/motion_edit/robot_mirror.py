"""Canonical G1 robot-only reflection for Motion Edit augmentation."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from somaforge_core.robot_assets import canonical_g1_urdf_path, decode_robot_asset_json

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.patches import patches_from_anchors
from motion_edit.contact.schema import ContactSurfaceRecord
from motion_edit.contact.surface_geometry import point_in_polygon_uv, surface_polygon_uv


ROBOT_MIRROR_SCHEMA = "motion_edit_robot_only_mirror_v2"
_ROBOT_LOCAL_REFLECTION = np.diag([1.0, -1.0, 1.0])
_RAW_CONTACT_PREFIX = "raw_contact_"


def _root_local_reflection_matrices(root_qpos: np.ndarray) -> np.ndarray:
    root = np.asarray(root_qpos, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 7:
        raise ValueError(f"root qpos must have shape [T,7], got {root.shape}")
    rotation = Rotation.from_quat(root[:, [4, 5, 6, 3]]).as_matrix()
    return np.einsum("tij,jk,tlk->til", rotation, _ROBOT_LOCAL_REFLECTION, rotation)


def _reflect_root_local_points(points: np.ndarray, root_qpos: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    root = np.asarray(root_qpos, dtype=np.float64)
    if value.shape[0] != root.shape[0] or value.shape[-1] != 3:
        raise ValueError("root-local point reflection requires matching frame counts and xyz values")
    center = root[:, :3].reshape((root.shape[0],) + (1,) * (value.ndim - 2) + (3,))
    return np.einsum("tij,t...j->t...i", _root_local_reflection_matrices(root), value - center) + center


def _reflect_root_local_vectors(vectors: np.ndarray, root_qpos: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64)
    root = np.asarray(root_qpos, dtype=np.float64)
    if value.shape[0] != root.shape[0] or value.shape[-1] != 3:
        raise ValueError("root-local vector reflection requires matching frame counts and xyz values")
    return np.einsum("tij,t...j->t...i", _root_local_reflection_matrices(root), value)


def swap_left_right_name(name: str) -> str:
    if name.startswith("left_"):
        return f"right_{name[5:]}"
    if name.startswith("right_"):
        return f"left_{name[6:]}"
    upper = name.upper()
    part_pairs = {
        "LHEE": "RHEE",
        "LTOE": "RTOE",
        "LH": "RH",
        "LK": "RK",
        "RHEE": "LHEE",
        "RTOE": "LTOE",
        "RH": "LH",
        "RK": "LK",
    }
    return part_pairs.get(upper, name)


def _partner_permutation(names: list[str]) -> np.ndarray:
    indices = {name: index for index, name in enumerate(names)}
    missing = [(name, swap_left_right_name(name)) for name in names if swap_left_right_name(name) not in indices]
    if missing:
        raise ValueError(f"left/right mirror partners are missing: {missing[:4]}")
    return np.asarray([indices[swap_left_right_name(name)] for name in names], dtype=np.int64)


def _canonical_joint_signs(joint_names: list[str]) -> np.ndarray:
    root = ET.parse(canonical_g1_urdf_path()).getroot()  # noqa: S314 - trusted canonical local asset
    axes: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name") or "")
        axis_node = joint.find("axis")
        if axis_node is None or not axis_node.get("xyz"):
            continue
        axis = np.fromstring(str(axis_node.get("xyz")), sep=" ", dtype=np.float64)
        if axis.shape == (3,) and np.linalg.norm(axis) > 1.0e-9:
            axes[name] = axis / np.linalg.norm(axis)
    signs: list[float] = []
    for name in joint_names:
        partner = swap_left_right_name(name)
        if name not in axes or partner not in axes:
            raise ValueError(f"canonical URDF has no mirrorable axis for joint {name!r}")
        transformed = -_ROBOT_LOCAL_REFLECTION @ axes[partner]
        alignment = float(np.dot(axes[name], transformed))
        if abs(abs(alignment) - 1.0) > 1.0e-6:
            raise ValueError(f"canonical mirror axes do not align for {name!r} and {partner!r}")
        signs.append(1.0 if alignment > 0.0 else -1.0)
    return np.asarray(signs, dtype=np.float64)


def _finite_difference_linear(values: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if values.shape[0] <= 1:
        return result
    result[0] = (values[1] - values[0]) * fps
    result[-1] = (values[-1] - values[-2]) * fps
    if values.shape[0] > 2:
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return result


def _canonicalize_quaternion_hemisphere_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Choose temporally continuous signs for an equivalent wxyz quaternion sequence."""

    value = np.asarray(quaternions).copy()
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"quaternion sequence must have shape [T,4], got {value.shape}")
    for frame in range(1, value.shape[0]):
        if float(np.dot(value[frame - 1], value[frame])) < 0.0:
            value[frame] *= -1.0
    return value


def _finite_difference_angular(matrices: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros(matrices.shape[:-2] + (3,), dtype=np.float64)
    if matrices.shape[0] <= 1:
        return result
    interval_delta = np.einsum("tbij,tbkj->tbik", matrices[1:], matrices[:-1])
    interval_velocity = Rotation.from_matrix(interval_delta.reshape(-1, 3, 3)).as_rotvec().reshape(
        matrices.shape[0] - 1, matrices.shape[1], 3
    ) * fps
    result[0] = interval_velocity[0]
    result[-1] = interval_velocity[-1]
    if matrices.shape[0] > 2:
        result[1:-1] = 0.5 * (interval_velocity[:-1] + interval_velocity[1:])
    return result


def _canonical_body_kinematics(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.asarray(payload["joint_pos"], dtype=np.float64)
    joint_names = [str(value) for value in np.asarray(payload["joint_names"]).reshape(-1).tolist()]
    body_names = [str(value) for value in np.asarray(payload["body_names"]).reshape(-1).tolist()]
    joint_values = {name: qpos[:, 7 + index] for index, name in enumerate(joint_names)}
    root = ET.parse(canonical_g1_urdf_path()).getroot()  # noqa: S314 - trusted canonical local asset
    canonical_links = {str(link.get("name") or "") for link in root.findall("link")}
    missing_bodies = [name for name in body_names if name not in canonical_links]
    if missing_bodies:
        raise ValueError(f"body_names are not canonical G1 links: {missing_bodies[:4]}")

    frame_count = qpos.shape[0]
    position_base: dict[str, np.ndarray] = {"pelvis": np.zeros((frame_count, 3), dtype=np.float64)}
    rotation_base: dict[str, np.ndarray] = {
        "pelvis": np.broadcast_to(np.eye(3, dtype=np.float64), (frame_count, 3, 3)).copy()
    }
    pending = list(root.findall("joint"))
    while pending:
        remaining = []
        progressed = False
        for joint in pending:
            parent_node = joint.find("parent")
            child_node = joint.find("child")
            if parent_node is None or child_node is None:
                raise ValueError("canonical URDF joint is missing parent or child")
            parent = str(parent_node.get("link") or "")
            child = str(child_node.get("link") or "")
            if parent not in position_base:
                remaining.append(joint)
                continue
            origin = joint.find("origin")
            xyz = np.fromstring(str(origin.get("xyz") if origin is not None else "0 0 0"), sep=" ")
            rpy = np.fromstring(str(origin.get("rpy") if origin is not None else "0 0 0"), sep=" ")
            if xyz.shape != (3,):
                xyz = np.zeros(3, dtype=np.float64)
            if rpy.shape != (3,):
                rpy = np.zeros(3, dtype=np.float64)
            origin_rotation = Rotation.from_euler("xyz", rpy).as_matrix()
            joint_type = str(joint.get("type") or "fixed")
            joint_name = str(joint.get("name") or "")
            local_rotation = np.broadcast_to(origin_rotation, (frame_count, 3, 3)).copy()
            if joint_type in {"revolute", "continuous"}:
                if joint_name not in joint_values:
                    raise ValueError(f"motion has no value for canonical joint {joint_name!r}")
                axis_node = joint.find("axis")
                axis = np.fromstring(
                    str(axis_node.get("xyz") if axis_node is not None else "1 0 0"), sep=" "
                )
                axis = axis / np.linalg.norm(axis)
                joint_rotation = Rotation.from_rotvec(joint_values[joint_name][:, None] * axis[None]).as_matrix()
                local_rotation = np.einsum("ij,tjk->tik", origin_rotation, joint_rotation)
            elif joint_type != "fixed":
                raise ValueError(f"unsupported canonical G1 joint type {joint_type!r}")
            parent_rotation = rotation_base[parent]
            position_base[child] = position_base[parent] + np.einsum("tij,j->ti", parent_rotation, xyz)
            rotation_base[child] = np.einsum("tij,tjk->tik", parent_rotation, local_rotation)
            progressed = True
        if not progressed:
            unresolved = [str(joint.get("name") or "") for joint in remaining]
            raise ValueError(f"could not resolve canonical G1 kinematic tree: {unresolved[:4]}")
        pending = remaining

    position = np.stack([position_base[name] for name in body_names], axis=1)
    matrices = np.stack([rotation_base[name] for name in body_names], axis=1)
    root_rotation = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    position_world = np.einsum("tij,tbj->tbi", root_rotation, position) + qpos[:, None, :3]
    rotation_world = np.einsum("tij,tbjk->tbik", root_rotation, matrices)
    quaternion_xyzw = Rotation.from_matrix(rotation_world.reshape(-1, 3, 3)).as_quat().reshape(
        frame_count, len(body_names), 4
    )
    quaternion_wxyz = quaternion_xyzw[..., [3, 0, 1, 2]]
    flat = quaternion_wxyz.reshape(frame_count, -1, 4)
    for frame in range(1, frame_count):
        flip = np.sum(flat[frame - 1] * flat[frame], axis=-1) < 0.0
        flat[frame, flip] *= -1.0
    return position_world, quaternion_wxyz, rotation_world


def _recompute_named_body_arrays(payload: dict[str, Any]) -> None:
    position, quaternion, matrices = _canonical_body_kinematics(payload)
    had_linear_velocity = "body_lin_vel_w" in payload
    had_angular_velocity = "body_ang_vel_w" in payload
    payload["body_pos_w"] = position
    payload["body_quat_w"] = quaternion
    fps = float(np.asarray(payload.get("fps", 50.0)).reshape(-1)[0])
    if had_linear_velocity:
        payload["body_lin_vel_w"] = _finite_difference_linear(position, fps)
    if had_angular_velocity:
        payload["body_ang_vel_w"] = _finite_difference_angular(matrices, fps)


def _mirror_joint_arrays(payload: dict[str, Any]) -> None:
    names = [str(value) for value in np.asarray(payload["joint_names"]).reshape(-1).tolist()]
    permutation = _partner_permutation(names)
    signs = _canonical_joint_signs(names)
    qpos = np.asarray(payload["joint_pos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(names):
        raise ValueError(f"joint_pos must contain xyz+wxyz+{len(names)} joints, got {qpos.shape}")
    mirrored_qpos = qpos.copy()
    mirrored_qpos[:, 3:7] = _canonicalize_quaternion_hemisphere_wxyz(mirrored_qpos[:, 3:7])
    mirrored_qpos[:, 7:] = qpos[:, 7 + permutation] * signs
    payload["joint_pos"] = mirrored_qpos

    if "joint_vel" not in payload:
        return
    velocity = np.asarray(payload["joint_vel"], dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 6 + len(names):
        raise ValueError(f"joint_vel must contain root linear/angular+{len(names)} joints, got {velocity.shape}")
    mirrored_velocity = velocity.copy()
    mirrored_velocity[:, 6:] = velocity[:, 6 + permutation] * signs
    payload["joint_vel"] = mirrored_velocity


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _stamp_json_provenance(payload: dict[str, Any], key: str, augmentation: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if key in payload:
        raw = np.asarray(payload[key]).item()
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, dict):
                existing = parsed
        except (TypeError, ValueError):
            existing = {"source_value": str(raw)}
    existing["motion_edit_augmentation"] = augmentation
    payload[key] = np.asarray(json.dumps(existing, sort_keys=True))


def mirror_motion_npz(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    payload = _load_npz_payload(source)
    decode_robot_asset_json(payload.get("robot_asset_json"), context=str(source))
    from somaforge_core.support_evidence import reference_only_support
    reference_only_support(payload, reason='mirrored trajectory is a reference, not an observed execution', retain_force_targets=True)
    for required in ("joint_names", "joint_pos", "body_names", "body_pos_w", "body_quat_w"):
        if required not in payload:
            raise ValueError(f"{source}: missing required motion field {required!r}")
    _mirror_joint_arrays(payload)
    _recompute_named_body_arrays(payload)
    augmentation = {
        "schema": ROBOT_MIRROR_SCHEMA,
        "kind": "robot_only_left_right",
        "source_path": str(source),
        "root_transform": "identity",
        "robot_local_reflection_matrix": _ROBOT_LOCAL_REFLECTION.tolist(),
        "obstacle_transform": "identity",
    }
    _stamp_json_provenance(payload, "kinematics_provenance_json", augmentation)
    payload["motion_edit_augmentation_json"] = np.asarray(json.dumps(augmentation, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def _mirror_contact_parts(payload: dict[str, Any], root_qpos: np.ndarray) -> None:
    part_order = [str(value) for value in np.asarray(payload["contact_force_part_order"]).reshape(-1).tolist()]
    permutation = _partner_permutation(part_order)
    for key in ("contact_force_part_mask", "contact_force_part_mask_raw", "contact_force_part_position_valid"):
        if key in payload:
            payload[key] = np.asarray(payload[key])[:, permutation]
    if "contact_force_part_w" in payload:
        payload["contact_force_part_w"] = _reflect_root_local_vectors(
            np.asarray(payload["contact_force_part_w"])[:, permutation], root_qpos
        )
    if "contact_force_part_position_w" in payload:
        payload["contact_force_part_position_w"] = _reflect_root_local_points(
            np.asarray(payload["contact_force_part_position_w"])[:, permutation], root_qpos
        )
    if "contact_force_part_history_w" in payload:
        payload["contact_force_part_history_w"] = _reflect_root_local_vectors(
            np.asarray(payload["contact_force_part_history_w"])[:, :, permutation], root_qpos
        )


def mirror_contact_force_npz(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    source_payload = _load_npz_payload(source)
    decode_robot_asset_json(source_payload.get("robot_asset_json"), context=str(source))
    root_qpos = np.asarray(source_payload["joint_pos"], dtype=np.float64)[:, :7].copy()
    payload = {key: value for key, value in source_payload.items() if not key.startswith(_RAW_CONTACT_PREFIX)}
    from somaforge_core.support_evidence import reference_only_support
    reference_only_support(payload, reason='mirrored force fields are reference targets, not measured loads', retain_force_targets=True)
    _mirror_joint_arrays(payload)
    _recompute_named_body_arrays(payload)
    _mirror_contact_parts(payload, root_qpos)
    augmentation = {
        "schema": ROBOT_MIRROR_SCHEMA,
        "kind": "robot_only_left_right",
        "source_path": str(source),
        "root_transform": "identity",
        "robot_local_reflection_matrix": _ROBOT_LOCAL_REFLECTION.tolist(),
        "obstacle_transform": "identity",
        "raw_contact_detail": "omitted_solver_specific_shape_and_body_ids",
        "aggregate_contact": "mirrored_8part_point_force_mask_and_history",
    }
    _stamp_json_provenance(payload, "kinematics_provenance_json", augmentation)
    _stamp_json_provenance(payload, "contact_force_provenance_json", augmentation)
    payload["motion_edit_augmentation_json"] = np.asarray(json.dumps(augmentation, sort_keys=True))
    payload["raw_contact_available"] = np.asarray(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def _points_on_surface(
    points: np.ndarray,
    surface: ContactSurfaceRecord,
    *,
    max_distance: float,
) -> np.ndarray:
    surface.validate()
    value = np.asarray(points, dtype=np.float64)
    origin = np.asarray(surface.origin, dtype=np.float64)
    normal = np.asarray(surface.normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent_u = np.asarray(surface.tangent_u, dtype=np.float64)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.asarray(surface.tangent_v, dtype=np.float64)
    tangent_v /= np.linalg.norm(tangent_v)
    delta = value - origin
    signed_distance = np.einsum("...i,i->...", delta, normal)
    projected_delta = delta - signed_distance[..., None] * normal
    u = np.einsum("...i,i->...", projected_delta, tangent_u)
    v = np.einsum("...i,i->...", projected_delta, tangent_v)
    accepted = np.isfinite(value).all(axis=-1) & (np.abs(signed_distance) <= float(max_distance))
    polygon = surface_polygon_uv(
        surface.metadata,
        origin=origin.tolist(),
        tangent_u=tangent_u.tolist(),
        tangent_v=tangent_v.tolist(),
    )
    if polygon:
        inside = np.asarray(
            [point_in_polygon_uv((float(u_i), float(v_i)), polygon) for u_i, v_i in zip(u.reshape(-1), v.reshape(-1))],
            dtype=bool,
        ).reshape(u.shape)
        return accepted & inside
    if surface.bounds is not None:
        u_bounds = surface.bounds.get("u")
        v_bounds = surface.bounds.get("v")
        if u_bounds is not None:
            accepted &= (u >= float(u_bounds[0])) & (u <= float(u_bounds[1]))
        if v_bounds is not None:
            accepted &= (v >= float(v_bounds[0])) & (v <= float(v_bounds[1]))
    return accepted


def filter_mirrored_contacts_to_surfaces(
    motion_path: str | Path,
    surfaces: list[ContactSurfaceRecord],
    *,
    max_distance: float = 0.08,
) -> dict[str, int]:
    """Remove mirrored force samples that no longer touch the unchanged terrain."""

    path = Path(motion_path).expanduser().resolve()
    payload = _load_npz_payload(path)
    positions = np.asarray(payload["contact_force_part_position_w"], dtype=np.float64)
    mask = np.asarray(payload["contact_force_part_mask"], dtype=bool)
    position_valid = np.asarray(payload["contact_force_part_position_valid"], dtype=bool)
    if positions.shape[:2] != mask.shape or position_valid.shape != mask.shape:
        raise ValueError("mirrored contact positions, validity, and mask must share [T,P]")
    on_surface = np.zeros(mask.shape, dtype=bool)
    for surface in surfaces:
        on_surface |= _points_on_surface(positions, surface, max_distance=max_distance)
    filtered = mask & position_valid & on_surface
    dropped = mask & ~filtered
    payload["contact_force_part_mask"] = filtered
    if "contact_force_part_mask_raw" in payload:
        payload["contact_force_part_mask_raw"] = (
            np.asarray(payload["contact_force_part_mask_raw"], dtype=bool) & position_valid & on_surface
        )
    payload["contact_force_part_position_valid"] = position_valid & filtered
    force = np.asarray(payload["contact_force_part_w"]).copy()
    force[~filtered] = 0.0
    payload["contact_force_part_w"] = force
    if "contact_force_part_history_w" in payload:
        history = np.asarray(payload["contact_force_part_history_w"]).copy()
        history *= filtered[:, None, :, None]
        payload["contact_force_part_history_w"] = history
    augmentation = {
        "schema": ROBOT_MIRROR_SCHEMA,
        "contact_redetection": "mirrored_points_within_unchanged_surface",
        "max_surface_distance_m": float(max_distance),
        "source_contact_samples": int(np.count_nonzero(mask)),
        "retained_contact_samples": int(np.count_nonzero(filtered)),
        "dropped_contact_samples": int(np.count_nonzero(dropped)),
    }
    _stamp_json_provenance(payload, "contact_force_provenance_json", augmentation)
    payload["motion_edit_contact_redetection_json"] = np.asarray(json.dumps(augmentation, sort_keys=True))
    np.savez_compressed(path, **payload)
    return {
        "source_contact_samples": int(np.count_nonzero(mask)),
        "retained_contact_samples": int(np.count_nonzero(filtered)),
        "dropped_contact_samples": int(np.count_nonzero(dropped)),
    }


def _surface_coordinates(anchor: Any, world_position: np.ndarray) -> dict[str, Any] | None:
    if anchor.surface_origin is None or anchor.surface_tangent_u is None or anchor.surface_tangent_v is None:
        return anchor.surface_coordinates
    delta = world_position - np.asarray(anchor.surface_origin, dtype=np.float64)
    coordinates = dict(anchor.surface_coordinates or {})
    coordinates["u"] = float(np.dot(delta, np.asarray(anchor.surface_tangent_u, dtype=np.float64)))
    coordinates["v"] = float(np.dot(delta, np.asarray(anchor.surface_tangent_v, dtype=np.float64)))
    return coordinates


def project_contact_to_surface(record: Any, world_position: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None, bool]:
    position = np.asarray(world_position, dtype=np.float64)
    origin_value = getattr(record, "surface_origin", None)
    normal_value = getattr(record, "surface_normal", None)
    tangent_u_value = getattr(record, "surface_tangent_u", None)
    tangent_v_value = getattr(record, "surface_tangent_v", None)
    bounds = getattr(record, "surface_bounds", None)
    if origin_value is None or normal_value is None or tangent_u_value is None or tangent_v_value is None:
        return position, getattr(record, "surface_coordinates", None), False
    origin = np.asarray(origin_value, dtype=np.float64)
    normal = np.asarray(normal_value, dtype=np.float64)
    tangent_u = np.asarray(tangent_u_value, dtype=np.float64)
    tangent_v = np.asarray(tangent_v_value, dtype=np.float64)
    projected = position - np.dot(position - origin, normal) * normal
    delta = projected - origin
    u = float(np.dot(delta, tangent_u))
    v = float(np.dot(delta, tangent_v))
    clamped = False
    if bounds is not None:
        u_clamped = float(np.clip(u, float(bounds["u"][0]), float(bounds["u"][1])))
        v_clamped = float(np.clip(v, float(bounds["v"][0]), float(bounds["v"][1])))
        clamped = abs(u_clamped - u) > 1.0e-9 or abs(v_clamped - v) > 1.0e-9
        u, v = u_clamped, v_clamped
        projected = origin + u * tangent_u + v * tangent_v
    coordinates = dict(getattr(record, "surface_coordinates", None) or {})
    coordinates.update({"u": u, "v": v})
    return projected, coordinates, clamped


def mirror_contact_graph(graph: ContactGraph, *, source_root_qpos: np.ndarray) -> ContactGraph:
    root_qpos = np.asarray(source_root_qpos, dtype=np.float64)
    anchor_ids: dict[str, str] = {}
    anchors = []
    for anchor in graph.anchors:
        anchor_id = f"mirror_lr::{anchor.anchor_id}"
        anchor_ids[anchor.anchor_id] = anchor_id
        position = anchor.world_position
        coordinates = anchor.surface_coordinates
        clamped = False
        if anchor.world_position is not None:
            representative = min(
                max((int(anchor.start_frame) + int(anchor.end_frame) - 1) // 2, 0),
                root_qpos.shape[0] - 1,
            )
            reflected = _reflect_root_local_points(
                np.asarray(anchor.world_position, dtype=np.float64)[None],
                root_qpos[representative : representative + 1],
            )[0]
            projected, coordinates, clamped = project_contact_to_surface(anchor, reflected)
            position = projected.tolist()
        anchors.append(
            replace(
                anchor,
                anchor_id=anchor_id,
                body=swap_left_right_name(anchor.body),
                world_position=position,
                object_position=None,
                surface_coordinates=coordinates,
                patch_id=None,
                position_source="robot_only_mirror",
                source="robot_only_mirror",
                metadata={
                    "schema": ROBOT_MIRROR_SCHEMA,
                    "source_anchor_id": anchor.anchor_id,
                    "source_body": anchor.body,
                    "mean_drift_xy": anchor.metadata.get("mean_drift_xy"),
                    "root_transform": "identity",
                    "world_contact_transform": "robot_local_reflection_then_same_surface_projection",
                    "surface_clamped": bool(clamped),
                    "obstacle_transform": "identity",
                },
            )
        )

    event_ids = {event.event_id: f"mirror_lr::{event.event_id}" for event in graph.events}
    events = [
        replace(
            event,
            event_id=event_ids[event.event_id],
            body=swap_left_right_name(event.body),
            contact_before=[swap_left_right_name(value) for value in event.contact_before],
            contact_after=[swap_left_right_name(value) for value in event.contact_after],
            source="robot_only_mirror",
            metadata={"schema": ROBOT_MIRROR_SCHEMA, "source_event_id": event.event_id},
        )
        for event in graph.events
    ]
    transitions = [
        replace(
            transition,
            transition_id=f"mirror_lr::{transition.transition_id}",
            active_body=(swap_left_right_name(transition.active_body) if transition.active_body else None),
            support_bodies=[swap_left_right_name(value) for value in transition.support_bodies],
            start_event_id=event_ids.get(transition.start_event_id) if transition.start_event_id else None,
            end_event_id=event_ids.get(transition.end_event_id) if transition.end_event_id else None,
            source_anchor_id=anchor_ids.get(transition.source_anchor_id) if transition.source_anchor_id else None,
            target_anchor_id=anchor_ids.get(transition.target_anchor_id) if transition.target_anchor_id else None,
            source="robot_only_mirror",
            metadata={"schema": ROBOT_MIRROR_SCHEMA, "source_transition_id": transition.transition_id},
        )
        for transition in graph.transitions
    ]
    return ContactGraph(
        motion_id=graph.motion_id,
        events=events,
        anchors=anchors,
        patches=patches_from_anchors(anchors),
        transitions=transitions,
    )


__all__ = [
    "ROBOT_MIRROR_SCHEMA",
    "filter_mirrored_contacts_to_surfaces",
    "mirror_contact_force_npz",
    "mirror_contact_graph",
    "mirror_motion_npz",
    "project_contact_to_surface",
    "swap_left_right_name",
]
