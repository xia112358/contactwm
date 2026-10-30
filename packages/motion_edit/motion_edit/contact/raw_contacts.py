from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.patches import patches_from_anchors
from motion_edit.contact.schema import ContactAnchorRecord, ContactSurfaceRecord
from motion_edit.contact.surface_geometry import point_in_polygon_uv, surface_polygon_uv

PART_ALIASES = {
    # Whole-foot aliases remain valid for raw-contact diagnostics. Production
    # force/contact channels are still the canonical heel/toe 8-part schema.
    "lf": ("lf", "left_foot", "left_ankle", "left_ankle_roll_link"),
    "rf": ("rf", "right_foot", "right_ankle", "right_ankle_roll_link"),
    "lhee": ("lhee", "left_heel"),
    "ltoe": ("ltoe", "left_toe"),
    "rhee": ("rhee", "right_heel"),
    "rtoe": ("rtoe", "right_toe"),
    "lh": ("lh", "left_hand", *CONTACT_BODY_NAMES_BY_PART["left_hand"]),
    "rh": ("rh", "right_hand", *CONTACT_BODY_NAMES_BY_PART["right_hand"]),
    "lk": ("lk", "left_knee", "left_knee_link"),
    "rk": ("rk", "right_knee", "right_knee_link"),
}

UP_DOT_THRESHOLD = 0.5
EDGE_DISTANCE_THRESHOLD = 0.05
FOOT_PARTS = {
    "left_foot",
    "left_heel",
    "left_toe",
    "left_sole",
    "right_foot",
    "right_heel",
    "right_toe",
    "right_sole",
    "lf",
    "rf",
    "lhee",
    "ltoe",
    "rhee",
    "rtoe",
}


@dataclass(frozen=True)
class _PreparedSurface:
    record: ContactSurfaceRecord
    origin: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    polygon: list[tuple[float, float]]


class RawContactMotion:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.data = np.load(self.path, allow_pickle=True)
        required = [
            "raw_contact_count",
            "raw_contact_point0_w",
            "raw_contact_point1_w",
            "contact_force_part_order",
        ]
        missing = [name for name in required if name not in self.data]
        if missing:
            raise ValueError(f"{self.path}: missing raw contact fields: {', '.join(missing)}")
        self.raw_contact_count = np.asarray(self.data["raw_contact_count"], dtype=np.int64)
        self.raw_contact_point0_w = np.asarray(self.data["raw_contact_point0_w"], dtype=np.float64)
        self.raw_contact_point1_w = np.asarray(self.data["raw_contact_point1_w"], dtype=np.float64)
        self.raw_contact_force_w = (
            np.asarray(self.data["raw_contact_force_w"], dtype=np.float64)
            if "raw_contact_force_w" in self.data
            else None
        )
        self.raw_contact_shape1 = (
            np.asarray(self.data["raw_contact_shape1"], dtype=np.int64)
            if "raw_contact_shape1" in self.data
            else None
        )
        self.raw_contact_shape0 = (
            np.asarray(self.data["raw_contact_shape0"], dtype=np.int64)
            if "raw_contact_shape0" in self.data
            else None
        )
        pairing = (
            str(self.data["raw_contact_point_pairing"].item())
            if "raw_contact_point_pairing" in self.data
            else ""
        )
        source = (
            str(self.data["raw_contact_source"].item())
            if "raw_contact_source" in self.data
            else ""
        )
        robot_is_zero = pairing == "body0_point0_body1_point1" or (
            not pairing
            and source == "multi_rollout_stable_label_median_external_contacts"
        )
        if robot_is_zero:
            self.robot_contact_point_w = self.raw_contact_point0_w
            self.surface_contact_point_w = self.raw_contact_point1_w
            self.robot_contact_shape = self.raw_contact_shape0
        else:
            self.robot_contact_point_w = self.raw_contact_point1_w
            self.surface_contact_point_w = self.raw_contact_point0_w
            self.robot_contact_shape = self.raw_contact_shape1
        self.part_order = [str(item).lower() for item in np.asarray(self.data["contact_force_part_order"]).tolist()]
        self.part_position_w = (
            np.asarray(self.data["contact_force_part_position_w"], dtype=np.float64)
            if "contact_force_part_position_w" in self.data
            else _part_positions_from_body_fk(self.data, self.part_order)
        )
        self.part_index_by_alias = _part_index_by_alias(self.part_order)
        self.nearest_part_index, self.nearest_part_distance = self._compute_nearest_parts()

    def _compute_nearest_parts(self) -> tuple[np.ndarray, np.ndarray]:
        nearest = np.full(self.robot_contact_point_w.shape[:2], -1, dtype=np.int16)
        nearest_distance = np.full(self.robot_contact_point_w.shape[:2], np.inf, dtype=np.float32)
        for frame, raw_count in enumerate(self.raw_contact_count):
            count = min(int(raw_count), self.raw_contact_point1_w.shape[1])
            if count <= 0:
                continue
            robot_points = self.robot_contact_point_w[frame, :count]
            part_positions = self.part_position_w[frame]
            distances = np.linalg.norm(robot_points[:, None, :] - part_positions[None, :, :], axis=-1)
            frame_nearest = np.argmin(distances, axis=-1)
            nearest[frame, :count] = frame_nearest.astype(np.int16)
            nearest_distance[frame, :count] = distances[np.arange(count), frame_nearest].astype(np.float32)
        return nearest, nearest_distance


def _part_positions_from_body_fk(
    motion: Any,
    part_order: list[str],
) -> np.ndarray:
    required = ("body_pos_w", "body_names")
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(
            "raw contact refinement requires contact_force_part_position_w or "
            f"canonical body FK fields: {', '.join(missing)}"
        )
    body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
    body_names = [str(value) for value in np.asarray(motion["body_names"]).tolist()]
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(
            f"body_pos_w must have shape [T,B,3], got {body_pos.shape}"
        )
    if body_pos.shape[1] != len(body_names):
        raise ValueError(
            "body_pos_w/body_names length mismatch: "
            f"{body_pos.shape[1]} != {len(body_names)}"
        )
    index_by_name = {name.lower(): index for index, name in enumerate(body_names)}
    trajectories: list[np.ndarray] = []
    for raw_part in part_order:
        part = str(raw_part).lower()
        aliases = PART_ALIASES.get(part, (part,))
        canonical_part = next(
            (
                candidate
                for candidate in (
                    "left_heel",
                    "left_toe",
                    "right_heel",
                    "right_toe",
                    "left_hand",
                    "right_hand",
                    "left_knee",
                    "right_knee",
                )
                if candidate in aliases
            ),
            part,
        )
        physical_names = CONTACT_BODY_NAMES_BY_PART.get(canonical_part, ())
        indices = [
            index_by_name[name.lower()]
            for name in physical_names
            if name.lower() in index_by_name
        ]
        if not indices:
            indices = [
                index_by_name[alias.lower()]
                for alias in aliases
                if alias.lower() in index_by_name
            ]
        if not indices:
            raise ValueError(
                f"cannot derive raw-contact assignment position for {raw_part!r} "
                "from canonical body_names"
            )
        trajectories.append(np.mean(body_pos[:, indices, :], axis=1))
    return np.stack(trajectories, axis=1)


def _part_index_by_alias(part_order: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, part in enumerate(part_order):
        aliases = PART_ALIASES.get(part.lower(), (part.lower(),))
        for alias in aliases:
            out[alias] = index
    return out


def _drop_short_near_duplicate_subanchors(
    anchors: list[ContactAnchorRecord],
    *,
    min_duration: int = 5,
    max_distance: float = 0.03,
) -> list[ContactAnchorRecord]:
    if len(anchors) <= 1:
        return anchors
    ordered = sorted(anchors, key=lambda item: (item.start_frame, item.end_frame, item.anchor_id))
    kept: list[ContactAnchorRecord] = []
    for index, anchor in enumerate(ordered):
        duration = int(anchor.end_frame) - int(anchor.start_frame)
        previous_anchor = kept[-1] if kept else None
        next_anchor = ordered[index + 1] if index + 1 < len(ordered) else None
        if duration < min_duration and (
            _near_anchor(anchor, previous_anchor, max_distance=max_distance)
            or _near_anchor(anchor, next_anchor, max_distance=max_distance)
        ):
            continue
        kept.append(anchor)
    return kept


def _near_anchor(left: ContactAnchorRecord, right: ContactAnchorRecord | None, *, max_distance: float) -> bool:
    if right is None or left.body != right.body or left.surface_id != right.surface_id:
        return False
    if left.metadata.get("patch_role") != right.metadata.get("patch_role"):
        return False
    if not (
        int(right.start_frame) <= int(left.start_frame)
        and int(right.end_frame) >= int(left.end_frame)
    ):
        return False
    if left.world_position is None or right.world_position is None:
        return False
    distance = float(np.linalg.norm(np.asarray(left.world_position, dtype=np.float64) - np.asarray(right.world_position, dtype=np.float64)))
    return distance <= max_distance


def _anchor_part_index(anchor: ContactAnchorRecord, raw: RawContactMotion) -> int | None:
    body = anchor.body.lower()
    if body in raw.part_index_by_alias:
        return raw.part_index_by_alias[body]
    for canonical, aliases in PART_ALIASES.items():
        if any(alias in body for alias in aliases):
            return raw.part_index_by_alias.get(canonical)
    return None


def _is_foot_anchor(anchor: ContactAnchorRecord) -> bool:
    body = anchor.body.lower()
    return body in FOOT_PARTS or "foot" in body or "ankle" in body or "heel" in body or "toe" in body


def _prepare_surface(surface: ContactSurfaceRecord, *, require_polygon: bool) -> _PreparedSurface:
    surface.validate()
    polygon = (
        surface_polygon_uv(
            surface.metadata,
            origin=surface.origin,
            tangent_u=surface.tangent_u,
            tangent_v=surface.tangent_v,
        )
        if require_polygon
        else []
    )
    return _PreparedSurface(
        record=surface,
        origin=np.asarray(surface.origin, dtype=np.float64),
        normal=np.asarray(surface.normal, dtype=np.float64),
        tangent_u=np.asarray(surface.tangent_u, dtype=np.float64),
        tangent_v=np.asarray(surface.tangent_v, dtype=np.float64),
        polygon=polygon,
    )


def _project_sample_to_surface(
    point: np.ndarray,
    surface: ContactSurfaceRecord,
    max_distance: float,
    *,
    require_polygon: bool = True,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    surface.validate()
    origin = np.asarray(surface.origin, dtype=np.float64)
    normal = np.asarray(surface.normal, dtype=np.float64)
    tangent_u = np.asarray(surface.tangent_u, dtype=np.float64)
    tangent_v = np.asarray(surface.tangent_v, dtype=np.float64)
    signed_distance = float(np.dot(point - origin, normal))
    if abs(signed_distance) > max_distance:
        return None
    projected = point - signed_distance * normal
    local = projected - origin
    u = float(np.dot(local, tangent_u))
    v = float(np.dot(local, tangent_v))
    polygon = (
        surface_polygon_uv(
            surface.metadata,
            origin=surface.origin,
            tangent_u=surface.tangent_u,
            tangent_v=surface.tangent_v,
        )
        if require_polygon
        else []
    )
    if require_polygon and surface.surface_type == "mesh_face" and not polygon:
        return None
    inside = point_in_polygon_uv((u, v), polygon) if polygon else _inside_bounds(u, v, surface.bounds)
    if not inside:
        return None
    metadata = {
        "surface_id": surface.surface_id,
        "object_id": surface.object_id,
        "surface_type": surface.surface_type,
        "signed_surface_distance": signed_distance,
        "surface_coordinates": {"u": u, "v": v},
    }
    return projected, metadata


def _project_sample_to_prepared_surface(
    point: np.ndarray,
    surface: _PreparedSurface,
    max_distance: float,
    *,
    require_polygon: bool,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    record = surface.record
    signed_distance = float(np.dot(point - surface.origin, surface.normal))
    if abs(signed_distance) > max_distance:
        return None
    projected = point - signed_distance * surface.normal
    local = projected - surface.origin
    u = float(np.dot(local, surface.tangent_u))
    v = float(np.dot(local, surface.tangent_v))
    if require_polygon and record.surface_type == "mesh_face" and not surface.polygon:
        return None
    inside = point_in_polygon_uv((u, v), surface.polygon) if surface.polygon else _inside_bounds(u, v, record.bounds)
    if not inside:
        return None
    metadata = {
        "surface_id": record.surface_id,
        "object_id": record.object_id,
        "surface_type": record.surface_type,
        "signed_surface_distance": signed_distance,
        "surface_coordinates": {"u": u, "v": v},
    }
    return projected, metadata


def _sample_surface_relation(point: np.ndarray, surface: ContactSurfaceRecord) -> dict[str, Any] | None:
    surface.validate()
    origin = np.asarray(surface.origin, dtype=np.float64)
    normal = np.asarray(surface.normal, dtype=np.float64)
    tangent_u = np.asarray(surface.tangent_u, dtype=np.float64)
    tangent_v = np.asarray(surface.tangent_v, dtype=np.float64)
    signed_distance = float(np.dot(point - origin, normal))
    projected = point - signed_distance * normal
    local = projected - origin
    u = float(np.dot(local, tangent_u))
    v = float(np.dot(local, tangent_v))
    polygon = surface_polygon_uv(
        surface.metadata,
        origin=surface.origin,
        tangent_u=surface.tangent_u,
        tangent_v=surface.tangent_v,
    )
    if polygon:
        inside = point_in_polygon_uv((u, v), polygon)
        distance_to_boundary = _distance_to_polygon_boundary((u, v), polygon)
    else:
        inside = _inside_bounds(u, v, surface.bounds)
        distance_to_boundary = _distance_to_bounds_boundary(u, v, surface.bounds)
    return {
        "surface_id": surface.surface_id,
        "surface_type": surface.surface_type,
        "normal": surface.normal,
        "normal_z": float(surface.normal[2]),
        "signed_surface_distance": signed_distance,
        "abs_surface_distance": abs(signed_distance),
        "surface_coordinates": {"u": u, "v": v},
        "inside": inside,
        "distance_to_boundary": distance_to_boundary,
    }


def _distance_to_polygon_boundary(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float | None:
    if len(polygon) < 2:
        return None
    return min(_point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon)))


def _point_segment_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    closest = _closest_point_on_segment(point, a, b)
    return float(np.hypot(point[0] - closest[0], point[1] - closest[1]))


def _closest_point_on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (point[0] - a[0], point[1] - a[1])
    denom = ab[0] * ab[0] + ab[1] * ab[1]
    if denom == 0.0:
        return a
    t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / denom))
    return (a[0] + t * ab[0], a[1] + t * ab[1])


def _distance_to_bounds_boundary(u: float, v: float, bounds: dict[str, Any] | None) -> float | None:
    if bounds is None:
        return None
    u_bounds = bounds.get("u")
    v_bounds = bounds.get("v")
    if not isinstance(u_bounds, list) or len(u_bounds) != 2:
        return None
    if not isinstance(v_bounds, list) or len(v_bounds) != 2:
        return None
    return min(abs(u - float(u_bounds[0])), abs(u - float(u_bounds[1])), abs(v - float(v_bounds[0])), abs(v - float(v_bounds[1])))


def _inside_bounds(u: float, v: float, bounds: dict[str, Any] | None) -> bool:
    if bounds is None:
        return True
    u_bounds = bounds.get("u")
    v_bounds = bounds.get("v")
    if not isinstance(u_bounds, list) or len(u_bounds) != 2:
        return False
    if not isinstance(v_bounds, list) or len(v_bounds) != 2:
        return False
    return float(u_bounds[0]) <= u <= float(u_bounds[1]) and float(v_bounds[0]) <= v <= float(v_bounds[1])


def _classify_surface_candidates(
    *,
    assigned_points: list[np.ndarray],
    accepted_count: int,
    surfaces: list[ContactSurfaceRecord] | None,
    max_surface_distance: float,
) -> dict[str, Any]:
    if accepted_count > 0:
        if surfaces and len(surfaces) == 1:
            surface = surfaces[0]
            if surface.surface_id == "terrain_ground_z0":
                return {"binding_candidate_class": "ground"}
            if float(surface.normal[2]) > UP_DOT_THRESHOLD:
                return {"binding_candidate_class": "top"}
            return {"binding_candidate_class": "side_filtered"}
        accepted_relations = [
            relation
            for point in assigned_points
            for surface in surfaces or []
            if (relation := _sample_surface_relation(point, surface)) is not None
            and bool(relation["inside"])
            and float(relation["abs_surface_distance"]) <= max_surface_distance
        ]
        if accepted_relations:
            hits = _count_by_surface(accepted_relations)
            selected_surface_id = max(hits.items(), key=lambda item: item[1])[0]
            selected = next(relation for relation in accepted_relations if relation["surface_id"] == selected_surface_id)
            if selected_surface_id == "terrain_ground_z0":
                return {"binding_candidate_class": "ground"}
            if float(selected["normal_z"]) > UP_DOT_THRESHOLD:
                return {"binding_candidate_class": "top"}
            return {"binding_candidate_class": "side_filtered"}
        return {"binding_candidate_class": "top"}
    if not assigned_points:
        return {"binding_candidate_class": "raw_missing"}
    if not surfaces:
        return {"binding_candidate_class": "outside_known_surfaces"}
    relations = [
        relation
        for point in assigned_points
        for surface in surfaces
        if (relation := _sample_surface_relation(point, surface)) is not None
    ]
    near_relations = [relation for relation in relations if float(relation["abs_surface_distance"]) <= max_surface_distance]
    if not near_relations:
        return {"binding_candidate_class": "outside_known_surfaces"}
    side_hits = [
        relation
        for relation in near_relations
        if abs(float(relation["normal_z"])) <= UP_DOT_THRESHOLD and bool(relation["inside"])
    ]
    if side_hits:
        return {
            "binding_candidate_class": "side_filtered",
            "side_filtered_surface_hits": _count_by_surface(side_hits),
            "side_filtered_example": side_hits[0],
        }
    edge_hits = [
        relation
        for relation in near_relations
        if float(relation["normal_z"]) > UP_DOT_THRESHOLD
        and relation.get("distance_to_boundary") is not None
        and float(relation["distance_to_boundary"]) <= EDGE_DISTANCE_THRESHOLD
    ]
    if edge_hits:
        return {
            "binding_candidate_class": "edge_candidate",
            "edge_candidate_surface_hits": _count_by_surface(edge_hits),
            "edge_candidate_example": edge_hits[0],
        }
    return {
        "binding_candidate_class": "outside_known_surfaces",
        "nearest_surface_examples": sorted(near_relations, key=lambda item: float(item["abs_surface_distance"]))[:5],
    }


def _count_by_surface(relations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation in relations:
        surface_id = str(relation["surface_id"])
        counts[surface_id] = counts.get(surface_id, 0) + 1
    return counts


def estimate_anchor_position_from_raw_contacts(
    anchor: ContactAnchorRecord,
    raw: RawContactMotion,
    *,
    surfaces: list[ContactSurfaceRecord] | None = None,
    max_part_distance: float = 0.25,
    max_surface_distance: float = 0.05,
) -> tuple[list[float] | None, dict[str, Any]]:
    part_index = _anchor_part_index(anchor, raw)
    metadata: dict[str, Any] = {
        "raw_contact_source": str(raw.data["raw_contact_source"]) if "raw_contact_source" in raw.data else "raw_contact",
        "anchor_body": anchor.body,
        "part_index": part_index,
        "part_order": raw.part_order,
    }
    if part_index is None:
        metadata["failure_reason"] = "anchor_body_not_in_contact_force_part_order"
        return None, metadata
    start = max(0, int(anchor.start_frame))
    end = min(int(anchor.end_frame), raw.raw_contact_count.shape[0])
    if end <= start:
        metadata["failure_reason"] = "anchor_interval_outside_motion"
        return None, metadata

    samples: list[np.ndarray] = []
    accepted_robot_points: list[np.ndarray] = []
    accepted_surface_points: list[np.ndarray] = []
    accepted_forces: list[np.ndarray] = []
    accepted_shapes: list[int] = []
    accepted_frames: list[int] = []
    raw_samples = 0
    assigned_samples = 0
    rejected_surface_samples = 0
    nearest_distances: list[float] = []
    surface_hits: dict[str, int] = {}
    surface_sample_metadata: list[dict[str, Any]] = []
    assigned_surface_points: list[np.ndarray] = []
    candidate_surfaces = _surfaces_for_anchor(anchor, surfaces)
    require_surface_polygon = not bool(anchor.surface_id and candidate_surfaces)
    prepared_surfaces = (
        [_prepare_surface(surface, require_polygon=require_surface_polygon) for surface in candidate_surfaces]
        if candidate_surfaces
        else None
    )
    for frame in range(start, end):
        count = int(raw.raw_contact_count[frame])
        if count <= 0:
            continue
        count = min(count, raw.raw_contact_point0_w.shape[1])
        part_positions = raw.part_position_w[frame]
        for contact_index in range(count):
            raw_samples += 1
            robot_point = raw.robot_contact_point_w[frame, contact_index]
            nearest = int(raw.nearest_part_index[frame, contact_index])
            nearest_distance = float(raw.nearest_part_distance[frame, contact_index])
            if nearest != part_index or nearest_distance > max_part_distance:
                continue
            assigned_samples += 1
            nearest_distances.append(nearest_distance)
            surface_point = raw.surface_contact_point_w[frame, contact_index]
            assigned_surface_points.append(surface_point)
            if prepared_surfaces:
                projected_candidates = [
                    candidate
                    for surface in prepared_surfaces
                    if (
                        candidate := _project_sample_to_prepared_surface(
                            surface_point,
                            surface,
                            max_surface_distance,
                            require_polygon=require_surface_polygon,
                        )
                    )
                    is not None
                ]
                if not projected_candidates:
                    rejected_surface_samples += 1
                    continue
                projected, sample_meta = min(projected_candidates, key=lambda item: abs(float(item[1]["signed_surface_distance"])))
                samples.append(projected)
                accepted_surface_points.append(projected)
                surface_id = str(sample_meta["surface_id"])
                surface_hits[surface_id] = surface_hits.get(surface_id, 0) + 1
                surface_sample_metadata.append(sample_meta)
            else:
                samples.append(surface_point)
                accepted_surface_points.append(surface_point)
            accepted_robot_points.append(robot_point)
            accepted_forces.append(
                np.asarray(raw.raw_contact_force_w[frame, contact_index], dtype=np.float64)
                if raw.raw_contact_force_w is not None
                else np.zeros(3, dtype=np.float64)
            )
            accepted_shapes.append(
                int(raw.robot_contact_shape[frame, contact_index])
                if raw.robot_contact_shape is not None
                else -1
            )
            accepted_frames.append(frame)

    metadata.update(
        {
            "raw_contact_frame_start": start,
            "raw_contact_frame_end": end,
            "raw_contact_sample_count": raw_samples,
            "assigned_raw_contact_sample_count": assigned_samples,
            "accepted_raw_contact_sample_count": len(samples),
            "rejected_surface_sample_count": rejected_surface_samples,
            "max_part_distance": max_part_distance,
            "max_surface_distance": max_surface_distance,
        }
    )
    metadata.update(
        _classify_surface_candidates(
            assigned_points=assigned_surface_points,
            accepted_count=len(samples),
            surfaces=candidate_surfaces,
            max_surface_distance=max_surface_distance,
        )
    )
    if nearest_distances:
        distances = np.asarray(nearest_distances, dtype=np.float64)
        metadata["nearest_part_distance_median"] = float(np.median(distances))
        metadata["nearest_part_distance_max"] = float(np.max(distances))
    if surface_hits:
        metadata["surface_hits"] = surface_hits
        metadata["selected_surface_id"] = max(surface_hits.items(), key=lambda item: item[1])[0]

    if not samples:
        metadata["failure_reason"] = "no_raw_contact_points_inside_surface_polygon" if surfaces else "no_assigned_raw_contact_points"
        return None, metadata

    positions = np.asarray(samples, dtype=np.float64)
    position = np.median(positions, axis=0)
    metadata["mean_raw_contact_point0_w"] = np.mean(positions, axis=0).tolist()
    metadata["median_raw_contact_point0_w"] = position.tolist()
    metadata["first_raw_contact_point0_w"] = positions[0].tolist()
    metadata["last_raw_contact_point0_w"] = positions[-1].tolist()
    if surface_sample_metadata:
        metadata["surface_sample_metadata"] = surface_sample_metadata[:10]
    if _is_foot_anchor(anchor):
        heel_toe = estimate_heel_toe_contact_summary(
            anchor=anchor,
            robot_points=accepted_robot_points,
            surface_points=accepted_surface_points,
            forces=accepted_forces,
            shape_ids=accepted_shapes,
            frames=accepted_frames,
        )
        if heel_toe is not None:
            metadata["foot_contact_summary"] = heel_toe
    return position.tolist(), metadata


def _surfaces_for_anchor(anchor: ContactAnchorRecord, surfaces: list[ContactSurfaceRecord] | None) -> list[ContactSurfaceRecord] | None:
    if not surfaces:
        return None
    if anchor.surface_id:
        selected = [surface for surface in surfaces if surface.surface_id == anchor.surface_id]
        if selected:
            return selected
    return surfaces


def estimate_heel_toe_contact_summary(
    *,
    anchor: ContactAnchorRecord,
    robot_points: list[np.ndarray],
    surface_points: list[np.ndarray],
    forces: list[np.ndarray],
    shape_ids: list[int],
    frames: list[int],
) -> dict[str, Any] | None:
    if len(surface_points) < 2:
        return None
    surface_arr = np.asarray(surface_points, dtype=np.float64)
    robot_arr = np.asarray(robot_points, dtype=np.float64)
    force_arr = np.asarray(forces, dtype=np.float64) if forces else np.zeros_like(surface_arr)
    rel = robot_arr - np.median(robot_arr, axis=0, keepdims=True)
    axis = _principal_horizontal_axis(rel)
    coord = rel @ axis
    split = _largest_gap_split(coord)
    is_toe_positive = _toe_positive_for_body(anchor.body)
    role_by_sample = np.asarray(
        [
            _shape_heel_toe_role(shape)
            or ("toe" if (value > split if is_toe_positive else value < split) else "heel")
            for value, shape in zip(coord, shape_ids)
        ],
        dtype=object,
    )
    groups = {
        "heel": role_by_sample == "heel",
        "toe": role_by_sample == "toe",
    }
    intervals = _foot_role_intervals(role_by_sample.tolist(), frames)
    contacts: dict[str, Any] = {}
    for name, mask in groups.items():
        if int(np.count_nonzero(mask)) == 0:
            continue
        contacts[name] = _foot_subcontact_stats(
            name=name,
            surface_points=surface_arr[mask],
            robot_points=robot_arr[mask],
            forces=force_arr[mask],
            shape_ids=[shape_ids[index] for index, value in enumerate(mask) if bool(value)],
            frames=[frames[index] for index, value in enumerate(mask) if bool(value)],
            axis_coordinates=coord[mask],
        )
    if not contacts:
        return None
    return {
        "schema_version": 1,
        "method": "raw_contact_pca_heel_toe",
        "binding_granularity": "heel_toe_point_contacts",
        "axis_world": axis.tolist(),
        "axis_split": split,
        "toe_positive": is_toe_positive,
        "sample_count": int(len(surface_points)),
        "intervals": intervals,
        "contacts": contacts,
    }


def _principal_horizontal_axis(points: np.ndarray) -> np.ndarray:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    if xy.shape[0] < 2 or float(np.linalg.norm(xy)) <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_xy = vh[0]
    if float(np.linalg.norm(axis_xy)) <= 1.0e-12:
        axis_xy = np.asarray([1.0, 0.0], dtype=np.float64)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if axis[0] < 0.0 or (abs(float(axis[0])) <= 1.0e-9 and axis[1] < 0.0):
        axis = -axis
    return axis


def _largest_gap_split(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size < 2:
        return float(np.median(ordered)) if ordered.size else 0.0
    gaps = np.diff(ordered)
    index = int(np.argmax(gaps))
    if float(gaps[index]) <= 1.0e-9:
        return float(np.median(ordered))
    return float(0.5 * (ordered[index] + ordered[index + 1]))


def _foot_role_intervals(sample_roles: list[str], frames: list[int]) -> list[dict[str, Any]]:
    roles_by_frame: dict[int, set[str]] = {}
    for role, frame in zip(sample_roles, frames):
        roles_by_frame.setdefault(int(frame), set()).add(str(role))
    intervals: list[dict[str, Any]] = []
    active_role: str | None = None
    active_start: int | None = None
    previous_frame: int | None = None
    for frame in sorted(roles_by_frame):
        frame_roles = roles_by_frame[frame]
        role = "sole" if "toe" in frame_roles and "heel" in frame_roles else next(iter(frame_roles))
        if active_role is None:
            active_role = role
            active_start = frame
        elif role != active_role or previous_frame is None or frame > previous_frame + 1:
            intervals.append({"patch_role": active_role, "frame_start": active_start, "frame_end": int(previous_frame) + 1})
            active_role = role
            active_start = frame
        previous_frame = frame
    if active_role is not None and active_start is not None and previous_frame is not None:
        intervals.append({"patch_role": active_role, "frame_start": active_start, "frame_end": int(previous_frame) + 1})
    return intervals


def _toe_positive_for_body(body: str) -> bool:
    # In the current G1 rollout convention, larger coordinate along the
    # principal foot axis corresponds to the front/toe cluster. If future
    # assets provide an explicit foot frame, this should use that frame.
    return True


def _shape_heel_toe_role(shape_id: int) -> str | None:
    # Current G1 foot contact spheres are ordered from heel/midfoot to toe.
    # Raw Newton contact shape IDs are more stable than PCA sign for deciding
    # whether a short foot subcontact is toe-first or heel-first.
    shape = int(shape_id)
    if shape in {43, 44, 45}:
        return "toe"
    if shape in {39, 40, 41, 42}:
        return "heel"
    return None


def _foot_subcontact_stats(
    *,
    name: str,
    surface_points: np.ndarray,
    robot_points: np.ndarray,
    forces: np.ndarray,
    shape_ids: list[int],
    frames: list[int],
    axis_coordinates: np.ndarray,
) -> dict[str, Any]:
    unique_shapes = sorted({int(shape) for shape in shape_ids if int(shape) >= 0})
    force_norm = np.linalg.norm(forces, axis=1) if len(forces) else np.asarray([], dtype=np.float64)
    return {
        "name": name,
        "world_position": np.median(surface_points, axis=0).tolist(),
        "robot_position_world": np.median(robot_points, axis=0).tolist(),
        "sample_count": int(surface_points.shape[0]),
        "frame_start": int(min(frames)) if frames else None,
        "frame_end": int(max(frames) + 1) if frames else None,
        "raw_shape_ids": unique_shapes,
        "mean_force_w": np.mean(forces, axis=0).tolist() if len(forces) else [0.0, 0.0, 0.0],
        "force_norm_mean": float(np.mean(force_norm)) if len(force_norm) else 0.0,
        "axis_coordinate_median": float(np.median(axis_coordinates)),
        "confidence": float(min(1.0, surface_points.shape[0] / 20.0)),
    }


def refine_contact_graph_anchor_positions_from_raw_contacts(
    graph: ContactGraph,
    motion_path: str | Path,
    *,
    surfaces: list[ContactSurfaceRecord] | None = None,
    max_part_distance: float = 0.25,
    max_surface_distance: float = 0.05,
) -> ContactGraph:
    raw = RawContactMotion(motion_path)
    anchors: list[ContactAnchorRecord] = []
    for anchor in graph.anchors:
        position, raw_metadata = estimate_anchor_position_from_raw_contacts(
            anchor,
            raw,
            surfaces=surfaces,
            max_part_distance=max_part_distance,
            max_surface_distance=max_surface_distance,
        )
        metadata = dict(anchor.metadata)
        metadata["raw_contact_position_refinement"] = raw_metadata
        if position is None:
            metadata["raw_contact_position_refinement_failed"] = True
            anchors.append(replace(anchor, metadata=metadata))
            continue
        metadata["raw_contact_position_refinement_failed"] = False
        anchors.append(
            replace(
                anchor,
                world_position=position,
                position_source="raw_contact_point0_w_polygon_median" if surfaces else "raw_contact_point0_w_median",
                metadata=metadata,
            )
        )
    return replace(graph, anchors=anchors, patches=patches_from_anchors(anchors))


def split_foot_contact_anchors(graph: ContactGraph, *, source: str = "raw_contact_heel_toe_split") -> ContactGraph:
    """Split foot anchors with raw heel/toe summaries into editable sub-anchors.

    The main contact mask stays fixed to the six canonical parts. This only
    creates multiple anchors for one foot when raw rigid contacts show distinct
    heel/toe contact patches over different frame intervals.
    """

    anchors: list[ContactAnchorRecord] = []
    for anchor in graph.anchors:
        refinement = anchor.metadata.get("raw_contact_position_refinement")
        summary = refinement.get("foot_contact_summary") if isinstance(refinement, dict) else None
        contacts = summary.get("contacts") if isinstance(summary, dict) else None
        if not _is_foot_anchor(anchor) or not isinstance(contacts, dict):
            anchors.append(anchor)
            continue
        subanchors = _subanchors_from_foot_contact_summary(anchor, contacts=contacts, source=source)
        anchors.extend(subanchors or [anchor])
    anchors = sorted(anchors, key=lambda item: (item.start_frame, item.end_frame, item.body, item.anchor_id))
    return replace(graph, anchors=anchors, patches=patches_from_anchors(anchors))


def _subanchors_from_foot_contact_summary(
    anchor: ContactAnchorRecord,
    *,
    contacts: dict[str, Any],
    source: str,
) -> list[ContactAnchorRecord]:
    out: list[ContactAnchorRecord] = []
    intervals = anchor.metadata.get("raw_contact_position_refinement", {}).get("foot_contact_summary", {}).get("intervals")
    if not isinstance(intervals, list):
        intervals = []
    if not intervals:
        intervals = [
            {"patch_role": role, "frame_start": stats.get("frame_start"), "frame_end": stats.get("frame_end")}
            for role in ("toe", "heel")
            if isinstance((stats := contacts.get(role)), dict)
        ]
    intervals = _smooth_foot_role_intervals(intervals, anchor_start=anchor.start_frame, anchor_end=anchor.end_frame)
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        patch_role = str(interval.get("patch_role") or "")
        if patch_role not in {"toe", "heel", "sole"}:
            continue
        stats = _stats_for_patch_role(contacts, patch_role)
        if not isinstance(stats, dict):
            continue
        start = interval.get("frame_start")
        end = interval.get("frame_end")
        position = _position_for_patch_role(contacts, patch_role)
        if start is None or end is None or position is None:
            continue
        start_frame = max(int(anchor.start_frame), int(start))
        end_frame = min(int(anchor.end_frame), int(end))
        if end_frame <= start_frame:
            continue
        metadata = dict(anchor.metadata)
        metadata["patch_role"] = patch_role
        metadata["parent_anchor_id"] = anchor.anchor_id
        metadata["foot_subcontact"] = stats
        splits = list(metadata.get("foot_contact_splits") or [])
        splits.append(
            {
                "source": source,
                "parent_anchor_id": anchor.anchor_id,
                "patch_role": patch_role,
                "old_bounds": {"start_frame": anchor.start_frame, "end_frame": anchor.end_frame},
                "new_bounds": {"start_frame": start_frame, "end_frame": end_frame},
            }
        )
        metadata["foot_contact_splits"] = splits
        out.append(
            replace(
                anchor,
                anchor_id=f"{anchor.motion_id}_anchor_{anchor.body}_{patch_role}_{start_frame:06d}_{end_frame:06d}",
                start_frame=start_frame,
                end_frame=end_frame,
                world_position=[float(item) for item in position],
                position_source=f"{source}_{patch_role}",
                source=source,
                metadata=metadata,
            )
        )
    return _drop_short_near_duplicate_subanchors(out)


def _smooth_foot_role_intervals(
    intervals: list[dict[str, Any]],
    *,
    anchor_start: int,
    anchor_end: int,
    min_duration: int = 3,
    min_keep_duration: int = 2,
    max_same_role_gap: int = 3,
) -> list[dict[str, Any]]:
    cleaned = []
    for interval in intervals:
        role = interval.get("patch_role")
        start = interval.get("frame_start")
        end = interval.get("frame_end")
        if role is None or start is None or end is None:
            continue
        start_i = max(int(anchor_start), int(start))
        end_i = min(int(anchor_end), int(end))
        if end_i <= start_i:
            continue
        if end_i - start_i < min_keep_duration:
            continue
        cleaned.append({"patch_role": str(role), "frame_start": start_i, "frame_end": end_i})
    merged: list[dict[str, Any]] = []
    for interval in sorted(cleaned, key=lambda item: (int(item["frame_start"]), int(item["frame_end"]), str(item["patch_role"]))):
        if (
            merged
            and merged[-1]["patch_role"] == interval["patch_role"]
            and int(interval["frame_start"]) - int(merged[-1]["frame_end"]) <= max_same_role_gap
        ):
            merged[-1]["frame_end"] = max(int(merged[-1]["frame_end"]), int(interval["frame_end"]))
        else:
            merged.append(dict(interval))
    compressed = _compress_sole_dominant_intervals(merged, min_keep_duration=min_keep_duration)
    if compressed is not None:
        return _cover_foot_contact_episode(
            compressed,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
        )
    if len(merged) <= 1:
        return _cover_foot_contact_episode(
            merged,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
        )
    out: list[dict[str, Any]] = []
    for index, interval in enumerate(merged):
        duration = int(interval["frame_end"]) - int(interval["frame_start"])
        touches_anchor_boundary = int(interval["frame_start"]) <= int(anchor_start) or int(interval["frame_end"]) >= int(anchor_end)
        if duration >= min_duration or interval["patch_role"] == "sole" or touches_anchor_boundary:
            out.append(interval)
            continue
        previous_interval = out[-1] if out else None
        next_interval = merged[index + 1] if index + 1 < len(merged) else None
        target = previous_interval if previous_interval is not None else None
        if next_interval is not None and (
            target is None
            or int(next_interval["frame_start"]) - int(interval["frame_end"])
            < int(interval["frame_start"]) - int(target["frame_end"])
        ):
            target = next_interval
        if target is None:
            out.append(interval)
        elif target is previous_interval:
            target["frame_end"] = max(int(target["frame_end"]), int(interval["frame_end"]))
        else:
            target["frame_start"] = min(int(target["frame_start"]), int(interval["frame_start"]))
    return _cover_foot_contact_episode(
        out,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
    )


def _cover_foot_contact_episode(
    intervals: list[dict[str, Any]],
    *,
    anchor_start: int,
    anchor_end: int,
) -> list[dict[str, Any]]:
    """Assign every parent contact frame a heel/toe/sole role.

    The parent force/contact-mask episode is authoritative for whether contact
    exists. Raw Newton shape observations only classify that contact. Missing
    or rejected shape samples therefore inherit the nearest observed role
    instead of creating a false release in the middle or at a boundary.
    """

    frame_count = max(0, int(anchor_end) - int(anchor_start))
    if frame_count == 0 or not intervals:
        return []
    roles = np.full(frame_count, None, dtype=object)
    priority = {"heel": 1, "toe": 1, "sole": 2}
    role_priority = np.zeros(frame_count, dtype=np.int8)
    for interval in intervals:
        role = str(interval["patch_role"])
        start = max(int(anchor_start), int(interval["frame_start"]))
        end = min(int(anchor_end), int(interval["frame_end"]))
        if end <= start:
            continue
        begin = start - int(anchor_start)
        stop = end - int(anchor_start)
        candidate_priority = int(priority.get(role, 0))
        replace_mask = candidate_priority >= role_priority[begin:stop]
        replace_indices = np.flatnonzero(replace_mask) + begin
        roles[replace_indices] = role
        role_priority[replace_indices] = candidate_priority

    observed = np.flatnonzero(roles != None)  # noqa: E711
    if observed.size == 0:
        return []
    for frame in np.flatnonzero(roles == None):  # noqa: E711
        insertion = int(np.searchsorted(observed, frame))
        if insertion == 0:
            nearest = int(observed[0])
        elif insertion == observed.size:
            nearest = int(observed[-1])
        else:
            left = int(observed[insertion - 1])
            right = int(observed[insertion])
            nearest = left if frame - left <= right - frame else right
        roles[frame] = roles[nearest]

    output: list[dict[str, Any]] = []
    active_role = str(roles[0])
    active_start = int(anchor_start)
    for offset in range(1, frame_count):
        role = str(roles[offset])
        if role == active_role:
            continue
        output.append(
            {
                "patch_role": active_role,
                "frame_start": active_start,
                "frame_end": int(anchor_start) + offset,
            }
        )
        active_role = role
        active_start = int(anchor_start) + offset
    output.append(
        {
            "patch_role": active_role,
            "frame_start": active_start,
            "frame_end": int(anchor_end),
        }
    )
    return output


def _compress_sole_dominant_intervals(intervals: list[dict[str, Any]], *, min_keep_duration: int) -> list[dict[str, Any]] | None:
    sole_intervals = [item for item in intervals if item.get("patch_role") == "sole"]
    if not sole_intervals:
        return None
    sole_start = min(int(item["frame_start"]) for item in sole_intervals)
    sole_end = max(int(item["frame_end"]) for item in sole_intervals)
    out: list[dict[str, Any]] = []
    prefix = _dominant_nonsole_interval(intervals, end_before=sole_start)
    if prefix is not None and int(prefix["frame_end"]) - int(prefix["frame_start"]) >= min_keep_duration:
        out.append(prefix)
    out.append({"patch_role": "sole", "frame_start": sole_start, "frame_end": sole_end})
    suffix = _dominant_nonsole_interval(intervals, start_after=sole_end)
    if suffix is not None and int(suffix["frame_end"]) - int(suffix["frame_start"]) >= min_keep_duration:
        out.append(suffix)
    return out


def _dominant_nonsole_interval(
    intervals: list[dict[str, Any]],
    *,
    end_before: int | None = None,
    start_after: int | None = None,
) -> dict[str, Any] | None:
    selected = []
    for item in intervals:
        if item.get("patch_role") == "sole":
            continue
        start = int(item["frame_start"])
        end = int(item["frame_end"])
        if end_before is not None and end > end_before:
            continue
        if start_after is not None and start < start_after:
            continue
        selected.append(item)
    if not selected:
        return None
    durations: dict[str, int] = {}
    for item in selected:
        role = str(item["patch_role"])
        durations[role] = durations.get(role, 0) + int(item["frame_end"]) - int(item["frame_start"])
    role = max(durations.items(), key=lambda item: item[1])[0]
    role_items = [item for item in selected if item.get("patch_role") == role]
    return {
        "patch_role": role,
        "frame_start": min(int(item["frame_start"]) for item in role_items),
        "frame_end": max(int(item["frame_end"]) for item in role_items),
    }


def _stats_for_patch_role(contacts: dict[str, Any], patch_role: str) -> dict[str, Any] | None:
    if patch_role in {"toe", "heel"}:
        stats = contacts.get(patch_role)
        return stats if isinstance(stats, dict) else None
    toe = contacts.get("toe")
    heel = contacts.get("heel")
    if not isinstance(toe, dict) or not isinstance(heel, dict):
        return None
    return {
        "name": "sole",
        "toe": toe,
        "heel": heel,
        "sample_count": int(toe.get("sample_count", 0)) + int(heel.get("sample_count", 0)),
        "confidence": min(float(toe.get("confidence", 0.0)), float(heel.get("confidence", 0.0))),
    }


def _position_for_patch_role(contacts: dict[str, Any], patch_role: str) -> list[float] | None:
    if patch_role in {"toe", "heel"}:
        stats = contacts.get(patch_role)
        position = stats.get("world_position") if isinstance(stats, dict) else None
        return [float(item) for item in position] if position is not None else None
    toe = contacts.get("toe")
    heel = contacts.get("heel")
    toe_position = toe.get("world_position") if isinstance(toe, dict) else None
    heel_position = heel.get("world_position") if isinstance(heel, dict) else None
    if toe_position is None or heel_position is None:
        return None
    return [(float(toe_position[index]) + float(heel_position[index])) * 0.5 for index in range(3)]
