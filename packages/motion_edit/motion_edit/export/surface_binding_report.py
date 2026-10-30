from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.schema import ContactAnchorRecord, ContactSurfaceRecord

BODY_COLORS: dict[str, list[int]] = {
    "lf": [60, 140, 255],
    "left_foot": [60, 140, 255],
    "rf": [255, 120, 65],
    "right_foot": [255, 120, 65],
    "lh": [80, 210, 130],
    "left_hand": [80, 210, 130],
    "rh": [210, 110, 255],
    "right_hand": [210, 110, 255],
    "lk": [255, 205, 70],
    "left_knee": [255, 205, 70],
    "rk": [90, 220, 220],
    "right_knee": [90, 220, 220],
}


def _latest_binding(anchor: ContactAnchorRecord) -> dict[str, Any] | None:
    bindings = anchor.metadata.get("surface_bindings")
    if isinstance(bindings, list) and bindings:
        latest = bindings[-1]
        if isinstance(latest, dict):
            return latest
    return None


def _bounds_for_axis(bounds: dict | None, axis: str) -> tuple[float, float] | None:
    if not bounds:
        return None
    value = bounds.get(axis)
    if not isinstance(value, list) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _outside_bounds(coords: dict | None, bounds: dict | None) -> bool:
    if not isinstance(coords, dict):
        return False
    for axis in ("u", "v"):
        axis_bounds = _bounds_for_axis(bounds, axis)
        if axis_bounds is None or axis not in coords:
            continue
        value = float(coords[axis])
        if value < axis_bounds[0] or value > axis_bounds[1]:
            return True
    return False


def _anchor_warnings(anchor: ContactAnchorRecord, binding: dict[str, Any] | None) -> list[str]:
    warnings: list[str] = []
    candidate_class = _binding_candidate_class(anchor)
    if candidate_class in {"side_filtered", "edge_candidate", "raw_missing", "outside_known_surfaces"}:
        warnings.append(f"binding candidate class: {candidate_class}")
    if anchor.metadata.get("surface_binding_failed"):
        warnings.append(str(anchor.metadata.get("surface_binding_failure_reason") or "surface binding failed"))
    if anchor.surface_id and binding is None:
        warnings.append("binding metadata missing")
    if binding is not None and _outside_bounds(binding.get("raw_surface_coordinates"), anchor.surface_bounds):
        warnings.append("raw surface coordinates outside bounds")
    if anchor.metadata.get("body_surface_compatibility_warning"):
        warnings.append(str(anchor.metadata["body_surface_compatibility_warning"]))
    return warnings


def _binding_candidate_class(anchor: ContactAnchorRecord) -> str | None:
    refinement = anchor.metadata.get("raw_contact_position_refinement")
    if isinstance(refinement, dict):
        value = refinement.get("binding_candidate_class")
        if value:
            return str(value)
    return None


def _anchor_status(anchor: ContactAnchorRecord) -> str:
    binding = _latest_binding(anchor)
    candidate_class = _binding_candidate_class(anchor)
    if candidate_class == "edge_candidate":
        return "suspicious"
    if anchor.metadata.get("surface_binding_failed"):
        return "failed"
    if not anchor.surface_id:
        return "unbound"
    if binding is not None and binding.get("clamped"):
        return "clamped"
    if anchor.metadata.get("surface_editor_status") == "edited":
        return "edited"
    if anchor.metadata.get("surface_editor_status") == "selected":
        return "selected"
    warnings = _anchor_warnings(anchor, binding)
    if warnings:
        return "suspicious"
    return "bound"


def _anchor_report(anchor: ContactAnchorRecord) -> dict[str, Any]:
    binding = _latest_binding(anchor)
    status = _anchor_status(anchor)
    patch_role = anchor.metadata.get("patch_role")
    return {
        "anchor_id": anchor.anchor_id,
        "body": anchor.body,
        "patch_role": patch_role,
        "start_frame": anchor.start_frame,
        "end_frame": anchor.end_frame,
        "world_position": anchor.world_position,
        "surface_id": anchor.surface_id,
        "object_id": anchor.object_id,
        "surface_type": anchor.surface_type,
        "surface_coordinates": anchor.surface_coordinates,
        "surface_normal": anchor.surface_normal,
        "surface_bounds": anchor.surface_bounds,
        "binding": binding,
        "status": status,
        "binding_candidate_class": _binding_candidate_class(anchor),
        "color": _body_color(anchor.body),
        "warnings": _anchor_warnings(anchor, binding),
        "raw_contact_position_refinement": anchor.metadata.get("raw_contact_position_refinement"),
        "binding_granularity": "anchor_point",
        "binding_note": "surface binding is an anchor-level association, not a full foot sole contact model",
    }


def _surface_dict(surface: ContactSurfaceRecord) -> dict[str, Any]:
    return surface.to_dict()


def _body_color(body: str) -> list[int]:
    normalized = body.lower()
    for key, color in BODY_COLORS.items():
        if key in normalized or normalized in key:
            return color
    return [80, 180, 255]


def _summary(graph: ContactGraph, surfaces: list[ContactSurfaceRecord]) -> dict[str, int]:
    statuses = [_anchor_status(anchor) for anchor in graph.anchors]
    return {
        "anchor_count": len(graph.anchors),
        "bound_count": statuses.count("bound") + statuses.count("clamped") + statuses.count("suspicious"),
        "unbound_count": statuses.count("unbound"),
        "failed_count": statuses.count("failed"),
        "clamped_count": statuses.count("clamped"),
        "surface_count": len(surfaces),
        "low_confidence_count": statuses.count("suspicious"),
    }


def export_surface_binding_report(
    output_path: str | Path,
    *,
    graph: ContactGraph,
    surfaces: Iterable[ContactSurfaceRecord] | None = None,
) -> Path:
    surface_list = list(surfaces or [])
    data = {
        "schema_version": 1,
        "motion_id": graph.motion_id,
        "binding_granularity": "anchor_point",
        "binding_note": "surface binding is an anchor-level association, not a full foot sole contact model",
        "summary": _summary(graph, surface_list),
        "surfaces": [_surface_dict(surface) for surface in surface_list],
        "anchors": [_anchor_report(anchor) for anchor in graph.anchors],
    }
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def _scale3(a: list[float], scale: float) -> list[float]:
    return [a[index] * scale for index in range(3)]


def _surface_quad_corners(surface: ContactSurfaceRecord) -> list[list[float]] | None:
    polygon = surface.metadata.get("polygon_world")
    if isinstance(polygon, list) and len(polygon) >= 3:
        return [[float(value) for value in point] for point in polygon]
    u_bounds = _bounds_for_axis(surface.bounds, "u")
    v_bounds = _bounds_for_axis(surface.bounds, "v")
    if u_bounds is None or v_bounds is None:
        return None
    corners = []
    for u, v in [
        (u_bounds[0], v_bounds[0]),
        (u_bounds[1], v_bounds[0]),
        (u_bounds[1], v_bounds[1]),
        (u_bounds[0], v_bounds[1]),
    ]:
        corners.append(_add3(surface.origin, _add3(_scale3(surface.tangent_u, u), _scale3(surface.tangent_v, v))))
    return corners


def _surface_overlay_object(surface: ContactSurfaceRecord) -> dict[str, Any]:
    item = {
        "type": "surface_quad",
        "surface_id": surface.surface_id,
        "object_id": surface.object_id,
        "origin": surface.origin,
        "normal": surface.normal,
        "tangent_u": surface.tangent_u,
        "tangent_v": surface.tangent_v,
        "bounds": surface.bounds,
        "status": "surface",
    }
    corners = _surface_quad_corners(surface)
    if corners is not None:
        item["corners"] = corners
        item["surface_shape"] = "polygon" if surface.metadata.get("polygon_world") else "quad"
    return item


def _normal_axis(surface: ContactSurfaceRecord, *, length: float = 0.1) -> dict[str, Any]:
    return {
        "type": "normal_axis",
        "surface_id": surface.surface_id,
        "from": surface.origin,
        "to": _add3(surface.origin, _scale3(surface.normal, length)),
    }


def _anchor_overlay_objects(anchor: ContactAnchorRecord) -> list[dict[str, Any]]:
    status = _anchor_status(anchor)
    binding = _latest_binding(anchor)
    position = anchor.world_position
    patch_role = anchor.metadata.get("patch_role")
    objects = [
        {
            "type": "anchor_point",
            "anchor_id": anchor.anchor_id,
            "body": anchor.body,
            "patch_role": patch_role,
            "start_frame": anchor.start_frame,
            "end_frame": anchor.end_frame,
            "position": position,
            "surface_id": anchor.surface_id,
            "status": status,
            "binding_candidate_class": _binding_candidate_class(anchor),
            "color": _body_color(anchor.body),
        }
    ]
    if binding is not None and binding.get("original_world_position") is not None and binding.get("bound_world_position") is not None:
        objects.append(
            {
                "type": "projection_line",
                "anchor_id": anchor.anchor_id,
                "from": binding["original_world_position"],
                "to": binding["bound_world_position"],
                "surface_id": anchor.surface_id,
            }
        )
    return objects


def export_surface_binding_overlay(
    output_path: str | Path,
    *,
    graph: ContactGraph,
    surfaces: Iterable[ContactSurfaceRecord] | None = None,
) -> Path:
    surface_list = list(surfaces or [])
    objects: list[dict[str, Any]] = []
    for surface in surface_list:
        objects.append(_surface_overlay_object(surface))
        objects.append(_normal_axis(surface))
    for anchor in graph.anchors:
        objects.extend(_anchor_overlay_objects(anchor))
    data = {
        "schema_version": 1,
        "motion_id": graph.motion_id,
        "objects": objects,
    }
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
