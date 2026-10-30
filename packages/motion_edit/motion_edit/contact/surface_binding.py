from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Iterable, Literal

from motion_edit.contact.schema import ContactAnchorRecord, ContactSurfaceRecord
from motion_edit.contact.surface_geometry import closest_point_on_polygon_uv, point_in_polygon_uv, surface_polygon_uv

BindingMode = Literal["reject", "clamp"]


def _dot3(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def _scale3(a: list[float], scale: float) -> list[float]:
    return [a[index] * scale for index in range(3)]


def _vector3(value: Iterable[float], *, name: str) -> list[float]:
    out = [float(item) for item in value]
    if len(out) != 3:
        raise ValueError(f"{name} must have length 3")
    return out


def _normalize3(value: Iterable[float], *, name: str) -> list[float]:
    out = _vector3(value, name=name)
    norm = sqrt(_dot3(out, out))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return [item / norm for item in out]


def _axis_bounds(bounds: dict | None, axis: str) -> tuple[float, float] | None:
    if bounds is None:
        return None
    value = bounds.get(axis)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"surface bounds[{axis!r}] must be a two-item list")
    return float(value[0]), float(value[1])


def _inside(value: float, bounds: tuple[float, float] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


def _clamp(value: float, bounds: tuple[float, float] | None) -> tuple[float, bool]:
    if bounds is None:
        return value, False
    clamped = min(max(value, bounds[0]), bounds[1])
    return clamped, clamped != value


def surface_compatible_with_body(body: str, surface: ContactSurfaceRecord) -> bool:
    _normalize3(surface.normal, name="surface.normal")
    return True


def _project(anchor: ContactAnchorRecord, surface: ContactSurfaceRecord) -> dict:
    if anchor.world_position is None:
        raise ValueError(f"{anchor.anchor_id}: world_position is required for surface binding")
    surface.validate()
    point = _vector3(anchor.world_position, name="anchor.world_position")
    origin = _vector3(surface.origin, name="surface.origin")
    normal = _normalize3(surface.normal, name="surface.normal")
    tangent_u = _normalize3(surface.tangent_u, name="surface.tangent_u")
    tangent_v = _normalize3(surface.tangent_v, name="surface.tangent_v")
    signed_distance = _dot3(_sub3(point, origin), normal)
    projected = _sub3(point, _scale3(normal, signed_distance))
    local = _sub3(projected, origin)
    u = _dot3(local, tangent_u)
    v = _dot3(local, tangent_v)
    return {
        "point": point,
        "origin": origin,
        "normal": normal,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "signed_distance": signed_distance,
        "projected": projected,
        "u": u,
        "v": v,
    }


def bind_anchor_to_surface(
    anchor: ContactAnchorRecord,
    surface: ContactSurfaceRecord,
    *,
    mode: BindingMode = "reject",
    max_distance: float = 0.05,
    project_world_position: bool = True,
) -> ContactAnchorRecord:
    if mode not in {"reject", "clamp"}:
        raise ValueError("mode must be 'reject' or 'clamp'")
    projection = _project(anchor, surface)
    if abs(float(projection["signed_distance"])) > float(max_distance):
        raise ValueError(
            f"{anchor.anchor_id}: surface {surface.surface_id} is farther than max_distance "
            f"({abs(float(projection['signed_distance'])):.6f} > {float(max_distance):.6f})"
        )
    raw_u = float(projection["u"])
    raw_v = float(projection["v"])
    polygon = surface_polygon_uv(
        surface.metadata,
        origin=projection["origin"],
        tangent_u=projection["tangent_u"],
        tangent_v=projection["tangent_v"],
    )
    if surface.surface_type == "mesh_face" and not polygon:
        raise ValueError(f"{surface.surface_id}: mesh_face surfaces require polygon_world metadata")
    if polygon:
        inside = point_in_polygon_uv((raw_u, raw_v), polygon)
        leave_message = "polygon"
    else:
        u_bounds = _axis_bounds(surface.bounds, "u")
        v_bounds = _axis_bounds(surface.bounds, "v")
        inside = _inside(raw_u, u_bounds) and _inside(raw_v, v_bounds)
        leave_message = "bounds"
    if not inside and mode == "reject":
        raise ValueError(f"{anchor.anchor_id}: projected point leaves surface {leave_message} for {surface.surface_id}")
    if polygon and not inside:
        u, v = closest_point_on_polygon_uv((raw_u, raw_v), polygon)
        clamped = True
    elif polygon:
        u, v = raw_u, raw_v
        clamped = False
    else:
        u_bounds = _axis_bounds(surface.bounds, "u")
        v_bounds = _axis_bounds(surface.bounds, "v")
        u, clamped_u = _clamp(raw_u, u_bounds)
        v, clamped_v = _clamp(raw_v, v_bounds)
        clamped = clamped_u or clamped_v
    projected = projection["projected"]
    bound_world = _add3(
        projection["origin"],
        _add3(_scale3(projection["tangent_u"], u), _scale3(projection["tangent_v"], v)),
    )
    metadata = dict(anchor.metadata)
    bindings = list(metadata.get("surface_bindings") or [])
    bindings.append(
        {
            "surface_id": surface.surface_id,
            "object_id": surface.object_id,
            "surface_type": surface.surface_type,
            "original_world_position": projection["point"],
            "projected_world_position": projected,
            "bound_world_position": bound_world,
            "signed_surface_distance": projection["signed_distance"],
            "surface_coordinates": {"u": u, "v": v},
            "raw_surface_coordinates": {"u": raw_u, "v": raw_v},
            "surface_binding_source": surface.source,
            "clamped": clamped,
            "polygon_surface_coordinates": [
                {"u": point[0], "v": point[1]}
                for point in polygon
            ] if polygon else None,
        }
    )
    metadata["surface_bindings"] = bindings
    metadata.pop("surface_binding_failed", None)
    metadata.pop("surface_binding_failure_reason", None)
    return replace(
        anchor,
        world_position=bound_world if project_world_position else anchor.world_position,
        object_id=surface.object_id,
        surface_id=surface.surface_id,
        surface_type=surface.surface_type,
        surface_normal=projection["normal"],
        surface_origin=projection["origin"],
        surface_tangent_u=projection["tangent_u"],
        surface_tangent_v=projection["tangent_v"],
        surface_bounds=surface.bounds,
        surface_coordinates={"u": u, "v": v},
        surface_binding_source=surface.source,
        metadata=metadata,
    )


def bind_anchors_to_surfaces(
    anchors: Iterable[ContactAnchorRecord],
    surfaces: Iterable[ContactSurfaceRecord],
    *,
    max_distance: float = 0.05,
    mode: BindingMode = "reject",
    project_world_position: bool = True,
) -> list[ContactAnchorRecord]:
    surface_list = list(surfaces)
    bound: list[ContactAnchorRecord] = []
    for anchor in anchors:
        candidates: list[tuple[float, int, ContactAnchorRecord]] = []
        failures: list[str] = []
        for surface in surface_list:
            if surface.motion_id != anchor.motion_id:
                continue
            if not surface_compatible_with_body(anchor.body, surface):
                failures.append(f"{surface.surface_id}: incompatible with body {anchor.body}")
                continue
            try:
                projection = _project(anchor, surface)
                polygon = surface_polygon_uv(
                    surface.metadata,
                    origin=projection["origin"],
                    tangent_u=projection["tangent_u"],
                    tangent_v=projection["tangent_v"],
                )
                if surface.surface_type == "mesh_face" and not polygon:
                    raise ValueError(f"{surface.surface_id}: mesh_face surfaces require polygon_world metadata")
                if polygon:
                    inside = point_in_polygon_uv((float(projection["u"]), float(projection["v"])), polygon)
                else:
                    u_bounds = _axis_bounds(surface.bounds, "u")
                    v_bounds = _axis_bounds(surface.bounds, "v")
                    inside = _inside(float(projection["u"]), u_bounds) and _inside(float(projection["v"]), v_bounds)
                candidate = bind_anchor_to_surface(
                    anchor,
                    surface,
                    mode=mode,
                    max_distance=max_distance,
                    project_world_position=project_world_position,
                )
                candidates.append((abs(float(projection["signed_distance"])), 0 if inside else 1, candidate))
            except ValueError as exc:
                failures.append(str(exc))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2].surface_id or ""))
            bound.append(candidates[0][2])
        else:
            metadata = dict(anchor.metadata)
            metadata["surface_binding_failed"] = True
            metadata["surface_binding_failure_reason"] = failures[0] if failures else "no compatible surface"
            bound.append(replace(anchor, metadata=metadata))
    return bound
