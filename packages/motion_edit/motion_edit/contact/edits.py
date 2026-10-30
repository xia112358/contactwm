from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from motion_edit.contact.schema import ContactAnchorEditRecord, ContactAnchorRecord
from motion_edit.contact.surface_geometry import closest_point_on_polygon_uv, point_in_polygon_uv


def _vector3(value: Iterable[float] | None, *, name: str) -> list[float] | None:
    if value is None:
        return None
    out = [float(item) for item in value]
    if len(out) != 3:
        raise ValueError(f"{name} must have length 3")
    return out


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] + b[index] for index in range(3)]


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def _scale3(a: list[float], value: float) -> list[float]:
    return [item * value for item in a]


def _dot3(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _vector2(value: Iterable[float] | None, *, name: str) -> list[float] | None:
    if value is None:
        return None
    out = [float(item) for item in value]
    if len(out) != 2:
        raise ValueError(f"{name} must have length 2")
    return out


def _surface_basis(anchor: ContactAnchorRecord) -> tuple[list[float], list[float], list[float], list[float]]:
    if anchor.world_position is None:
        raise ValueError("anchor has no world_position; cannot safely drag on surface")
    if anchor.surface_normal is None or anchor.surface_tangent_u is None or anchor.surface_tangent_v is None:
        raise ValueError("anchor has no surface binding; cannot safely drag on surface")
    origin = anchor.surface_origin or anchor.world_position
    return (
        _vector3(anchor.world_position, name="world_position") or [0.0, 0.0, 0.0],
        _vector3(anchor.surface_normal, name="surface_normal") or [0.0, 0.0, 1.0],
        _vector3(anchor.surface_tangent_u, name="surface_tangent_u") or [1.0, 0.0, 0.0],
        _vector3(anchor.surface_tangent_v, name="surface_tangent_v") or [0.0, 1.0, 0.0],
    ), _vector3(origin, name="surface_origin") or [0.0, 0.0, 0.0]


def _surface_coordinates(point: list[float], origin: list[float], tangent_u: list[float], tangent_v: list[float]) -> dict[str, float]:
    local = _sub3(point, origin)
    return {"u": _dot3(local, tangent_u), "v": _dot3(local, tangent_v)}


def _bounds_for_axis(bounds: dict | None, axis: str) -> tuple[float, float] | None:
    if not bounds:
        return None
    raw = bounds.get(axis)
    if raw is None:
        raw = bounds.get(f"{axis}_bounds")
    if raw is None:
        return None
    values = [float(item) for item in raw]
    if len(values) != 2:
        raise ValueError(f"surface_bounds[{axis!r}] must have length 2")
    return min(values), max(values)


def _clamp(value: float, bounds: tuple[float, float] | None) -> tuple[float, bool]:
    if bounds is None:
        return value, False
    lo, hi = bounds
    clamped = min(max(value, lo), hi)
    return clamped, clamped != value


def _latest_surface_polygon(anchor: ContactAnchorRecord) -> list[tuple[float, float]]:
    bindings = anchor.metadata.get("surface_bindings")
    if not isinstance(bindings, list) or not bindings:
        return []
    raw = bindings[-1].get("polygon_surface_coordinates") if isinstance(bindings[-1], dict) else None
    if not isinstance(raw, list) or len(raw) < 3:
        return []
    polygon: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, dict) or "u" not in item or "v" not in item:
            return []
        polygon.append((float(item["u"]), float(item["v"])))
    return polygon


def move_contact_anchor_free(
    anchor: ContactAnchorRecord,
    *,
    delta_world: Iterable[float] | None = None,
    new_world_position: Iterable[float] | None = None,
    position_source: str = "manual",
) -> ContactAnchorRecord:
    if delta_world is None and new_world_position is None:
        raise ValueError("move_contact_anchor_free requires delta_world or new_world_position")
    old_position = anchor.world_position
    delta = _vector3(delta_world, name="delta_world")
    new_position = _vector3(new_world_position, name="new_world_position")
    if new_position is None:
        if old_position is None:
            raise ValueError("delta_world requires anchor.world_position")
        new_position = _add3([float(item) for item in old_position], delta or [0.0, 0.0, 0.0])
    elif delta is None and old_position is not None:
        delta = _sub3(new_position, [float(item) for item in old_position])
    metadata = dict(anchor.metadata)
    edits = list(metadata.get("contact_anchor_edits") or [])
    edits.append(
        {
            "kind": "move_contact_anchor",
            "old_world_position": old_position,
            "new_world_position": new_position,
            "delta_world": delta,
            "source": position_source,
        }
    )
    metadata["contact_anchor_edits"] = edits
    return replace(anchor, world_position=new_position, position_source=position_source, metadata=metadata)


def move_contact_anchor(
    anchor: ContactAnchorRecord,
    *,
    delta_world: Iterable[float] | None = None,
    new_world_position: Iterable[float] | None = None,
    position_source: str = "manual",
) -> ContactAnchorRecord:
    return move_contact_anchor_free(
        anchor,
        delta_world=delta_world,
        new_world_position=new_world_position,
        position_source=position_source,
    )


def move_contact_anchor_on_surface(
    anchor: ContactAnchorRecord,
    *,
    tangent_delta: Iterable[float] | None = None,
    requested_world_delta: Iterable[float] | None = None,
    new_surface_coordinates: dict[str, float] | None = None,
    mode: str = "reject",
    source: str = "manual",
    edit_id: str | None = None,
) -> tuple[ContactAnchorRecord, ContactAnchorEditRecord]:
    if mode not in {"reject", "clamp"}:
        raise ValueError("mode must be 'reject' or 'clamp'")
    basis, origin = _surface_basis(anchor)
    old_position, normal, tangent_u, tangent_v = basis
    before = anchor.surface_coordinates or _surface_coordinates(old_position, origin, tangent_u, tangent_v)
    start_u = float(before.get("u", 0.0))
    start_v = float(before.get("v", 0.0))
    requested = _vector3(requested_world_delta, name="requested_world_delta")
    tangent = _vector2(tangent_delta, name="tangent_delta")
    if new_surface_coordinates is not None:
        target_u = float(new_surface_coordinates["u"])
        target_v = float(new_surface_coordinates["v"])
        tangent = [target_u - start_u, target_v - start_v]
    elif tangent is None and requested is not None:
        tangent = [_dot3(requested, tangent_u), _dot3(requested, tangent_v)]
        target_u = start_u + tangent[0]
        target_v = start_v + tangent[1]
    elif tangent is not None:
        target_u = start_u + tangent[0]
        target_v = start_v + tangent[1]
    else:
        raise ValueError("move_contact_anchor_on_surface requires tangent_delta, requested_world_delta, or new_surface_coordinates")

    raw_u, raw_v = target_u, target_v
    polygon = _latest_surface_polygon(anchor)
    if polygon:
        inside = point_in_polygon_uv((target_u, target_v), polygon)
        if not inside and mode == "reject":
            raise ValueError("requested contact anchor move leaves the original surface polygon")
        if inside:
            clamped = False
        else:
            target_u, target_v = closest_point_on_polygon_uv((target_u, target_v), polygon)
            clamped = True
    else:
        u_bounds = _bounds_for_axis(anchor.surface_bounds, "u")
        v_bounds = _bounds_for_axis(anchor.surface_bounds, "v")
        target_u, clamped_u = _clamp(target_u, u_bounds)
        target_v, clamped_v = _clamp(target_v, v_bounds)
        clamped = clamped_u or clamped_v
        if clamped and mode == "reject":
            raise ValueError("requested contact anchor move leaves the original surface bounds")

    after = {"u": target_u, "v": target_v}
    delta_u = target_u - start_u
    delta_v = target_v - start_v
    effective_delta = _add3(_scale3(tangent_u, delta_u), _scale3(tangent_v, delta_v))
    new_position = _add3(old_position, effective_delta)
    edit_tangent = [delta_u, delta_v]
    metadata = dict(anchor.metadata)
    edits = list(metadata.get("contact_anchor_edits") or [])
    edit_data = {
        "kind": "move_contact_anchor",
        "old_world_position": old_position,
        "new_world_position": new_position,
        "requested_delta_world": requested,
        "delta_world": effective_delta,
        "tangent_delta": edit_tangent,
        "surface_id": anchor.surface_id,
        "surface_normal": normal,
        "surface_coordinates_before": before,
        "surface_coordinates_after": after,
        "constraint_mode": mode,
        "clamped": clamped,
        "source": source,
    }
    if clamped:
        edit_data["requested_surface_coordinates"] = {"u": raw_u, "v": raw_v}
    edits.append(edit_data)
    metadata["contact_anchor_edits"] = edits
    moved = replace(
        anchor,
        world_position=new_position,
        surface_coordinates=after,
        position_source=source,
        metadata=metadata,
    )
    edit = ContactAnchorEditRecord(
        edit_id=edit_id or f"{anchor.anchor_id}_move_contact_anchor",
        motion_id=anchor.motion_id,
        anchor_id=anchor.anchor_id,
        body=anchor.body,
        old_world_position=old_position,
        new_world_position=new_position,
        requested_delta_world=requested,
        delta_world=effective_delta,
        tangent_delta=edit_tangent,
        affected_frames=[anchor.start_frame, anchor.end_frame],
        surface_id=anchor.surface_id,
        surface_normal=normal,
        surface_coordinates_before=before,
        surface_coordinates_after=after,
        constraint_mode=mode,
        clamped=clamped,
        source=source,
        metadata={"requested_surface_coordinates": {"u": raw_u, "v": raw_v}} if clamped else {},
    )
    edit.validate()
    return moved, edit


def make_anchor_move_edit(
    anchor: ContactAnchorRecord,
    *,
    delta_world: Iterable[float] | None = None,
    new_world_position: Iterable[float] | None = None,
    affected_frames: Iterable[int] | None = None,
    source: str = "manual",
    edit_id: str | None = None,
) -> ContactAnchorEditRecord:
    moved = move_contact_anchor_free(
        anchor,
        delta_world=delta_world,
        new_world_position=new_world_position,
        position_source=source,
    )
    old_position = anchor.world_position
    new_position = moved.world_position
    delta = _vector3(delta_world, name="delta_world")
    if delta is None and old_position is not None and new_position is not None:
        delta = _sub3(new_position, [float(item) for item in old_position])
    record = ContactAnchorEditRecord(
        edit_id=edit_id or f"{anchor.anchor_id}_move_contact_anchor",
        motion_id=anchor.motion_id,
        anchor_id=anchor.anchor_id,
        body=anchor.body,
        old_world_position=old_position,
        new_world_position=new_position,
        delta_world=delta,
        affected_frames=[int(frame) for frame in affected_frames] if affected_frames is not None else [anchor.start_frame, anchor.end_frame],
        source=source,
    )
    record.validate()
    return record
