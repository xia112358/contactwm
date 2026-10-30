from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Iterable

from motion_edit.contact.schema import ContactAnchorRecord, ContactSurfaceType


def _vector3(value: Iterable[float], *, name: str) -> list[float]:
    out = [float(item) for item in value]
    if len(out) != 3:
        raise ValueError(f"{name} must have length 3")
    return out


def _dot3(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross3(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _normalize3(value: list[float], *, name: str) -> list[float]:
    norm = sqrt(_dot3(value, value))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return [item / norm for item in value]


def _default_tangents(normal: list[float]) -> tuple[list[float], list[float]]:
    reference = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    tangent_u = _normalize3(_cross3(normal, reference), name="surface_tangent_u")
    tangent_v = _normalize3(_cross3(normal, tangent_u), name="surface_tangent_v")
    return tangent_u, tangent_v


def bind_anchor_to_plane(
    anchor: ContactAnchorRecord,
    *,
    surface_id: str,
    normal: Iterable[float],
    origin: Iterable[float] | None = None,
    tangent_u: Iterable[float] | None = None,
    tangent_v: Iterable[float] | None = None,
    bounds: dict | None = None,
    surface_type: ContactSurfaceType = "plane",
    source: str = "manual",
) -> ContactAnchorRecord:
    if anchor.world_position is None:
        raise ValueError("anchor.world_position is required to bind a contact surface")
    surface_normal = _normalize3(_vector3(normal, name="normal"), name="normal")
    if tangent_u is None or tangent_v is None:
        surface_tangent_u, surface_tangent_v = _default_tangents(surface_normal)
    else:
        surface_tangent_u = _normalize3(_vector3(tangent_u, name="tangent_u"), name="tangent_u")
        surface_tangent_v = _normalize3(_vector3(tangent_v, name="tangent_v"), name="tangent_v")
    surface_origin = _vector3(origin, name="origin") if origin is not None else [float(item) for item in anchor.world_position]
    local = [float(anchor.world_position[index]) - surface_origin[index] for index in range(3)]
    coordinates = {
        "u": _dot3(local, surface_tangent_u),
        "v": _dot3(local, surface_tangent_v),
    }
    return replace(
        anchor,
        surface_id=surface_id,
        surface_type=surface_type,
        surface_normal=surface_normal,
        surface_origin=surface_origin,
        surface_tangent_u=surface_tangent_u,
        surface_tangent_v=surface_tangent_v,
        surface_bounds=bounds,
        surface_coordinates=coordinates,
        surface_binding_source=source,
    )
