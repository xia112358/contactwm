from __future__ import annotations

import math
from typing import Any


def dot3(a: list[float], b: list[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def sub3(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(3)]


def surface_polygon_uv(surface_metadata: dict[str, Any], *, origin: list[float], tangent_u: list[float], tangent_v: list[float]) -> list[tuple[float, float]]:
    polygon = surface_metadata.get("polygon_world")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return []
    out: list[tuple[float, float]] = []
    for point in polygon:
        if not isinstance(point, list) or len(point) != 3:
            return []
        local = sub3([float(item) for item in point], origin)
        out.append((dot3(local, tangent_u), dot3(local, tangent_v)))
    return out


def point_in_polygon_uv(point: tuple[float, float], polygon: list[tuple[float, float]], *, eps: float = 1e-9) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        if _point_segment_distance(point, a, b) <= eps:
            return True
    inside = False
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        if (a[1] > y) != (b[1] > y):
            x_intersection = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < x_intersection:
                inside = not inside
    return inside


def closest_point_on_polygon_uv(point: tuple[float, float], polygon: list[tuple[float, float]]) -> tuple[float, float]:
    if len(polygon) < 3:
        raise ValueError("polygon must have at least three points")
    if point_in_polygon_uv(point, polygon):
        return point
    best = None
    best_distance = math.inf
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        candidate = _closest_point_on_segment(point, a, b)
        distance = _point_distance(point, candidate)
        if distance < best_distance:
            best = candidate
            best_distance = distance
    if best is None:
        raise ValueError("could not clamp point to polygon")
    return best


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return _point_distance(point, _closest_point_on_segment(point, a, b))


def _closest_point_on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    ab = (b[0] - a[0], b[1] - a[1])
    ap = (point[0] - a[0], point[1] - a[1])
    denom = ab[0] * ab[0] + ab[1] * ab[1]
    if denom == 0.0:
        return a
    t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / denom))
    return (a[0] + t * ab[0], a[1] + t * ab[1])
