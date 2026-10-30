from __future__ import annotations

from typing import Any

import numpy as np


def map_points_between_surface_frames(
    points_w: np.ndarray,
    source_surface: dict[str, Any],
    target_surface: dict[str, Any],
) -> np.ndarray:
    """Rigidly map world points while preserving surface-frame coordinates."""

    points = np.asarray(points_w, dtype=np.float64)
    source_origin, source_basis = _surface_frame(source_surface)
    target_origin, target_basis = _surface_frame(target_surface)
    if np.array_equal(source_basis, target_basis):
        return points + (target_origin - source_origin)
    coordinates = (points - source_origin) @ source_basis.T
    return target_origin + coordinates @ target_basis


def _surface_frame(surface: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(surface["origin"], dtype=np.float64)
    vectors = [
        np.asarray(surface[name], dtype=np.float64)
        for name in ("tangent_u", "tangent_v", "normal")
    ]
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("surface origin must be one finite 3-vector")
    normalized = []
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
            raise ValueError("surface basis vectors must be finite and nonzero")
        normalized.append(vector / norm)
    basis = np.stack(normalized, axis=0)
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-5):
        raise ValueError("surface basis must be orthonormal")
    if np.linalg.det(basis) < 0.0:
        raise ValueError("surface basis must be right-handed")
    return origin, basis
