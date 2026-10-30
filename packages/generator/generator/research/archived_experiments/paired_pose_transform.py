"""Preserved experiment helpers for regression tests; no training CLI."""
import numpy as np
from generator.fullbody_dataset import _quaternion_matrix_wxyz, _matrix_quaternion_wxyz

def yaw_basis(q):
    r = _quaternion_matrix_wxyz(q)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    (c, s) = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def transform_q(q, origin, basis, inverse=False):
    result = q.copy()
    if inverse:
        result[:3] = basis.T @ (q[:3] - origin)
        result[3:7] = _matrix_quaternion_wxyz(basis.T @ _quaternion_matrix_wxyz(q[3:7]))
    else:
        result[:3] = origin + basis @ q[:3]
        result[3:7] = _matrix_quaternion_wxyz(basis @ _quaternion_matrix_wxyz(q[3:7]))
    return result.astype(np.float32)
