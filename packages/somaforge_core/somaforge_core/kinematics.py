from __future__ import annotations

import numpy as np

POSE_FINITE_DIFFERENCE = "pose_finite_difference"


def validate_root_body_consistency(
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    *,
    root_body_index: int = 0,
    atol: float = 1.0e-5,
) -> None:
    """Validate Holosoma root fields against the corresponding FK body pose."""
    joint_pos = np.asarray(joint_pos)
    body_pos_w = np.asarray(body_pos_w)
    body_quat_w = np.asarray(body_quat_w)
    if joint_pos.ndim != 2 or joint_pos.shape[1] < 7:
        raise ValueError(f"joint_pos must have shape [T,7+J], got {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[0] != joint_pos.shape[0] or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T,B,3], got {body_pos_w.shape}")
    if body_quat_w.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError(f"body_quat_w must have shape [T,B,4], got {body_quat_w.shape}")
    if not 0 <= root_body_index < body_pos_w.shape[1]:
        raise ValueError(f"root_body_index is out of range: {root_body_index}")

    root_pos_error = np.max(np.abs(joint_pos[:, :3] - body_pos_w[:, root_body_index]))
    if root_pos_error > atol:
        raise ValueError(f"root body position is inconsistent with joint_pos: max error={root_pos_error}")

    root_quat = joint_pos[:, 3:7]
    body_quat = body_quat_w[:, root_body_index]
    root_quat = root_quat / np.maximum(np.linalg.norm(root_quat, axis=-1, keepdims=True), 1.0e-12)
    body_quat = body_quat / np.maximum(np.linalg.norm(body_quat, axis=-1, keepdims=True), 1.0e-12)
    quat_error = np.minimum(
        np.max(np.abs(root_quat - body_quat), axis=-1),
        np.max(np.abs(root_quat + body_quat), axis=-1),
    ).max()
    if quat_error > atol:
        raise ValueError(f"root body quaternion is inconsistent with joint_pos: max error={quat_error}")


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def angular_velocity_wxyz(quat: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate world-frame WXYZ quaternions along their first axis."""
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim < 2 or quat.shape[0] < 2 or quat.shape[-1] != 4:
        raise ValueError(f"quat must have shape [T,...,4] with T >= 2, got {quat.shape}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")

    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-12)
    previous = np.concatenate((quat[:1], quat[:-1]), axis=0)
    following = np.concatenate((quat[1:], quat[-1:]), axis=0)
    conjugate = previous.copy()
    conjugate[..., 1:] *= -1.0
    relative = _quat_mul_wxyz(following, conjugate)
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    relative /= np.maximum(np.linalg.norm(relative, axis=-1, keepdims=True), 1.0e-12)

    angle = 2.0 * np.arccos(np.clip(relative[..., 0], -1.0, 1.0))
    sin_half = np.sqrt(np.maximum(1.0 - relative[..., 0] ** 2, 0.0))
    axis = np.divide(
        relative[..., 1:],
        sin_half[..., None],
        out=np.zeros_like(relative[..., 1:]),
        where=sin_half[..., None] > 1.0e-8,
    )
    scale = np.full(angle.shape, 2.0 * dt, dtype=np.float64)
    scale[0] = dt
    scale[-1] = dt
    return (axis * (angle / scale)[..., None]).astype(np.float32)


def body_velocities_from_pose(
    body_pos_w: np.ndarray, body_quat_w: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Derive world-frame link velocities from a sampled pose sequence."""
    body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
    body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape [T,B,3], got {body_pos_w.shape}")
    if body_quat_w.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError(f"body_quat_w must match body_pos_w with a quaternion tail, got {body_quat_w.shape}")
    if body_pos_w.shape[0] < 2:
        raise ValueError("body pose sequence must contain at least two frames")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not np.isfinite(body_pos_w).all() or not np.isfinite(body_quat_w).all():
        raise ValueError("body pose sequence contains non-finite values")

    quat_norm = np.linalg.norm(body_quat_w, axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-4):
        raise ValueError(f"body quaternion norm error is too large: {np.max(np.abs(quat_norm - 1.0))}")

    dt = 1.0 / float(fps)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = angular_velocity_wxyz(body_quat_w, dt)
    return body_lin_vel_w, body_ang_vel_w


def validate_pose_velocity_consistency(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    fps: float,
    *,
    atol: float = 1.0e-5,
) -> None:
    expected_lin, expected_ang = body_velocities_from_pose(body_pos_w, body_quat_w, fps)
    if not np.allclose(body_lin_vel_w, expected_lin, atol=atol, rtol=0.0):
        error = float(np.max(np.abs(np.asarray(body_lin_vel_w) - expected_lin)))
        raise ValueError(f"body linear velocity is inconsistent with body pose: max error={error}")
    if not np.allclose(body_ang_vel_w, expected_ang, atol=atol, rtol=0.0):
        error = float(np.max(np.abs(np.asarray(body_ang_vel_w) - expected_ang)))
        raise ValueError(f"body angular velocity is inconsistent with body pose: max error={error}")
