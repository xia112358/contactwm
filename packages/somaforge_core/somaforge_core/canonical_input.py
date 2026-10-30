from __future__ import annotations

from pathlib import Path

import numpy as np
from somaforge_core import G1_29DOF_JOINT_ORDER, angular_velocity_wxyz
from somaforge_core.robot_assets import decode_robot_asset_json


INPUT_FORMATS = ("omniretarget_qpos", "holosoma_joint_pos")


def load_canonical_qpos_input(
    path: str | Path,
    *,
    input_format: str,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an input motion as Omni-style ``[quat, pos, joints]`` qpos/qvel."""

    source = Path(path).expanduser().resolve()
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"input_format must be one of {INPUT_FORMATS}, got {input_format!r}")
    if not np.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError("output_fps must be finite and positive")

    with np.load(source, allow_pickle=False) as data:
        input_fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
        if input_format == "omniretarget_qpos":
            if "qpos" not in data:
                raise KeyError(f"{source} is missing qpos")
            qpos = np.asarray(data["qpos"], dtype=np.float32)
        else:
            decode_robot_asset_json(data.get("robot_asset_json"), context=f"motion {source}")
            qpos = _holosoma_joint_pos_to_qpos(data, source)

    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{source} qpos must have shape [T,36], got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"{source} must contain at least two frames")
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"{source} qpos contains NaN or Inf")

    qpos = resample_qpos(qpos, input_fps=input_fps, output_fps=output_fps)
    dt = 1.0 / float(output_fps)
    root_linear = np.gradient(qpos[:, 4:7], dt, axis=0).astype(np.float32)
    root_angular = angular_velocity_wxyz(qpos[:, :4], dt)
    joint_velocity = np.gradient(qpos[:, 7:], dt, axis=0).astype(np.float32)
    qvel = np.concatenate((root_linear, root_angular, joint_velocity), axis=1)
    return qpos, qvel


def resample_qpos(qpos: np.ndarray, *, input_fps: float, output_fps: float) -> np.ndarray:
    value = np.asarray(qpos, dtype=np.float32)
    if not np.isfinite(input_fps) or input_fps <= 0.0:
        raise ValueError("input_fps must be finite and positive")
    if np.isclose(input_fps, output_fps, rtol=0.0, atol=1.0e-9):
        return value.copy()

    duration = (value.shape[0] - 1) / float(input_fps)
    frame_count = int(round(duration * float(output_fps))) + 1
    times = np.linspace(0.0, duration, frame_count, dtype=np.float64)
    phase = times * float(input_fps)
    i0 = np.floor(phase).astype(np.int64)
    i1 = np.minimum(i0 + 1, value.shape[0] - 1)
    blend = phase - i0
    out = value[i0] * (1.0 - blend[:, None]) + value[i1] * blend[:, None]
    out[:, :4] = _slerp_wxyz(value[i0, :4], value[i1, :4], blend)
    return out.astype(np.float32)


def _holosoma_joint_pos_to_qpos(data: np.lib.npyio.NpzFile, source: Path) -> np.ndarray:
    if "joint_pos" not in data:
        raise KeyError(f"{source} is missing joint_pos")
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"{source} joint_pos must have shape [T,36], got {joint_pos.shape}")
    if "joint_names" not in data:
        raise KeyError(f"{source} is missing joint_names")
    joint_names = tuple(str(name) for name in np.asarray(data["joint_names"]).reshape(-1).tolist())
    if len(joint_names) != len(G1_29DOF_JOINT_ORDER) or len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{source} joint_names must contain 29 unique names")
    missing = [name for name in G1_29DOF_JOINT_ORDER if name not in joint_names]
    if missing:
        raise ValueError(f"{source} is missing canonical joints: {missing}")
    ordered_joints = joint_pos[:, 7:][:, [joint_names.index(name) for name in G1_29DOF_JOINT_ORDER]]
    return np.concatenate((joint_pos[:, 3:7], joint_pos[:, :3], ordered_joints), axis=1)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1.0e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1.0e-12)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    t = blend[:, None]
    close = sin_theta < 1.0e-8
    out = np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1.0e-12) * q0
    out += np.sin(t * theta) / np.maximum(sin_theta, 1.0e-12) * q1
    out = np.where(close, (1.0 - t) * q0 + t * q1, out)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1.0e-12)
