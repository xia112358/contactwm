from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_motion_npz(path: str | Path) -> dict[str, Any]:
    motion_path = Path(path).expanduser().resolve()
    with np.load(motion_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["_path"] = str(motion_path)
    return arrays


def motion_length(path: str | Path) -> int:
    data = load_motion_npz(path)
    for key in ("joint_pos", "qpos", "body_pos_w"):
        if key in data:
            return int(data[key].shape[0])
    raise KeyError(f"{path} has no joint_pos/qpos/body_pos_w")


def motion_fps(arrays: dict[str, Any], default: float = 50.0) -> float:
    if "fps" not in arrays:
        return default
    value = np.asarray(arrays["fps"]).reshape(-1)
    return float(value[0]) if value.size else default


def qpos_like(arrays: dict[str, Any]) -> np.ndarray:
    if "qpos" in arrays:
        return np.asarray(arrays["qpos"])
    if "joint_pos" in arrays:
        return np.asarray(arrays["joint_pos"])
    raise KeyError("motion npz has neither qpos nor joint_pos")


def subset_arrays(arrays: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = None
    for key in (
        "joint_pos",
        "qpos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_vel",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        if key in arrays:
            n = arrays[key].shape[0]
            break
    if n is None:
        raise KeyError("motion npz has no frame-major arrays")
    start = max(0, min(int(start), int(n)))
    end = max(start + 1, min(int(end), int(n)))
    for key, value in arrays.items():
        if key.startswith("_"):
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == n:
            out[key] = arr[start:end].copy()
        else:
            out[key] = arr.copy()
    out["source_start_frame"] = np.asarray(start, dtype=np.int64)
    out["source_end_frame"] = np.asarray(end, dtype=np.int64)
    from somaforge_core.support_evidence import reference_only_support
    reference_only_support(out, reason='cropped output requires a new binding to the native source recording', retain_force_targets=True)
    return out
