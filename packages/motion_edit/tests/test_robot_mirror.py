from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from somaforge_core.robot_assets import encode_robot_asset_json
from somaforge_core.motion_schema import G1_29DOF_JOINT_ORDER

from motion_edit.robot_mirror import (
    _canonicalize_quaternion_hemisphere_wxyz,
    mirror_motion_npz,
)


def _write_motion(path: Path) -> None:
    joints = list(G1_29DOF_JOINT_ORDER)
    bodies = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, :3] = [0.5, 0.5, 0.0]
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.arange(1, 30, dtype=np.float32)
    velocity = np.zeros((2, 35), dtype=np.float32)
    velocity[:, 6:] = np.arange(31, 60, dtype=np.float32)
    body_pos = np.asarray(
        [
            [[0.5, 0.5, 0.9], [0.2, 0.5, 0.1], [0.8, 0.5, 0.1]],
            [[0.5, 0.4, 0.9], [0.25, 0.4, 0.1], [0.75, 0.4, 0.1]],
        ],
        dtype=np.float32,
    )
    body_quat = np.zeros((2, 3, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    payload = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "joint_names": np.asarray(joints),
        "body_names": np.asarray(bodies),
        "joint_pos": qpos,
        "joint_vel": velocity,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros_like(body_pos),
        "body_ang_vel_w": np.zeros_like(body_pos),
        "robot_asset_json": np.asarray(encode_robot_asset_json()),
    }
    np.savez(path, **payload)


def test_robot_mirror_canonicalizes_equivalent_root_quaternion_signs() -> None:
    quaternions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    canonical = _canonicalize_quaternion_hemisphere_wxyz(quaternions)
    assert np.all(np.sum(canonical[:-1] * canonical[1:], axis=1) >= 0.0)
    assert np.allclose(np.abs(canonical[:, 0]), 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.npz"
        output = root / "mirrored.npz"
        _write_motion(source)
        with np.load(source, allow_pickle=True) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        qpos = np.repeat(np.asarray(payload["joint_pos"][:1]), 3, axis=0)
        qpos[:, 3:7] = quaternions
        payload["joint_pos"] = qpos
        for key in ("joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            payload[key] = np.repeat(np.asarray(payload[key][:1]), 3, axis=0)
        np.savez(source, **payload)

        mirror_motion_npz(source, output)
        with np.load(output, allow_pickle=True) as mirrored:
            root_quaternion = np.asarray(mirrored["joint_pos"][:, 3:7])
            assert np.all(np.sum(root_quaternion[:-1] * root_quaternion[1:], axis=1) >= 0.0)
