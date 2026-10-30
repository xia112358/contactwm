from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from holosoma.managers.command.terms.wbt import MultiMotionLoader
from somaforge_core import (
    G1_29DOF_JOINT_ORDER,
    encode_kinematics_provenance,
    encode_robot_asset_json,
    newton_kinematics_provenance,
)


def _write_motion(
    path: Path,
    body_names: list[str],
    *,
    fps: float = 50.0,
    sim_velocity: float | None = None,
) -> None:
    frames = 3
    joint_pos = np.zeros((frames, 36), dtype=np.float32)
    joint_pos[:, 2] = 1.0
    joint_pos[:, 3] = 1.0
    joint_vel = np.zeros((frames, 35), dtype=np.float32)
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_pos[:, body_names.index("pelvis"), 2] = 1.0
    body_pos[:, body_names.index("torso_link"), 2] = 1.5
    body_vel = np.zeros_like(body_pos)
    provenance = newton_kinematics_provenance(
        source_path="source.npz",
        source_sha256="a" * 64,
        output_fps=fps,
        body_names=body_names,
    )
    payload = dict(
        fps=np.asarray([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_vel,
        body_ang_vel_w=body_vel,
        body_names=np.asarray(body_names),
        joint_names=np.asarray(G1_29DOF_JOINT_ORDER),
        robot_asset_json=np.asarray(encode_robot_asset_json()),
        kinematics_provenance_json=np.asarray(encode_kinematics_provenance(provenance)),
    )
    if sim_velocity is not None:
        sim_joint_vel = np.full_like(joint_vel, sim_velocity)
        payload.update(
            sim_joint_vel=sim_joint_vel,
            sim_root_lin_vel=np.full((frames, 3), sim_velocity, dtype=np.float32),
            sim_root_ang_vel=np.full((frames, 3), -sim_velocity, dtype=np.float32),
        )
    np.savez_compressed(path, **payload)


def _load(directory: Path) -> MultiMotionLoader:
    return MultiMotionLoader(
        str(directory),
        ["pelvis", "torso_link"],
        list(G1_29DOF_JOINT_ORDER),
        device="cpu",
    )


def test_multi_motion_canonicalizes_each_body_order(tmp_path: Path) -> None:
    _write_motion(tmp_path / "a.npz", ["pelvis", "torso_link"])
    _write_motion(tmp_path / "b.npz", ["torso_link", "pelvis"])

    motion = _load(tmp_path)

    np.testing.assert_allclose(motion.body_pos_w[:, 0, 2].numpy(), 1.0)
    np.testing.assert_allclose(motion.body_pos_w[:, 1, 2].numpy(), 1.5)


def test_multi_motion_fails_closed_on_invalid_file(tmp_path: Path) -> None:
    _write_motion(tmp_path / "a.npz", ["pelvis", "torso_link"])
    np.savez_compressed(tmp_path / "b.npz", fps=np.asarray([50.0], dtype=np.float32))

    with pytest.raises(ValueError, match="missing keys"):
        _load(tmp_path)


def test_multi_motion_rejects_mixed_fps(tmp_path: Path) -> None:
    _write_motion(tmp_path / "a.npz", ["pelvis", "torso_link"], fps=50.0)
    _write_motion(tmp_path / "b.npz", ["pelvis", "torso_link"], fps=40.0)

    with pytest.raises(ValueError, match="same FPS"):
        _load(tmp_path)


def test_multi_motion_keeps_reference_and_reset_velocities_separate(tmp_path: Path) -> None:
    _write_motion(tmp_path / "a.npz", ["pelvis", "torso_link"], sim_velocity=2.0)

    motion = _load(tmp_path)

    np.testing.assert_allclose(motion.joint_vel.numpy(), 0.0)
    np.testing.assert_allclose(motion.sim_joint_vel.numpy(), 2.0)
    np.testing.assert_allclose(motion.sim_root_lin_vel.numpy(), 2.0)
    np.testing.assert_allclose(motion.sim_root_ang_vel.numpy(), -2.0)
