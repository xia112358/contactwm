from __future__ import annotations

from pathlib import Path

import numpy as np

from somaforge_core.canonical_input import load_canonical_qpos_input
from somaforge_core import G1_29DOF_JOINT_ORDER, encode_robot_asset_json


def test_holosoma_input_is_reordered_and_same_fps_keeps_every_frame(tmp_path: Path) -> None:
    path = tmp_path / "candidate.npz"
    names = tuple(reversed(G1_29DOF_JOINT_ORDER))
    joint_pos = np.zeros((3, 36), dtype=np.float32)
    joint_pos[:, :3] = np.asarray([1.0, 2.0, 3.0])
    joint_pos[:, 3] = 1.0
    for column, name in enumerate(names):
        joint_pos[:, 7 + column] = float(G1_29DOF_JOINT_ORDER.index(name))
    np.savez(
        path,
        joint_pos=joint_pos,
        joint_names=np.asarray(names),
        fps=np.asarray(50.0),
        robot_asset_json=np.asarray(encode_robot_asset_json()),
    )

    qpos, qvel = load_canonical_qpos_input(
        path,
        input_format="holosoma_joint_pos",
        output_fps=50.0,
    )

    assert qpos.shape == (3, 36)
    np.testing.assert_array_equal(qpos[:, :4], joint_pos[:, 3:7])
    np.testing.assert_array_equal(qpos[:, 4:7], joint_pos[:, :3])
    np.testing.assert_array_equal(qpos[0, 7:], np.arange(29, dtype=np.float32))
    assert qvel.shape == (3, 35)


def test_omniretarget_resampling_keeps_the_final_frame(tmp_path: Path) -> None:
    path = tmp_path / "raw.npz"
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 0] = 1.0
    qpos[:, 4] = np.asarray([0.0, 1.0, 2.0])
    np.savez(path, qpos=qpos, fps=np.asarray(25.0))

    converted, _ = load_canonical_qpos_input(
        path,
        input_format="omniretarget_qpos",
        output_fps=50.0,
    )

    assert converted.shape[0] == 5
    np.testing.assert_array_equal(converted[-1], qpos[-1])
