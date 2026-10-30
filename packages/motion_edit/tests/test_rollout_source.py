from __future__ import annotations

import numpy as np

from somaforge_core.rollout_source import (
    complete_rollout_env_ids,
    merge_qpos,
    merge_raw_contacts,
)


def test_complete_rollout_env_ids_keeps_only_exact_full_sequences() -> None:
    timestep = np.asarray(
        [[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 1, 4]], dtype=np.int32
    )
    assert complete_rollout_env_ids(timestep, frame_count=4).tolist() == [0]


def test_merge_qpos_aligns_quaternion_signs() -> None:
    q = merge_qpos(
        root_pos=np.asarray([[[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]]),
        root_quat_xyzw=np.asarray(
            [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]]]
        ),
        dof_pos=np.asarray([[[1.0], [3.0]]]),
    )
    np.testing.assert_allclose(q[0, :3], [0.1, 0.0, 1.0])
    np.testing.assert_allclose(q[0, 3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q[0, 7:], [2.0])


def test_merge_raw_contacts_removes_env_identity_and_self_contacts() -> None:
    recording = {
        "raw_contact_count": np.asarray([3], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 2, 0]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[1, 1, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 1, 0]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[-1, -1, 1]], dtype=np.int32),
        "raw_contact_point0_w": np.asarray(
            [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        "raw_contact_point1_w": np.zeros((1, 3, 3)),
        "raw_contact_normal_w": np.tile([0.0, 0.0, 1.0], (1, 3, 1)),
        "raw_contact_force_w": np.tile([0.0, 0.0, 10.0], (1, 3, 1)),
    }
    metadata = {
        "newton_body_labels": [
            "/World/envs/env_0/Robot/left_knee_link",
            "/World/envs/env_1/Robot/left_knee_link",
        ],
        "newton_shape_labels": [
            "/World/envs/env_0/Robot/left_knee_shape",
            "/World/ground/terrain",
            "/World/envs/env_1/Robot/left_knee_shape",
        ],
    }
    fused = merge_raw_contacts(
        recording,
        metadata,
        frame_count=1,
        env_ids=[0, 1],
    )
    assert fused["raw_contact_count"].tolist() == [1]
    assert fused["raw_contact_body0"][0, 0] == 0
    assert fused["raw_contact_shape0"][0, 0] == 0
    assert fused["raw_contact_shape1"][0, 0] == 1
    assert fused["newton_body_labels"][0] == "/World/Robot/left_knee_link"
    assert fused["newton_body_labels"][1] == ""
    assert not any("env_" in value for value in fused["newton_shape_labels"])
    np.testing.assert_allclose(fused["raw_contact_point0_w"][0, 0], [2.0, 0.0, 0.0])


def test_merge_raw_contacts_preserves_env0_newton_shape_ids() -> None:
    recording = {
        "raw_contact_count": np.asarray([2], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[5, 9]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[3, 7]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[-1, -1]], dtype=np.int32),
        "raw_contact_point0_w": np.asarray(
            [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
        ),
        "raw_contact_point1_w": np.zeros((1, 2, 3)),
        "raw_contact_normal_w": np.tile([0.0, 0.0, 1.0], (1, 2, 1)),
        "raw_contact_force_w": np.tile([0.0, 0.0, 10.0], (1, 2, 1)),
    }
    body_labels = [""] * 8
    body_labels[3] = "/World/envs/env_0/Robot/right_ankle_roll_link"
    body_labels[7] = "/World/envs/env_1/Robot/right_ankle_roll_link"
    shape_labels = [""] * 10
    shape_labels[2] = "/World/ground/collision"
    shape_labels[5] = "/World/envs/env_0/Robot/right_foot/sphere_5"
    shape_labels[9] = "/World/envs/env_1/Robot/right_foot/sphere_5"

    fused = merge_raw_contacts(
        recording,
        {
            "newton_body_labels": body_labels,
            "newton_shape_labels": shape_labels,
        },
        frame_count=1,
        env_ids=[0, 1],
    )

    assert fused["raw_contact_body0"][0, 0] == 3
    assert fused["raw_contact_shape0"][0, 0] == 5
    assert fused["raw_contact_shape1"][0, 0] == 2
    assert fused["newton_shape_labels"][5] == "/World/Robot/right_foot/sphere_5"
