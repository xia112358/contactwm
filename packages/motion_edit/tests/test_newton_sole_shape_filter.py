from __future__ import annotations

import numpy as np

from motion_edit.contact.newton_bindings import bind_newton_contact_patches
from motion_edit.contact.schema import ContactAnchorRecord


def test_sole_anchor_reads_nested_heel_and_toe_shape_ids_only() -> None:
    motion = {
        "body_names": np.asarray(["left_ankle_roll_link"], dtype=object),
        "body_pos_w": np.asarray([[[0.0, 0.0, 0.0]]], dtype=np.float64),
        "body_quat_w": np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]],
            dtype=np.float64,
        ),
        "newton_body_labels": np.asarray(
            [
                "/World/envs/env_0/Robot/left_ankle_roll_link",
                "/World/ground",
            ],
            dtype=object,
        ),
        "newton_shape_labels": np.asarray(
            ["heel", "toe", "midfoot_noise", "ground"],
            dtype=object,
        ),
        "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
        "raw_contact_count": np.asarray([3], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 1, 2]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[3, 3, 3]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 0, 0]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[1, 1, 1]], dtype=np.int32),
        "raw_contact_point0_w": np.asarray(
            [[[-0.05, 0.0, -0.03], [0.14, 0.0, -0.03], [0.04, 0.0, -0.03]]],
            dtype=np.float64,
        ),
        "raw_contact_point1_w": np.asarray(
            [[[-0.04, 0.0, -0.04], [0.13, 0.0, -0.04], [0.03, 0.0, -0.04]]],
            dtype=np.float64,
        ),
        "raw_contact_normal_w": np.asarray(
            [[[0.0, 0.0, -1.0]] * 3],
            dtype=np.float64,
        ),
        "raw_contact_force_w": np.asarray(
            [[[0.0, 0.0, 100.0]] * 3],
            dtype=np.float64,
        ),
    }
    anchor = ContactAnchorRecord(
        motion_id="motion_a",
        anchor_id="left_sole_anchor",
        body="left_foot",
        start_frame=0,
        end_frame=1,
        metadata={
            "patch_role": "sole",
            "foot_subcontact": {
                "name": "sole",
                "heel": {"raw_shape_ids": [0]},
                "toe": {"raw_shape_ids": [1]},
            },
        },
    )

    patches, summary = bind_newton_contact_patches([anchor], motion)
    patch = patches[0]

    assert patch.newton_body_label == "left_ankle_roll_link"
    assert patch.newton_shape_labels == ["heel", "toe"]
    assert summary["raw_shape_ids_by_anchor"] == {
        "left_sole_anchor": [0, 1]
    }
    assert summary["filtered_raw_contact_count"] == 1
