from __future__ import annotations

import numpy as np

from motion_edit.contact.newton_bindings import bind_newton_contact_patches
from motion_edit.contact.schema import ContactAnchorRecord


def test_canonical_foot_body_matches_runtime_spheres_before_shape_filter() -> None:
    motion = {
        "body_names": np.asarray(["right_ankle_roll_link"], dtype=object),
        "body_pos_w": np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64),
        "body_quat_w": np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]],
            dtype=np.float64,
        ),
        "newton_body_labels": np.asarray(
            [
                "/World/envs/env_0/Robot/right_ankle_roll_sphere_1_link",
                "/World/envs/env_0/Robot/right_ankle_roll_sphere_5_link",
                "/World/ground",
            ],
            dtype=object,
        ),
        "newton_shape_labels": np.asarray(
            ["heel", "toe", "ground"],
            dtype=object,
        ),
        "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
        "raw_contact_count": np.asarray([2], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_point0_w": np.zeros((1, 2, 3), dtype=np.float64),
        "raw_contact_point1_w": np.asarray(
            [[[0.95, 0.0, -0.03], [1.14, 0.0, -0.03]]],
            dtype=np.float64,
        ),
        "raw_contact_normal_w": np.asarray(
            [[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]],
            dtype=np.float64,
        ),
        "raw_contact_force_w": np.asarray(
            [[[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        ),
    }
    anchors = [
        ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="right_foot_heel",
            body="right_foot",
            start_frame=0,
            end_frame=1,
            metadata={
                "patch_role": "heel",
                "foot_subcontact": {
                    "name": "heel",
                    "raw_shape_ids": [0],
                },
            },
        ),
        ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="right_foot_toe",
            body="right_foot",
            start_frame=0,
            end_frame=1,
            metadata={
                "patch_role": "toe",
                "foot_subcontact": {
                    "name": "toe",
                    "raw_shape_ids": [1],
                },
            },
        ),
    ]

    patches, summary = bind_newton_contact_patches(anchors, motion)
    by_anchor = {patch.anchor_id: patch for patch in patches}

    assert by_anchor["right_foot_heel"].newton_shape_labels == ["heel"]
    assert by_anchor["right_foot_toe"].newton_shape_labels == ["toe"]
    assert by_anchor["right_foot_heel"].newton_body_label == (
        "right_ankle_roll_link"
    )
    assert by_anchor["right_foot_toe"].newton_body_label == (
        "right_ankle_roll_link"
    )
    assert summary["bound_patch_count"] == 2
    assert summary["fallback_patch_count"] == 0
