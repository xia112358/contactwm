from __future__ import annotations

import numpy as np

from motion_edit.contact.newton_bindings import bind_newton_contact_patches
from motion_edit.contact.schema import ContactAnchorRecord


def test_nested_split_foot_shape_ids_use_one_canonical_ankle_frame() -> None:
    frame_count = 1
    motion = {
        # The persisted motion exposes the articulated foot frame, not runtime
        # fixed sphere bodies.
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
            ["right_heel_collision", "right_toe_collision", "ground"],
            dtype=object,
        ),
        "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
        "raw_contact_count": np.asarray([2], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_point0_w": np.asarray(
            [[[0.95, 0.0, -0.03], [1.14, 0.0, -0.03]]],
            dtype=np.float64,
        ),
        "raw_contact_point1_w": np.asarray(
            [[[0.94, 0.0, -0.04], [1.13, 0.0, -0.04]]],
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
    nested_metadata = {
        "foot_subcontact": {
            "heel": {"raw_shape_ids": [0]},
            "toe": {"raw_shape_ids": [1]},
        }
    }
    anchors = [
        ContactAnchorRecord(
            "motion_a",
            "right_heel_anchor",
            "right_heel",
            0,
            1,
            metadata=nested_metadata,
        ),
        ContactAnchorRecord(
            "motion_a",
            "right_toe_anchor",
            "right_toe",
            0,
            1,
            metadata=nested_metadata,
        ),
    ]

    patches, summary = bind_newton_contact_patches(anchors, motion)
    by_anchor = {patch.anchor_id: patch for patch in patches}
    heel = by_anchor["right_heel_anchor"]
    toe = by_anchor["right_toe_anchor"]

    assert heel.newton_shape_labels == ["right_heel_collision"]
    assert toe.newton_shape_labels == ["right_toe_collision"]
    assert heel.newton_body_label == "right_ankle_roll_link"
    assert toe.newton_body_label == "right_ankle_roll_link"
    assert heel.link_names == ["right_ankle_roll_link"]
    assert toe.link_names == ["right_ankle_roll_link"]
    np.testing.assert_allclose(heel.robot_points_local, [[-0.05, 0.0, -0.03]])
    np.testing.assert_allclose(toe.robot_points_local, [[0.14, 0.0, -0.03]])

    heel_binding = heel.metadata["newton_robot_contact_binding"]
    toe_binding = toe.metadata["newton_robot_contact_binding"]
    assert heel_binding["local_point_frame_label"] == "right_ankle_roll_link"
    assert toe_binding["local_point_frame_label"] == "right_ankle_roll_link"
    assert heel_binding["runtime_body_labels_source"] == [
        "/World/envs/env_0/Robot/right_ankle_roll_sphere_1_link"
    ]
    assert toe_binding["runtime_body_labels_source"] == [
        "/World/envs/env_0/Robot/right_ankle_roll_sphere_5_link"
    ]

    assert summary["frame_contract"] == (
        "newton_body_label_equals_robot_points_local_frame"
    )
    assert summary["strict_shape_filter_anchor_count"] == 2
    assert summary["raw_shape_ids_by_anchor"] == {
        "right_heel_anchor": [0],
        "right_toe_anchor": [1],
    }
    assert summary["local_point_frame_by_anchor"] == {
        "right_heel_anchor": "right_ankle_roll_link",
        "right_toe_anchor": "right_ankle_roll_link",
    }


def test_pca_heel_region_keeps_recorded_sphere_and_drops_generic_ankle_shape() -> None:
    motion = {
        "body_names": np.asarray(["left_ankle_roll_link"], dtype=object),
        "body_pos_w": np.zeros((1, 1, 3), dtype=np.float64),
        "body_quat_w": np.asarray([[[1.0, 0.0, 0.0, 0.0]]]),
        "newton_body_labels": np.asarray(
            ["/World/envs/env_0/Robot/left_ankle_roll_link", "/World/ground"],
            dtype=object,
        ),
        "newton_shape_labels": np.asarray(
            ["left_ankle_roll_link", "left_ankle_roll_sphere_4", "ground"],
            dtype=object,
        ),
        "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
        "raw_contact_count": np.asarray([2], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 0]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[1, 1]], dtype=np.int32),
        "raw_contact_point0_w": np.zeros((1, 2, 3), dtype=np.float64),
        "raw_contact_point1_w": np.zeros((1, 2, 3), dtype=np.float64),
        "raw_contact_normal_w": np.asarray([[[0.0, 0.0, -1.0]] * 2]),
        "raw_contact_force_w": np.zeros((1, 2, 3), dtype=np.float64),
    }
    anchor = ContactAnchorRecord(
        "motion_a",
        "left_toe_heel_region",
        "left_toe",
        0,
        1,
        metadata={
            "patch_role": "heel",
            "foot_subcontact": {"raw_shape_ids": [0, 1]},
        },
    )

    patches, summary = bind_newton_contact_patches([anchor], motion)

    assert patches[0].newton_shape_labels == ["left_ankle_roll_sphere_4"]
    assert summary["raw_shape_ids_by_anchor"] == {"left_toe_heel_region": [1]}


def test_foot_anchor_without_shape_metadata_rejects_generic_ankle_shape() -> None:
    motion = {
        "body_names": np.asarray(["right_ankle_roll_link"], dtype=object),
        "body_pos_w": np.zeros((1, 1, 3), dtype=np.float64),
        "body_quat_w": np.asarray([[[1.0, 0.0, 0.0, 0.0]]]),
        "newton_body_labels": np.asarray(
            ["/World/envs/env_0/Robot/right_ankle_roll_link", "/World/ground"],
            dtype=object,
        ),
        "newton_shape_labels": np.asarray(
            ["right_ankle_roll_link", "right_ankle_roll_sphere_1", "ground"],
            dtype=object,
        ),
        "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
        "raw_contact_count": np.asarray([2], dtype=np.int32),
        "raw_contact_shape0": np.asarray([[0, 1]], dtype=np.int32),
        "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
        "raw_contact_body0": np.asarray([[0, 0]], dtype=np.int32),
        "raw_contact_body1": np.asarray([[1, 1]], dtype=np.int32),
        "raw_contact_point0_w": np.zeros((1, 2, 3), dtype=np.float64),
        "raw_contact_point1_w": np.zeros((1, 2, 3), dtype=np.float64),
        "raw_contact_normal_w": np.asarray([[[0.0, 0.0, -1.0]] * 2]),
        "raw_contact_force_w": np.zeros((1, 2, 3), dtype=np.float64),
    }
    anchor = ContactAnchorRecord(
        "motion_a", "right_heel_without_ids", "right_heel", 0, 1
    )

    patches, summary = bind_newton_contact_patches([anchor], motion)

    assert patches[0].newton_shape_labels == ["right_ankle_roll_sphere_1"]
    assert summary["filtered_raw_contact_count"] == 1
