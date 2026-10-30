from __future__ import annotations

import unittest

import numpy as np

from motion_edit.contact.newton_bindings import (
    bind_newton_contact_patches,
    body_local_point_to_world,
    world_point_to_body_local,
)
from motion_edit.contact.schema import ContactAnchorRecord


class NewtonContactBindingTests(unittest.TestCase):
    def test_world_body_point_roundtrip(self) -> None:
        point_w = np.asarray([1.2, -0.4, 0.7], dtype=np.float64)
        body_pos = np.asarray([1.0, -0.5, 0.5], dtype=np.float64)
        body_quat_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        point_b = world_point_to_body_local(point_w, body_pos, body_quat_wxyz)
        reconstructed = body_local_point_to_world(point_b, body_pos, body_quat_wxyz)

        np.testing.assert_allclose(point_b, [0.2, 0.1, 0.2], atol=1.0e-9)
        np.testing.assert_allclose(reconstructed, point_w, atol=1.0e-9)

    def test_bind_patch_uses_one_persisted_local_frame(self) -> None:
        frame_count = 3
        body_pos = np.zeros((frame_count, 1, 3), dtype=np.float64)
        body_pos[:, 0, 0] = np.arange(frame_count, dtype=np.float64)
        body_quat = np.zeros((frame_count, 1, 4), dtype=np.float64)
        body_quat[..., 0] = 1.0

        point0 = np.zeros((frame_count, 1, 3), dtype=np.float64)
        point0[:, 0] = body_pos[:, 0] + np.asarray([0.0, 0.0, -0.1])
        point1 = np.zeros((frame_count, 1, 3), dtype=np.float64)
        point1[:, 0] = body_pos[:, 0] + np.asarray([0.0, 0.0, -0.2])

        motion = {
            "body_names": np.asarray(["left_ankle_roll_link"], dtype=object),
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "newton_body_labels": np.asarray(
                [
                    "/World/envs/env_0/Robot/left_ankle_roll_link",
                    "/World/ground",
                ],
                dtype=object,
            ),
            "newton_shape_labels": np.asarray(
                [
                    "/World/envs/env_0/Robot/left_ankle_roll_link/heel_collision",
                    "/World/ground/collision",
                ],
                dtype=object,
            ),
            "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
            "raw_contact_count": np.ones(frame_count, dtype=np.int32),
            "raw_contact_shape0": np.zeros((frame_count, 1), dtype=np.int32),
            "raw_contact_shape1": np.ones((frame_count, 1), dtype=np.int32),
            "raw_contact_body0": np.zeros((frame_count, 1), dtype=np.int32),
            "raw_contact_body1": np.ones((frame_count, 1), dtype=np.int32),
            "raw_contact_point0_w": point0,
            "raw_contact_point1_w": point1,
            "raw_contact_normal_w": np.tile(
                np.asarray([0.0, 0.0, -1.0]),
                (frame_count, 1, 1),
            ),
            "raw_contact_force_w": np.tile(
                np.asarray([0.0, 0.0, 100.0]),
                (frame_count, 1, 1),
            ),
        }
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            start_frame=0,
            end_frame=frame_count,
            world_position=[1.0, 0.0, -0.1],
        )

        patches, summary = bind_newton_contact_patches(
            [anchor],
            motion,
            min_force_norm=10.0,
        )

        self.assertEqual(summary["bound_patch_count"], 1)
        self.assertEqual(summary["fallback_patch_count"], 0)
        self.assertEqual(summary["raw_sample_count"], frame_count)
        self.assertEqual(
            summary["frame_contract"],
            "newton_body_label_equals_robot_points_local_frame",
        )
        self.assertEqual(len(patches), 1)

        patch = patches[0]
        self.assertEqual(patch.robot_binding_backend, "newton_mjwarp")
        self.assertEqual(patch.robot_binding_source, "newton_raw_contact")
        self.assertEqual(patch.newton_body_label, "left_ankle_roll_link")
        self.assertEqual(patch.link_names, ["left_ankle_roll_link"])
        self.assertEqual(
            patch.newton_shape_labels,
            ["/World/envs/env_0/Robot/left_ankle_roll_link/heel_collision"],
        )
        np.testing.assert_allclose(
            np.asarray(patch.robot_points_local),
            [[0.0, 0.0, -0.1]],
            atol=1.0e-9,
        )
        self.assertEqual(patch.source_target_frames, [0, 1, 2])
        np.testing.assert_allclose(
            patch.source_target_points_w,
            point1,
            atol=1.0e-9,
        )
        binding = patch.metadata["newton_robot_contact_binding"]
        self.assertEqual(binding["local_point_frame_label"], "left_ankle_roll_link")
        self.assertEqual(
            binding["runtime_body_labels_source"],
            ["/World/envs/env_0/Robot/left_ankle_roll_link"],
        )
        self.assertLess(float(binding["reconstruction_error_max_m"]), 1.0e-9)

    def test_unmatched_anchor_falls_back_without_inventing_local_point(self) -> None:
        motion = {
            "body_names": np.asarray(["left_ankle_roll_link"], dtype=object),
            "body_pos_w": np.zeros((1, 1, 3), dtype=np.float64),
            "body_quat_w": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]]],
                dtype=np.float64,
            ),
            "newton_body_labels": np.asarray(
                ["/World/envs/env_0/Robot/left_ankle_roll_link"],
                dtype=object,
            ),
            "raw_contact_count": np.zeros(1, dtype=np.int32),
            "raw_contact_shape0": np.full((1, 1), -1, dtype=np.int32),
            "raw_contact_shape1": np.full((1, 1), -1, dtype=np.int32),
            "raw_contact_body0": np.full((1, 1), -1, dtype=np.int32),
            "raw_contact_body1": np.full((1, 1), -1, dtype=np.int32),
            "raw_contact_point0_w": np.zeros((1, 1, 3), dtype=np.float64),
            "raw_contact_point1_w": np.zeros((1, 1, 3), dtype=np.float64),
            "raw_contact_normal_w": np.zeros((1, 1, 3), dtype=np.float64),
            "raw_contact_force_w": np.zeros((1, 1, 3), dtype=np.float64),
        }
        anchor = ContactAnchorRecord("motion_a", "anchor_lh", "left_hand", 0, 1)

        patches, summary = bind_newton_contact_patches([anchor], motion)

        self.assertEqual(summary["bound_patch_count"], 0)
        self.assertEqual(patches[0].robot_binding_source, "legacy_center")
        self.assertIsNone(patches[0].robot_points_local)

    def test_robot_body1_uses_owned_point1_and_counterpart_target_point0(self) -> None:
        motion = {
            "body_names": np.asarray(["left_ankle_roll_link"], dtype=object),
            "body_pos_w": np.asarray([[[0.0, 0.0, 0.75]]], dtype=np.float64),
            "body_quat_w": np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64),
            "newton_body_labels": np.asarray(
                ["/World/ground", "/World/envs/env_0/Robot/left_ankle_roll_link"],
                dtype=object,
            ),
            "newton_shape_labels": np.asarray(["ground", "left_sole"], dtype=object),
            "raw_contact_selected_env_id": np.asarray(0, dtype=np.int32),
            "raw_contact_count": np.asarray([1], dtype=np.int32),
            "raw_contact_shape0": np.asarray([[0]], dtype=np.int32),
            "raw_contact_shape1": np.asarray([[1]], dtype=np.int32),
            "raw_contact_body0": np.asarray([[-1]], dtype=np.int32),
            "raw_contact_body1": np.asarray([[1]], dtype=np.int32),
            "raw_contact_point0_w": np.asarray([[[0.1, 0.0, 0.70]]], dtype=np.float64),
            "raw_contact_point1_w": np.asarray([[[0.1, 0.0, 0.72]]], dtype=np.float64),
            "raw_contact_normal_w": np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float64),
            "raw_contact_force_w": np.asarray([[[0.0, 0.0, 100.0]]], dtype=np.float64),
        }
        anchor = ContactAnchorRecord("motion_a", "left_foot", "left_foot", 0, 1)

        patches, summary = bind_newton_contact_patches([anchor], motion)

        np.testing.assert_allclose(patches[0].robot_points_local, [[0.1, 0.0, -0.03]])
        np.testing.assert_allclose(
            patches[0].source_target_points_w,
            [[[0.1, 0.0, 0.70]]],
        )
        self.assertEqual(summary["point_body_pairing"], "body0->point0,body1->point1")

    def test_heel_and_toe_bind_to_distinct_shapes_in_one_foot_frame(self) -> None:
        frame_count = 1
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
                ["right_heel_collision", "right_toe_collision", "ground"],
                dtype=object,
            ),
            "raw_contact_count": np.asarray([2], dtype=np.int32),
            "raw_contact_shape0": np.asarray([[0, 1]], dtype=np.int32),
            "raw_contact_shape1": np.asarray([[2, 2]], dtype=np.int32),
            "raw_contact_body0": np.asarray([[0, 1]], dtype=np.int32),
            "raw_contact_body1": np.asarray([[2, 2]], dtype=np.int32),
            "raw_contact_point0_w": np.asarray(
                [[[0.95, 0.0, -0.03], [1.14, 0.0, -0.03]]],
                dtype=np.float64,
            ),
            "raw_contact_point1_w": np.zeros((frame_count, 2, 3), dtype=np.float64),
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
            ContactAnchorRecord("motion_a", "right_heel", "right_heel", 0, 1),
            ContactAnchorRecord("motion_a", "right_toe", "right_toe", 0, 1),
        ]

        patches, summary = bind_newton_contact_patches(anchors, motion)
        by_anchor = {patch.anchor_id: patch for patch in patches}

        self.assertEqual(summary["bound_patch_count"], 2)
        self.assertEqual(
            by_anchor["right_heel"].newton_shape_labels,
            ["right_heel_collision"],
        )
        self.assertEqual(
            by_anchor["right_toe"].newton_shape_labels,
            ["right_toe_collision"],
        )
        self.assertEqual(
            by_anchor["right_heel"].newton_body_label,
            "right_ankle_roll_link",
        )
        self.assertEqual(
            by_anchor["right_toe"].newton_body_label,
            "right_ankle_roll_link",
        )

    def test_top_anchor_rejects_counterpart_samples_off_top_surface(self) -> None:
        motion = {
            "body_names": np.asarray(["right_knee_link"], dtype=object),
            "body_pos_w": np.zeros((2, 1, 3), dtype=np.float64),
            "body_quat_w": np.asarray(
                [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
                dtype=np.float64,
            ),
            "newton_body_labels": np.asarray(
                ["/World/envs/env_0/Robot/right_knee_link", "/World/terrain"],
                dtype=object,
            ),
            "newton_shape_labels": np.asarray(["knee", "terrain"], dtype=object),
            "raw_contact_count": np.ones(2, dtype=np.int32),
            "raw_contact_shape0": np.zeros((2, 1), dtype=np.int32),
            "raw_contact_shape1": np.ones((2, 1), dtype=np.int32),
            "raw_contact_body0": np.zeros((2, 1), dtype=np.int32),
            "raw_contact_body1": np.ones((2, 1), dtype=np.int32),
            "raw_contact_point0_w": np.zeros((2, 1, 3), dtype=np.float64),
            "raw_contact_point1_w": np.asarray(
                [[[0.1, 0.1, 0.7]], [[0.1, 0.1, 0.67]]],
                dtype=np.float64,
            ),
            "raw_contact_normal_w": np.asarray(
                [[[0.0, 0.0, -1.0]], [[0.0, 0.0, -1.0]]],
                dtype=np.float64,
            ),
            "raw_contact_force_w": np.ones((2, 1, 3), dtype=np.float64),
        }
        anchor = ContactAnchorRecord(
            motion_id="motion_a",
            anchor_id="right_knee_top",
            body="right_knee",
            start_frame=0,
            end_frame=2,
            surface_id="box_top",
            surface_origin=[0.0, 0.0, 0.7],
            surface_normal=[0.0, 0.0, 1.0],
            surface_tangent_u=[1.0, 0.0, 0.0],
            surface_tangent_v=[0.0, 1.0, 0.0],
            surface_bounds={"u": [-0.5, 0.5], "v": [-0.5, 0.5]},
        )

        patches, summary = bind_newton_contact_patches([anchor], motion)

        self.assertEqual(summary["surface_filtered_raw_contact_count"], 1)
        np.testing.assert_allclose(
            patches[0].source_target_points_w,
            [[[0.1, 0.1, 0.7]], [[0.1, 0.1, 0.7]]],
        )


if __name__ == "__main__":
    unittest.main()
