from __future__ import annotations

import unittest

import numpy as np

from motion_edit.contact.stable_proto import StableContactProtoConfig, stable_contact_anchor_maps


class StableContactProtoTests(unittest.TestCase):
    def test_stable_touchdown_becomes_global_anchor(self) -> None:
        contact = np.zeros((12, 2), dtype=bool)
        contact[0:3, 0] = True
        contact[7:10, 0] = True
        contact[:, 1] = True

        anchors, touchdown_by_anchor, metadata = stable_contact_anchor_maps(
            contact_mask=contact,
            body_names=["left_foot", "right_hand"],
            config=StableContactProtoConfig(
                merge_transition_window=0,
                stable_touchdown_window=3,
                stable_touchdown_cluster_window=6,
            ),
        )

        self.assertEqual(anchors, [0, 7, 12])
        self.assertEqual(metadata["segmentation_kind"], "stable_contact_anchor")
        self.assertEqual(metadata["stable_touchdown_count"], 1)
        self.assertEqual(touchdown_by_anchor[7].tolist(), [True, False])

    def test_close_touchdowns_are_clustered(self) -> None:
        contact = np.zeros((16, 2), dtype=bool)
        contact[0:3, 0] = True
        contact[8:12, 0] = True
        contact[10:14, 1] = True

        anchors, touchdown_by_anchor, metadata = stable_contact_anchor_maps(
            contact_mask=contact,
            body_names=["left_foot", "right_foot"],
            config=StableContactProtoConfig(
                merge_transition_window=0,
                stable_touchdown_window=3,
                stable_touchdown_cluster_window=6,
            ),
        )

        self.assertEqual(anchors, [0, 10, 16])
        self.assertEqual(touchdown_by_anchor[10].tolist(), [True, True])
        self.assertEqual(metadata["internal_anchor_count"], 1)

    def test_velocity_delays_anchor_until_contact_is_stable(self) -> None:
        contact = np.zeros((14, 1), dtype=bool)
        contact[6:12, 0] = True
        pos = np.zeros((14, 1, 3), dtype=np.float64)
        pos[6, 0, 0] = 0.0
        pos[7, 0, 0] = 1.0
        pos[8:, 0, 0] = 1.0

        anchors, touchdown_by_anchor, _metadata = stable_contact_anchor_maps(
            contact_mask=contact,
            body_pos_w=pos,
            body_names=["left_foot"],
            config=StableContactProtoConfig(
                fps=1.0,
                merge_transition_window=0,
                stable_touchdown_window=2,
                stable_touchdown_speed_thresh=0.2,
                stable_touchdown_cluster_window=2,
            ),
        )

        self.assertEqual(anchors, [0, 8, 14])
        self.assertIn(8, touchdown_by_anchor)


if __name__ == "__main__":
    unittest.main()
