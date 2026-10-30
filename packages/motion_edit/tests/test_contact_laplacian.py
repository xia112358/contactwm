from __future__ import annotations

import unittest

import numpy as np

from motion_edit.contact.dynamics import ContactLoadProfile
from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.schema import ContactAnchorEditRecord, ContactAnchorRecord


class BatchContactLaplacianTests(unittest.TestCase):


    def test_contact_edit_intervals_are_half_open_and_validated(self) -> None:
        edit = ContactAnchorEditRecord(
            edit_id="edit_interval",
            motion_id="motion_a",
            anchor_id="anchor_lf",
            body="left_foot",
            delta_world=[0.1, 0.0, 0.0],
            affected_frames=[3, 12],
        )
        edit.validate()
        self.assertEqual(list(range(*edit.affected_frames)), list(range(3, 12)))
        self.assertEqual(len(range(*edit.affected_frames)), 9)

        invalid_intervals = ([3], [3, 3], [-1, 3])
        for affected_frames in invalid_intervals:
            with self.subTest(affected_frames=affected_frames):
                invalid = ContactAnchorEditRecord(
                    edit_id="bad_interval",
                    motion_id="motion_a",
                    anchor_id="anchor_lf",
                    body="left_foot",
                    delta_world=[0.1, 0.0, 0.0],
                    affected_frames=list(affected_frames),
                )
                with self.assertRaises(ValueError):
                    invalid.validate()


if __name__ == "__main__":
    unittest.main()
