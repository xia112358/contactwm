from __future__ import annotations

import unittest

import numpy as np

from motion_edit.contact import (
    ContactPhase,
    build_load_profile_from_contact_phase,
    match_source_phase,
    phases_from_part_mask,
    resample_phase_values,
)


class ContactPhaseTests(unittest.TestCase):
    def test_splits_mask_into_contact_local_phases_with_signals(self) -> None:
        mask = np.asarray([False, True, True, False, True], dtype=bool)
        position = np.arange(15, dtype=np.float64).reshape(5, 3)
        normal = np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), (5, 1))

        phases = phases_from_part_mask(
            mask,
            part="left_foot",
            position_w=position,
            normal_w=normal,
            metadata={"source": "unit_test"},
        )

        self.assertEqual(len(phases), 2)
        self.assertEqual((phases[0].start_frame, phases[0].end_frame), (1, 3))
        self.assertEqual((phases[1].start_frame, phases[1].end_frame), (4, 5))
        np.testing.assert_allclose(phases[0].local_phase, [0.0, 1.0])
        np.testing.assert_allclose(phases[1].local_phase, [0.5])
        np.testing.assert_allclose(phases[0].position_w, position[1:3])
        self.assertEqual(phases[0].metadata["phase_index"], 0)
        self.assertEqual(phases[0].metadata["source"], "unit_test")

    def test_resamples_phase_values_in_contact_local_coordinates(self) -> None:
        values = np.asarray([[0.0], [10.0], [0.0]], dtype=np.float64)

        resampled = resample_phase_values(values, 5)

        self.assertEqual(resampled.shape, (5, 1))
        self.assertEqual(int(np.argmax(resampled[:, 0])), 2)
        self.assertAlmostEqual(float(resampled[2, 0]), 10.0)

    def test_matches_extra_target_phase_by_global_center(self) -> None:
        source = [
            ContactPhase(part="left_foot", start_frame=0, end_frame=2),
            ContactPhase(part="left_foot", start_frame=8, end_frame=10),
        ]
        target = ContactPhase(part="left_foot", start_frame=7, end_frame=9)

        matched = match_source_phase(
            source,
            target_phase=target,
            target_phase_index=4,
            target_frame_count=10,
            source_frame_count=10,
        )

        self.assertIs(matched, source[1])

    def test_contact_phase_builds_mean_one_load_profile(self) -> None:
        phase = ContactPhase(
            part="left_foot",
            start_frame=10,
            end_frame=15,
            force_envelope_w=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0],
                    [0.0, 0.0, 20.0],
                    [0.0, 0.0, 10.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )

        profile = build_load_profile_from_contact_phase(
            phase,
            normal_w=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            min_strength=0.0,
            metadata={"target_frame_start": 20, "target_frame_end": 29},
        )

        self.assertAlmostEqual(float(np.mean(profile.strength)), 1.0)
        self.assertEqual(int(profile.source_frames[0]), 10)
        self.assertEqual(int(profile.source_frames[-1]), 14)
        self.assertEqual(profile.metadata["source_frame_start"], 10)
        self.assertEqual(profile.metadata["source_frame_end"], 15)
        self.assertEqual(profile.metadata["target_frame_start"], 20)
        self.assertEqual(profile.metadata["target_frame_end"], 29)
        self.assertGreater(float(profile.evaluate([0.5])[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
