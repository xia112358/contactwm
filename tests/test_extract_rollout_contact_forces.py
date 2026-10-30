from __future__ import annotations

import numpy as np

from scripts.extract_rollout_ref_contact_force_demo import (
    PART_BODY_NAMES,
    PART_ORDER,
    _part_force_history,
    _part_forces,
    _rollout_kinematics_provenance,
    _stable_contact_mask,
)
from somaforge_core import G1_29DOF_JOINT_ORDER, NEWTON_ROLLOUT_KINEMATICS_BACKEND, NEWTON_SIMULATION_STATE


def test_rollout_kinematics_provenance_tracks_direct_newton_recording(tmp_path) -> None:
    recording = tmp_path / "recording.npz"
    recording.write_bytes(b"newton rollout")

    provenance = _rollout_kinematics_provenance(
        recording,
        fps=50.0,
        body_names=["pelvis", "torso_link"],
        dof_names=list(G1_29DOF_JOINT_ORDER),
    )

    assert provenance["kinematics_backend"] == NEWTON_ROLLOUT_KINEMATICS_BACKEND
    assert provenance["velocity_derivation"] == NEWTON_SIMULATION_STATE
    assert provenance["source_path"] == str(recording.resolve())
    assert len(provenance["source_sha256"]) == 64


def test_part_forces_use_full_contact_sensor_bodies_for_heel_toe_split() -> None:
    sensor_body_names = [name for part in PART_ORDER for name in PART_BODY_NAMES[part]]
    history = np.zeros((2, 2, len(sensor_body_names), 3), dtype=np.float32)

    for sensor_id in range(len(sensor_body_names)):
        history[:, 0, sensor_id, 2] = float(sensor_id + 1)
        history[:, 1, sensor_id, 2] = -0.5

    forces = _part_forces(
        {"contact_sensor_forces_history": history},
        body_names=["left_ankle_roll_sphere_1_link"],
        part_body_names=PART_BODY_NAMES,
        force_reduce="sum",
        contact_sensor_body_names=sensor_body_names,
    )

    expected_z = []
    offset = 0
    for part in PART_ORDER:
        count = len(PART_BODY_NAMES[part])
        expected_z.append(sum(range(offset + 1, offset + count + 1)))
        offset += count

    assert forces.shape == (2, 8, 3)
    np.testing.assert_allclose(forces[:, :, 2], np.asarray([expected_z, expected_z], dtype=np.float32))
    assert forces[0, PART_ORDER.index("LHEE"), 2] != forces[0, PART_ORDER.index("LTOE"), 2]


def test_part_forces_use_time_aligned_latest_sample_not_per_body_history_peaks() -> None:
    sensor_body_names = [name for part in PART_ORDER for name in PART_BODY_NAMES[part]]
    history = np.zeros((1, 3, len(sensor_body_names), 3), dtype=np.float32)
    history[:, 0, :, 2] = 1.0
    history[:, 1, 0::2, 2] = 100.0
    history[:, 2, 1::2, 2] = 200.0
    latest = history[:, 0].copy()

    forces = _part_forces(
        {"contact_sensor_forces": latest, "contact_sensor_forces_history": history},
        body_names=[],
        part_body_names=PART_BODY_NAMES,
        force_reduce="sum",
        contact_sensor_body_names=sensor_body_names,
    )
    force_history = _part_force_history(
        {"contact_sensor_forces_history": history},
        body_names=[],
        part_body_names=PART_BODY_NAMES,
        force_reduce="sum",
        contact_sensor_body_names=sensor_body_names,
    )

    assert force_history is not None
    np.testing.assert_allclose(forces, force_history[:, 0])
    assert float(forces.max()) < 10.0
    assert float(force_history[:, 1:].max()) > float(forces.max())


def test_part_forces_reject_misaligned_latest_channel() -> None:
    sensor_body_names = [name for part in PART_ORDER for name in PART_BODY_NAMES[part]]
    history = np.zeros((1, 2, len(sensor_body_names), 3), dtype=np.float32)
    latest = np.ones((1, len(sensor_body_names), 3), dtype=np.float32)

    try:
        _part_forces(
            {"contact_sensor_forces": latest, "contact_sensor_forces_history": history},
            body_names=[],
            part_body_names=PART_BODY_NAMES,
            force_reduce="sum",
            contact_sensor_body_names=sensor_body_names,
        )
    except ValueError as exc:
        assert "not aligned" in str(exc)
    else:
        raise AssertionError("Misaligned latest force and force history must be rejected.")


def test_stable_contact_mask_closes_short_gaps_without_changing_force() -> None:
    force = np.zeros((7, 1, 3), dtype=np.float32)
    force[:, 0, 2] = [0.0, 12.0, 8.0, 0.0, 8.0, 12.0, 0.0]
    original = force.copy()

    raw, stable = _stable_contact_mask(
        force,
        on_threshold=10.0,
        off_threshold=5.0,
        close_gap_frames=2,
    )

    np.testing.assert_array_equal(raw[:, 0], [False, True, False, False, False, True, False])
    np.testing.assert_array_equal(stable[:, 0], [False, True, True, True, True, True, False])
    np.testing.assert_array_equal(force, original)


def test_stable_contact_mask_removes_short_pulses_and_release_gaps() -> None:
    force = np.zeros((20, 1, 3), dtype=np.float32)
    force[1:3, 0, 2] = 12.0
    force[9:18, 0, 2] = 12.0
    force[13:15, 0, 2] = 0.0

    _, stable = _stable_contact_mask(
        force,
        on_threshold=10.0,
        off_threshold=5.0,
        close_gap_frames=0,
        min_on_frames=6,
        min_off_frames=6,
    )

    expected = np.zeros(20, dtype=bool)
    expected[9:18] = True
    np.testing.assert_array_equal(stable[:, 0], expected)


def test_part_forces_reject_merged_recording_without_toe_bodies() -> None:
    recording = {"contact_forces": np.zeros((3, 1, 3), dtype=np.float32)}

    try:
        _part_forces(
            recording,
            body_names=["left_ankle_roll_sphere_1_link"],
            part_body_names=PART_BODY_NAMES,
            force_reduce="sum",
        )
    except ValueError as exc:
        assert "Re-record this rollout" in str(exc)
    else:
        raise AssertionError("Merged recordings must not silently produce an invalid 8-part force signal.")
