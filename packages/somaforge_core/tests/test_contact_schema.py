from __future__ import annotations

import re

import pytest
from somaforge_core.contact_schema import (
    CONTACT_BODY_NAMES_BY_PART,
    CONTACT_FORCE_PART_ORDER,
    CONTACT_KINEMATIC_BODY_NAMES_BY_PART,
    canonical_contact_part_id,
    canonical_contact_part_name,
    decode_contact_force_provenance,
    diagnostic_contact_provenance,
    disallowed_contact_body_pattern,
    encode_contact_force_provenance,
    newton_contact_provenance,
)


def test_canonical_eight_part_aliases() -> None:
    assert canonical_contact_part_name("LHEE") == "left_heel"
    assert canonical_contact_part_id("left_toe") == "LTOE"


def test_semantic_contact_body_groups_use_sphere_hand_and_complete_feet() -> None:
    assert CONTACT_BODY_NAMES_BY_PART["left_hand"] == (
        "left_sphere_hand_link",
        "left_sphere_hand_tip_link",
    )
    assert "left_wrist_yaw_link" not in CONTACT_BODY_NAMES_BY_PART["left_hand"]
    assert CONTACT_BODY_NAMES_BY_PART["left_foot"] == (
        "left_ankle_roll_link",
        "left_ankle_roll_sphere_1_link",
        "left_ankle_roll_sphere_2_link",
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "left_ankle_roll_sphere_5_link",
    )
    assert CONTACT_KINEMATIC_BODY_NAMES_BY_PART["left_hand"] == ("left_wrist_yaw_link",)
    assert CONTACT_KINEMATIC_BODY_NAMES_BY_PART["left_foot"] == ("left_ankle_roll_link",)


def test_disallowed_contact_pattern_matches_everything_outside_allowlist() -> None:
    pattern = disallowed_contact_body_pattern(list(CONTACT_BODY_NAMES_BY_PART["left_hand"]))
    assert re.match(pattern, "left_sphere_hand_link") is None
    assert re.match(pattern, "left_sphere_hand_tip_link") is None
    assert re.match(pattern, "left_wrist_yaw_link") is not None


def test_newton_contact_provenance_round_trip() -> None:
    metadata = newton_contact_provenance(
        solver_config={"nconmax": 64, "njmax": 512, "control_decimation": 4},
        history_sample_count=3,
    )
    decoded = decode_contact_force_provenance(
        encode_contact_force_provenance(metadata), context="test force reference"
    )
    assert tuple(decoded["part_order"]) == CONTACT_FORCE_PART_ORDER
    assert decoded["training_eligible"] is True
    assert decoded["solver_config_sha256"]
    assert decoded["force_sampling"] == "latest_physics_step"
    assert decoded["position_sampling"] == "latest_physics_step"
    assert decoded["force_position_time_aligned"] is True
    assert decoded["contact_mask_filter_alters_force"] is False
    assert decoded["force_history_sample_count"] == 3
    assert decoded["force_history_complete_control_interval"] is False


def test_diagnostic_contact_force_is_rejected_for_training() -> None:
    metadata = diagnostic_contact_provenance(source_backend="mujoco_prescribed")
    with pytest.raises(ValueError, match="not sourced from the Newton"):
        decode_contact_force_provenance(
            encode_contact_force_provenance(metadata), context="test force reference"
        )


def test_diagnostic_contact_force_can_be_inspected() -> None:
    metadata = diagnostic_contact_provenance(source_backend="mujoco_prescribed")
    decoded = decode_contact_force_provenance(
        encode_contact_force_provenance(metadata), context="test force reference", require_newton=False
    )
    assert decoded["training_eligible"] is False
