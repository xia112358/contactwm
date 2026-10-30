from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

CONTACT_FORCE_SCHEMA = "somaforge_contact_force_8part_v1"
NEWTON_CONTACT_BACKEND = "isaaclab3_newton_mjwarp"
NEWTON_COLLISION_PIPELINE = "newton"

CONTACT_FORCE_PART_ORDER = (
    "LHEE",
    "LTOE",
    "RHEE",
    "RTOE",
    "LH",
    "RH",
    "LK",
    "RK",
)

CONTACT_FORCE_PART_NAMES = (
    "left_heel",
    "left_toe",
    "right_heel",
    "right_toe",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
)

CONTACT_FORCE_PART_NAME_BY_ID = dict(zip(CONTACT_FORCE_PART_ORDER, CONTACT_FORCE_PART_NAMES, strict=True))
CONTACT_FORCE_PART_ID_BY_NAME = dict(zip(CONTACT_FORCE_PART_NAMES, CONTACT_FORCE_PART_ORDER, strict=True))

CONTACT_FORCE_PART_BODY_NAMES: dict[str, tuple[str, ...]] = {
    "LHEE": ("left_ankle_roll_sphere_1_link", "left_ankle_roll_sphere_2_link"),
    "LTOE": (
        "left_ankle_roll_sphere_3_link",
        "left_ankle_roll_sphere_4_link",
        "left_ankle_roll_sphere_5_link",
    ),
    "RHEE": ("right_ankle_roll_sphere_1_link", "right_ankle_roll_sphere_2_link"),
    "RTOE": (
        "right_ankle_roll_sphere_3_link",
        "right_ankle_roll_sphere_4_link",
        "right_ankle_roll_sphere_5_link",
    ),
    "LH": ("left_sphere_hand_link", "left_sphere_hand_tip_link"),
    "RH": ("right_sphere_hand_link", "right_sphere_hand_tip_link"),
    "LK": ("left_knee_link",),
    "RK": ("right_knee_link",),
}

CONTACT_BODY_NAMES_BY_PART: dict[str, tuple[str, ...]] = {
    **{
        part_name: CONTACT_FORCE_PART_BODY_NAMES[part_id]
        for part_id, part_name in CONTACT_FORCE_PART_NAME_BY_ID.items()
    },
    "left_foot": (
        "left_ankle_roll_link",
        *CONTACT_FORCE_PART_BODY_NAMES["LHEE"],
        *CONTACT_FORCE_PART_BODY_NAMES["LTOE"],
    ),
    "right_foot": (
        "right_ankle_roll_link",
        *CONTACT_FORCE_PART_BODY_NAMES["RHEE"],
        *CONTACT_FORCE_PART_BODY_NAMES["RTOE"],
    ),
    "left_hip": ("left_hip_roll_link",),
    "right_hip": ("right_hip_roll_link",),
}

CONTACT_KINEMATIC_BODY_NAMES_BY_PART: dict[str, tuple[str, ...]] = {
    "left_heel": ("left_ankle_roll_link",),
    "left_toe": ("left_ankle_roll_link",),
    "right_heel": ("right_ankle_roll_link",),
    "right_toe": ("right_ankle_roll_link",),
    "left_foot": ("left_ankle_roll_link",),
    "right_foot": ("right_ankle_roll_link",),
    "left_hand": ("left_wrist_yaw_link",),
    "right_hand": ("right_wrist_yaw_link",),
    "left_knee": ("left_knee_link",),
    "right_knee": ("right_knee_link",),
    "left_hip": ("left_hip_roll_link",),
    "right_hip": ("right_hip_roll_link",),
}


def disallowed_contact_body_pattern(allowed_body_names: tuple[str, ...] | list[str]) -> str:
    """Build a full-match regex selecting every body except the contact allowlist."""

    names = tuple(dict.fromkeys(str(name) for name in allowed_body_names))
    exclusions = "".join(f"(?!{re.escape(name)}$)" for name in names)
    return f"^{exclusions}.+$"


def canonical_contact_part_id(value: str) -> str:
    """Return the canonical short ID for one production contact channel."""

    raw = str(value).strip()
    upper = raw.upper()
    if upper in CONTACT_FORCE_PART_NAME_BY_ID:
        return upper
    lowered = raw.lower()
    if lowered in CONTACT_FORCE_PART_ID_BY_NAME:
        return CONTACT_FORCE_PART_ID_BY_NAME[lowered]
    raise ValueError(f"unknown 8-part contact channel: {value!r}")


def canonical_contact_part_name(value: str) -> str:
    return CONTACT_FORCE_PART_NAME_BY_ID[canonical_contact_part_id(value)]


def newton_contact_provenance(
    *,
    solver_config: Mapping[str, Any],
    source_recording: str | None = None,
    force_reduce: str = "sum",
    threshold_n: float = 10.0,
    mask_off_threshold_n: float = 5.0,
    mask_close_gap_frames: int = 2,
    history_sample_count: int | None = None,
) -> dict[str, Any]:
    config = _json_safe_mapping(solver_config)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    provenance: dict[str, Any] = {
        "schema": CONTACT_FORCE_SCHEMA,
        "source_backend": NEWTON_CONTACT_BACKEND,
        "collision_pipeline": NEWTON_COLLISION_PIPELINE,
        "use_mujoco_contacts": False,
        "force_frame": "world",
        "force_unit": "N",
        "force_direction": "environment_on_robot",
        "force_channel": "contact_sensor.net_forces_w",
        "force_sampling": "latest_physics_step",
        "force_history_order": "latest_first",
        "position_sampling": "latest_physics_step",
        "force_position_time_aligned": True,
        "part_order": list(CONTACT_FORCE_PART_ORDER),
        "part_body_names": {key: list(value) for key, value in CONTACT_FORCE_PART_BODY_NAMES.items()},
        "force_reduce": str(force_reduce),
        "contact_threshold_n": float(threshold_n),
        "contact_off_threshold_n": float(mask_off_threshold_n),
        "contact_mask_close_gap_frames": int(mask_close_gap_frames),
        "contact_mask_filter_alters_force": False,
        "solver_config": config,
        "solver_config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "training_eligible": True,
    }
    if history_sample_count is not None:
        control_decimation = int(config.get("control_decimation", 0) or 0)
        provenance["force_history_sample_count"] = int(history_sample_count)
        provenance["force_history_complete_control_interval"] = (
            control_decimation > 0 and int(history_sample_count) >= control_decimation
        )
    if source_recording is not None:
        provenance["source_recording"] = str(source_recording)
    return provenance


def diagnostic_contact_provenance(*, source_backend: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": CONTACT_FORCE_SCHEMA,
        "source_backend": str(source_backend),
        "part_order": list(CONTACT_FORCE_PART_ORDER),
        "training_eligible": False,
        "metadata": _json_safe_mapping(metadata or {}),
    }


def encode_contact_force_provenance(metadata: Mapping[str, Any]) -> str:
    return json.dumps(dict(metadata), sort_keys=True)


def decode_contact_force_provenance(value: Any, *, context: str, require_newton: bool = True) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{context} has no contact_force_provenance_json")
    raw = value.item() if hasattr(value, "item") else value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid contact-force provenance") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} contact-force provenance must decode to an object")
    validate_contact_force_provenance(metadata, context=context, require_newton=require_newton)
    return metadata


def validate_contact_force_provenance(
    metadata: Mapping[str, Any], *, context: str, require_newton: bool = True
) -> None:
    if str(metadata.get("schema", "")) != CONTACT_FORCE_SCHEMA:
        raise ValueError(f"{context} uses an unsupported contact-force schema")
    order = tuple(str(item) for item in metadata.get("part_order", ()))
    if order != CONTACT_FORCE_PART_ORDER:
        raise ValueError(f"{context} contact-force part order is {order}, expected {CONTACT_FORCE_PART_ORDER}")
    if not require_newton:
        return
    if str(metadata.get("source_backend", "")) != NEWTON_CONTACT_BACKEND:
        raise ValueError(f"{context} contact force is not sourced from the Newton training backend")
    if str(metadata.get("collision_pipeline", "")) != NEWTON_COLLISION_PIPELINE:
        raise ValueError(f"{context} contact force does not use the Newton collision pipeline")
    if bool(metadata.get("use_mujoco_contacts", True)):
        raise ValueError(f"{context} contact force enables MuJoCo contacts")
    if metadata.get("training_eligible") is not True:
        raise ValueError(f"{context} contact force is diagnostic-only")
    if not str(metadata.get("solver_config_sha256", "")):
        raise ValueError(f"{context} contact force has no solver config fingerprint")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, default=str))
