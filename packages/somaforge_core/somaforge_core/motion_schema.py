from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from somaforge_core.kinematics import POSE_FINITE_DIFFERENCE

CANONICAL_MOTION_SCHEMA = "somaforge_canonical_motion_v1"
NEWTON_KINEMATICS_BACKEND = "isaaclab3_newton_fk"
NEWTON_ROLLOUT_KINEMATICS_BACKEND = "isaaclab3_newton_rollout"
NEWTON_SIMULATION_STATE = "newton_simulation_state"
DIRECT_NEWTON_KINEMATICS_BACKEND = "newton_direct_fk"
NEWTON_KINEMATICS_BACKENDS = frozenset(
    {
        NEWTON_KINEMATICS_BACKEND,
        NEWTON_ROLLOUT_KINEMATICS_BACKEND,
        DIRECT_NEWTON_KINEMATICS_BACKEND,
    }
)

G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def newton_kinematics_provenance(
    *,
    source_path: str,
    source_sha256: str,
    output_fps: float,
    body_names: list[str],
    backend: str = NEWTON_KINEMATICS_BACKEND,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if backend not in NEWTON_KINEMATICS_BACKENDS:
        raise ValueError(f"unsupported Newton kinematics backend: {backend!r}")
    value = {
        "schema": CANONICAL_MOTION_SCHEMA,
        "kinematics_backend": str(backend),
        "source_path": str(source_path),
        "source_sha256": str(source_sha256),
        "output_fps": float(output_fps),
        "root_quaternion_order": "wxyz",
        "body_quaternion_order": "wxyz",
        "velocity_frame": "world",
        "velocity_derivation": POSE_FINITE_DIFFERENCE,
        "joint_order": list(G1_29DOF_JOINT_ORDER),
        "body_names": list(body_names),
    }
    if metadata:
        value["backend_metadata"] = dict(metadata)
    return value


def direct_newton_kinematics_provenance(
    *, source_path: str, source_sha256: str, output_fps: float, body_names: list[str], metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return newton_kinematics_provenance(
        source_path=source_path,
        source_sha256=source_sha256,
        output_fps=output_fps,
        body_names=body_names,
        backend=DIRECT_NEWTON_KINEMATICS_BACKEND,
        metadata=metadata,
    )


def newton_rollout_kinematics_provenance(
    *, source_path: str, source_sha256: str, output_fps: float, body_names: list[str]
) -> dict[str, Any]:
    return {
        "schema": CANONICAL_MOTION_SCHEMA,
        "kinematics_backend": NEWTON_ROLLOUT_KINEMATICS_BACKEND,
        "source_path": str(source_path),
        "source_sha256": str(source_sha256),
        "output_fps": float(output_fps),
        "root_quaternion_order": "wxyz",
        "body_quaternion_order": "wxyz",
        "velocity_frame": "world",
        "velocity_derivation": NEWTON_SIMULATION_STATE,
        "state_sampling": "control_step_post_physics",
        "joint_order": list(G1_29DOF_JOINT_ORDER),
        "body_names": list(body_names),
    }


def encode_kinematics_provenance(metadata: Mapping[str, Any]) -> str:
    return json.dumps(dict(metadata), sort_keys=True)


def decode_kinematics_provenance(value: Any, *, context: str, require_newton: bool = True) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{context} has no kinematics_provenance_json")
    raw = value.item() if hasattr(value, "item") else value
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid kinematics provenance") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} kinematics provenance must decode to an object")
    if str(metadata.get("schema", "")) != CANONICAL_MOTION_SCHEMA:
        raise ValueError(f"{context} uses an unsupported canonical motion schema")
    backend = str(metadata.get("kinematics_backend", ""))
    if require_newton and backend not in NEWTON_KINEMATICS_BACKENDS:
        raise ValueError(f"{context} was not canonicalized with a supported Newton FK backend")
    if tuple(metadata.get("joint_order", ())) != G1_29DOF_JOINT_ORDER:
        raise ValueError(f"{context} uses an incompatible G1 joint order")
    expected_fields = {
        "root_quaternion_order": "wxyz",
        "body_quaternion_order": "wxyz",
        "velocity_frame": "world",
    }
    for key, expected in expected_fields.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{context} has incompatible {key}: {metadata.get(key)!r}")
    expected_velocity_derivation = {
        NEWTON_KINEMATICS_BACKEND: POSE_FINITE_DIFFERENCE,
        NEWTON_ROLLOUT_KINEMATICS_BACKEND: NEWTON_SIMULATION_STATE,
        DIRECT_NEWTON_KINEMATICS_BACKEND: POSE_FINITE_DIFFERENCE,
    }.get(backend)
    if require_newton and metadata.get("velocity_derivation") != expected_velocity_derivation:
        raise ValueError(
            f"{context} has incompatible velocity_derivation for {backend}: "
            f"{metadata.get('velocity_derivation')!r}"
        )
    if backend == NEWTON_ROLLOUT_KINEMATICS_BACKEND and metadata.get("state_sampling") != "control_step_post_physics":
        raise ValueError(f"{context} has incompatible rollout state_sampling: {metadata.get('state_sampling')!r}")
    body_names = metadata.get("body_names")
    if not isinstance(body_names, list) or not body_names or any(not isinstance(name, str) for name in body_names):
        raise ValueError(f"{context} has invalid body_names provenance")
    output_fps = metadata.get("output_fps")
    if not isinstance(output_fps, (int, float)) or output_fps <= 0:
        raise ValueError(f"{context} has invalid output_fps provenance")
    return metadata
