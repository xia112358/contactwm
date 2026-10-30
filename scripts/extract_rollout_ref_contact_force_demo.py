#!/usr/bin/env python3
"""Build a rollout-ref contact-force motion from a complete eval recording."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from somaforge_core.contact_schema import (
    CONTACT_FORCE_PART_BODY_NAMES,
    CONTACT_FORCE_PART_ORDER,
    NEWTON_COLLISION_PIPELINE,
    NEWTON_CONTACT_BACKEND,
    encode_contact_force_provenance,
    newton_contact_provenance,
)
from somaforge_core.motion_schema import (
    G1_29DOF_JOINT_ORDER,
    encode_kinematics_provenance,
    newton_rollout_kinematics_provenance,
)
from somaforge_core.asset_registry import sha256_file
from somaforge_core.robot_assets import decode_robot_asset_json, encode_robot_asset_json


PART_BODY_NAMES: dict[str, list[str]] = {
    part: list(CONTACT_FORCE_PART_BODY_NAMES[part]) for part in CONTACT_FORCE_PART_ORDER
}
PART_ORDER = CONTACT_FORCE_PART_ORDER
ENV_RE = re.compile(r"/envs/env_(\d+)/")

PART_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
    part: tuple(names) for part, names in CONTACT_FORCE_PART_BODY_NAMES.items()
}


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _metadata(recording: dict[str, Any]) -> dict[str, Any]:
    if "_metadata_json" not in recording:
        raise ValueError("Recording is missing _metadata_json.")
    return json.loads(str(recording["_metadata_json"].item()))


def _validate_newton_recording_metadata(metadata: dict[str, Any], recording_path: Path) -> dict[str, Any]:
    if str(metadata.get("contact_source_backend", "")) != NEWTON_CONTACT_BACKEND:
        raise ValueError(f"{recording_path} is not an Isaac Lab 3/Newton recording")
    if str(metadata.get("contact_collision_pipeline", "")) != NEWTON_COLLISION_PIPELINE:
        raise ValueError(f"{recording_path} does not use the Newton collision pipeline")
    if bool(metadata.get("contact_use_mujoco_contacts", True)):
        raise ValueError(f"{recording_path} enables MuJoCo contacts and is not training eligible")
    solver_config = metadata.get("newton_solver_config")
    if not isinstance(solver_config, dict) or not solver_config:
        raise ValueError(f"{recording_path} is missing newton_solver_config metadata")
    return solver_config


def _rollout_kinematics_provenance(
    recording_path: Path,
    *,
    fps: float,
    body_names: list[str],
    dof_names: list[str],
) -> dict[str, Any]:
    if tuple(dof_names) != G1_29DOF_JOINT_ORDER:
        raise ValueError("Recording DOF order does not match the canonical G1 29-DOF order.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Recording FPS must be finite and positive, got {fps}.")
    return newton_rollout_kinematics_provenance(
        source_path=str(recording_path.resolve()),
        source_sha256=sha256_file(recording_path),
        output_fps=fps,
        body_names=body_names,
    )


def _with_metadata(recording: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    updated = dict(recording)
    updated["_metadata_json"] = np.asarray(json.dumps(metadata))
    return updated


def _slice_recording_env(recording: dict[str, Any], env_id: int, num_envs: int, metadata: dict[str, Any]) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in recording.items():
        if key == "_metadata_json":
            continue
        array = np.asarray(value)
        if not key.startswith(('solver_contact_', 'raw_contact_')) and array.ndim >= 2 and array.shape[1] == num_envs:
            sliced[key] = array[:, env_id]
        else:
            sliced[key] = array

    env_done_steps = metadata.get("env_done_steps")
    env_done_timeouts = metadata.get("env_done_is_timeout")
    env_done_terms = metadata.get("env_done_terms")
    selected_meta = dict(metadata)
    selected_meta["env_id"] = int(env_id)
    selected_meta["selected_env_id"] = int(env_id)
    if isinstance(env_done_steps, list) and env_id < len(env_done_steps) and env_done_steps[env_id] is not None:
        selected_meta["episode_done"] = True
        selected_meta["done_step"] = int(env_done_steps[env_id])
        selected_meta["done_is_timeout"] = (
            bool(env_done_timeouts[env_id])
            if isinstance(env_done_timeouts, list) and env_id < len(env_done_timeouts)
            else False
        )
        selected_meta["done_terms"] = (
            list(env_done_terms[env_id]) if isinstance(env_done_terms, list) and env_id < len(env_done_terms) else []
        )
    else:
        selected_meta["episode_done"] = False
        selected_meta.pop("done_step", None)
        selected_meta.pop("done_is_timeout", None)
        selected_meta.pop("done_terms", None)
    return _with_metadata(sliced, selected_meta)


def _select_recording_env(recording: dict[str, Any], num_frames: int, allow_timeout: bool) -> tuple[dict[str, Any], int | None]:
    metadata = _metadata(recording)
    time_steps = np.asarray(recording.get("motion_time_step"))
    if time_steps.ndim != 2:
        return recording, None

    num_envs = int(time_steps.shape[1])
    errors: list[str] = []
    for env_id in range(num_envs):
        candidate = _slice_recording_env(recording, env_id, num_envs, metadata)
        try:
            _validate_complete_rollout(candidate, num_frames, allow_timeout)
        except ValueError as exc:
            errors.append(f"env{env_id}: {exc}")
            continue
        return candidate, env_id
    raise ValueError("No parallel eval environment contains a complete successful rollout. " + " | ".join(errors[:8]))


def _validate_complete_rollout(recording: dict[str, Any], num_frames: int, allow_timeout: bool) -> np.ndarray:
    required = ("motion_id", "motion_time_step", "terminated", "timeout")
    missing = [key for key in required if key not in recording]
    if missing:
        raise ValueError(f"Recording is missing required rollout keys: {missing}")

    motion_ids = np.asarray(recording["motion_id"]).reshape(-1)
    if np.unique(motion_ids).size != 1:
        raise ValueError(f"Recording must contain exactly one motion_id, got {np.unique(motion_ids).tolist()}")

    time_steps = np.asarray(recording["motion_time_step"]).reshape(-1).astype(np.int64)
    expected = np.arange(num_frames, dtype=np.int64)
    terminated = np.asarray(recording["terminated"]).reshape(-1).astype(bool)
    timeout = np.asarray(recording["timeout"]).reshape(-1).astype(bool)
    metadata = _metadata(recording)
    if bool(metadata.get("episode_done", False)):
        done_step = metadata.get("done_step")
        done_is_timeout = bool(metadata.get("done_is_timeout", False))
        if done_is_timeout and allow_timeout:
            pass
        else:
            raise ValueError(
                "Recording ended early because the eval episode reset. "
                f"done_step={done_step}, done_is_timeout={done_is_timeout}, "
                f"recording_length={time_steps.size}, motion_length={num_frames}"
            )
    if not (time_steps.size == terminated.size == timeout.size == motion_ids.size):
        raise ValueError(
            "Rollout keys must have matching lengths: "
            f"motion_id={motion_ids.size}, motion_time_step={time_steps.size}, "
            f"terminated={terminated.size}, timeout={timeout.size}"
        )

    for start in range(0, time_steps.size - num_frames + 1):
        stop = start + num_frames
        window_steps = time_steps[start:stop]
        if not np.array_equal(window_steps, expected):
            continue
        if terminated[start:stop].any():
            continue
        if timeout[start:stop].any() and not allow_timeout:
            continue
        return np.arange(start, stop)

    unique_steps = np.unique(time_steps)
    missing = sorted(set(expected.tolist()) - set(unique_steps.tolist()))[:10]
    duplicate_count = int(max(0, time_steps.size - unique_steps.size))
    first_terminated = int(np.flatnonzero(terminated)[0]) if terminated.any() else None
    first_timeout = int(np.flatnonzero(timeout)[0]) if timeout.any() else None
    raise ValueError(
        "Recording does not contain a complete normal motion cycle. "
        f"recording_length={time_steps.size}, motion_length={num_frames}, "
        f"missing_head={missing}, duplicate_count={duplicate_count}, "
        f"first_terminated={first_terminated}, first_timeout={first_timeout}"
    )


def _latest_body_force_from_history(contact_history: np.ndarray) -> np.ndarray:
    if contact_history.ndim != 4 or contact_history.shape[-1] != 3:
        raise ValueError(f"Expected contact_forces_history [T,H,B,3], got {contact_history.shape}")
    if contact_history.shape[1] == 0:
        raise ValueError("Contact force history has no physics-step samples.")
    # Isaac Lab stores the most recent physics step at history index zero. Raw
    # Newton contact points are read from that same step, so this is the only
    # history sample that is time-aligned with the recorded contact geometry.
    return contact_history[:, 0]


def _reduce_part_force(selected: np.ndarray, reduce: str) -> np.ndarray:
    if reduce == "sum":
        return selected.sum(axis=1)
    if reduce == "mean":
        return selected.mean(axis=1)
    if reduce == "max":
        idx = np.linalg.norm(selected, axis=-1).argmax(axis=1)
        return selected[np.arange(selected.shape[0]), idx]
    raise ValueError(f"Unsupported --force-reduce '{reduce}'.")


def _part_forces(
    recording: dict[str, Any],
    body_names: list[str],
    part_body_names: dict[str, list[str]],
    force_reduce: str,
    contact_sensor_body_names: list[str] | None = None,
) -> np.ndarray:
    if "contact_sensor_forces" in recording:
        if not contact_sensor_body_names:
            raise ValueError("Recording has contact_sensor_forces but no contact_sensor_body_names metadata.")
        body_force = np.asarray(recording["contact_sensor_forces"], dtype=np.float32)
        force_body_names = contact_sensor_body_names
        if "contact_sensor_forces_history" in recording:
            latest = _latest_body_force_from_history(
                np.asarray(recording["contact_sensor_forces_history"], dtype=np.float32)
            )
            if latest.shape != body_force.shape or not np.allclose(latest, body_force, rtol=1e-5, atol=1e-5):
                raise ValueError("contact_sensor_forces is not aligned with history index zero.")
    elif "contact_sensor_forces_history" in recording:
        if not contact_sensor_body_names:
            raise ValueError("Recording has contact_sensor_forces_history but no contact_sensor_body_names metadata.")
        body_force = _latest_body_force_from_history(
            np.asarray(recording["contact_sensor_forces_history"], dtype=np.float32)
        )
        force_body_names = contact_sensor_body_names
    elif "contact_forces" in recording:
        body_force = np.asarray(recording["contact_forces"], dtype=np.float32)
        force_body_names = body_names
    elif "contact_forces_history" in recording:
        body_force = _latest_body_force_from_history(
            np.asarray(recording["contact_forces_history"], dtype=np.float32)
        )
        force_body_names = body_names
    else:
        raise ValueError("Recording has no contact sensor force channel.")

    if body_force.shape[1] != len(force_body_names):
        raise ValueError(
            "Contact force body dimension does not match its body-name metadata: "
            f"forces={body_force.shape}, body_names={len(force_body_names)}"
        )

    part_forces = []
    for part_name in PART_ORDER:
        names = part_body_names[part_name]
        ids = [force_body_names.index(name) for name in names if name in force_body_names]
        if not ids:
            raise ValueError(
                f"No force sensor bodies found for part {part_name}: {names}. "
                "Re-record this rollout with full contact_sensor_forces enabled."
            )
        selected = body_force[:, ids, :]
        part_forces.append(_reduce_part_force(selected, force_reduce))
    return np.stack(part_forces, axis=1).astype(np.float32)


def _part_force_history(
    recording: dict[str, Any],
    body_names: list[str],
    part_body_names: dict[str, list[str]],
    force_reduce: str,
    contact_sensor_body_names: list[str] | None = None,
) -> np.ndarray | None:
    if "contact_sensor_forces_history" in recording:
        history = np.asarray(recording["contact_sensor_forces_history"], dtype=np.float32)
        force_body_names = contact_sensor_body_names or []
    elif "contact_forces_history" in recording:
        history = np.asarray(recording["contact_forces_history"], dtype=np.float32)
        force_body_names = body_names
    else:
        return None
    if history.ndim != 4 or history.shape[-1] != 3:
        raise ValueError(f"Expected contact force history [T,H,B,3], got {history.shape}")
    if history.shape[2] != len(force_body_names):
        raise ValueError(
            "Contact force history body dimension does not match its body-name metadata: "
            f"history={history.shape}, body_names={len(force_body_names)}"
        )

    part_history = []
    for part_name in PART_ORDER:
        ids = [force_body_names.index(name) for name in part_body_names[part_name] if name in force_body_names]
        if not ids:
            raise ValueError(f"No force sensor bodies found for part {part_name}: {part_body_names[part_name]}")
        selected = history[:, :, ids, :]
        if force_reduce == "sum":
            reduced = selected.sum(axis=2)
        elif force_reduce == "mean":
            reduced = selected.mean(axis=2)
        elif force_reduce == "max":
            max_ids = np.linalg.norm(selected, axis=-1).argmax(axis=2)
            reduced = np.take_along_axis(selected, max_ids[:, :, None, None], axis=2).squeeze(2)
        else:
            raise ValueError(f"Unsupported --force-reduce '{force_reduce}'.")
        part_history.append(reduced)
    return np.stack(part_history, axis=2).astype(np.float32)


def _stable_contact_mask(
    part_force: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    close_gap_frames: int,
    min_on_frames: int = 0,
    min_off_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= off_threshold <= on_threshold:
        raise ValueError("Contact mask thresholds must satisfy 0 <= off_threshold <= on_threshold.")
    magnitude = np.linalg.norm(part_force, axis=-1)
    raw = magnitude > on_threshold
    stable = np.zeros_like(raw)
    active = np.zeros(raw.shape[1], dtype=bool)
    for frame_idx in range(raw.shape[0]):
        active = np.where(active, magnitude[frame_idx] > off_threshold, raw[frame_idx])
        stable[frame_idx] = active

    if min_on_frames < 0 or min_off_frames < 0:
        raise ValueError("Minimum contact on/off frames must be nonnegative.")
    if close_gap_frames > 0:
        for part_idx in range(stable.shape[1]):
            values = stable[:, part_idx]
            start = 0
            while start < values.size:
                end = start + 1
                while end < values.size and values[end] == values[start]:
                    end += 1
                is_internal_gap = not values[start] and start > 0 and end < values.size
                if is_internal_gap and end - start <= close_gap_frames:
                    values[start:end] = True
                start = end

    # Offline reference extraction can reject sensor pulses that never persist
    # long enough to represent an actionable contact phase. Iterate because
    # removing a short on-run may create a short off-gap, and vice versa.
    for _ in range(8):
        previous = stable.copy()
        for part_idx in range(stable.shape[1]):
            values = stable[:, part_idx]
            # Fill release gaps first so one sustained contact is not split
            # into multiple short pulses and incorrectly removed below.
            start = 0
            while start < values.size:
                end = start + 1
                while end < values.size and values[end] == values[start]:
                    end += 1
                run_length = end - start
                if (
                    not values[start]
                    and min_off_frames > 0
                    and start > 0
                    and end < values.size
                    and run_length < min_off_frames
                ):
                    values[start:end] = True
                start = end
            start = 0
            while start < values.size:
                end = start + 1
                while end < values.size and values[end] == values[start]:
                    end += 1
                if values[start] and min_on_frames > 0 and end - start < min_on_frames:
                    values[start:end] = False
                start = end
        if np.array_equal(previous, stable):
            break
    return raw, stable


def _parse_part_body_names(value: str | None) -> dict[str, list[str]]:
    if value is None:
        return dict(PART_BODY_NAMES)
    parsed = json.loads(value)
    return {part: [str(name) for name in parsed[part]] for part in PART_ORDER}


def _label_env_id(label: str) -> int | None:
    match = ENV_RE.search(label)
    if match is None:
        return None
    return int(match.group(1))


def _label_part(label: str, env_id: int) -> str | None:
    label_env = _label_env_id(label)
    if label_env != env_id:
        return None
    for part, patterns in PART_LABEL_PATTERNS.items():
        if any(pattern in label for pattern in patterns):
            return part
    return None


def _raw_contact_part_positions(
    recording: dict[str, Any],
    metadata: dict[str, Any],
    order: np.ndarray,
    body_pos_w: np.ndarray,
    part_body_names: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray] | None:
    required = (
        "raw_contact_count",
        "raw_contact_body0",
        "raw_contact_body1",
        "raw_contact_point0_w",
        "raw_contact_point1_w",
    )
    if any(key not in recording for key in required):
        return None

    selected_env_id = int(metadata.get("selected_env_id", metadata.get("env_id", 0)))
    newton_body_labels = [str(x) for x in metadata.get("newton_body_labels", [])]
    if not newton_body_labels:
        return None

    body_names = [str(name) for name in metadata.get("body_names", [])]
    fallback_body_ids: dict[str, int | None] = {}
    for part in PART_ORDER:
        ids = [body_names.index(name) for name in part_body_names[part] if name in body_names]
        fallback_body_ids[part] = ids[0] if ids else None

    counts = np.asarray(recording["raw_contact_count"])[order].astype(np.int64)
    body0 = np.asarray(recording["raw_contact_body0"])[order].astype(np.int64)
    body1 = np.asarray(recording["raw_contact_body1"])[order].astype(np.int64)
    point0 = np.asarray(recording["raw_contact_point0_w"])[order].astype(np.float32)
    point1 = np.asarray(recording["raw_contact_point1_w"])[order].astype(np.float32)

    part_index = {part: idx for idx, part in enumerate(PART_ORDER)}
    positions = np.zeros((order.size, len(PART_ORDER), 3), dtype=np.float32)
    valid = np.zeros((order.size, len(PART_ORDER)), dtype=bool)

    for frame_idx in range(order.size):
        sums = np.zeros((len(PART_ORDER), 3), dtype=np.float64)
        weights = np.zeros(len(PART_ORDER), dtype=np.float64)
        n_contacts = max(0, min(int(counts[frame_idx]), body0.shape[1]))
        for contact_idx in range(n_contacts):
            b0 = int(body0[frame_idx, contact_idx])
            b1 = int(body1[frame_idx, contact_idx])
            label0 = newton_body_labels[b0] if 0 <= b0 < len(newton_body_labels) else ""
            label1 = newton_body_labels[b1] if 0 <= b1 < len(newton_body_labels) else ""
            part0 = _label_part(label0, selected_env_id) if label0 else None
            part1 = _label_part(label1, selected_env_id) if label1 else None

            if part0 is not None:
                idx = part_index[part0]
                sums[idx] += point1[frame_idx, contact_idx]
                weights[idx] += 1.0
            if part1 is not None:
                idx = part_index[part1]
                sums[idx] += point0[frame_idx, contact_idx]
                weights[idx] += 1.0

        nonzero = weights > 0.0
        positions[frame_idx, nonzero] = (sums[nonzero] / weights[nonzero, None]).astype(np.float32)
        valid[frame_idx, nonzero] = True

        for part, fallback_id in fallback_body_ids.items():
            idx = part_index[part]
            if valid[frame_idx, idx] or fallback_id is None:
                continue
            positions[frame_idx, idx] = body_pos_w[frame_idx, fallback_id]

    return positions, valid


def _raw_contacts(recording: dict[str, Any], metadata: dict[str, Any], order: np.ndarray) -> dict[str, np.ndarray]:
    raw_keys = (
        "raw_contact_count",
        "raw_contact_shape0",
        "raw_contact_shape1",
        "raw_contact_body0",
        "raw_contact_body1",
        "raw_contact_point0_w",
        "raw_contact_point1_w",
        "raw_contact_normal_w",
        "raw_contact_force_w",
    )
    if any(key not in recording for key in raw_keys):
        return {}

    selected_env_id = int(metadata.get("selected_env_id", metadata.get("env_id", 0)))
    newton_body_labels = [str(x) for x in metadata.get("newton_body_labels", [])]
    if not newton_body_labels:
        return {key: np.asarray(recording[key])[order] for key in raw_keys}

    counts = np.asarray(recording["raw_contact_count"])[order].astype(np.int32)
    shape0 = np.asarray(recording["raw_contact_shape0"])[order].astype(np.int32)
    shape1 = np.asarray(recording["raw_contact_shape1"])[order].astype(np.int32)
    body0 = np.asarray(recording["raw_contact_body0"])[order].astype(np.int32)
    body1 = np.asarray(recording["raw_contact_body1"])[order].astype(np.int32)
    point0_w = np.asarray(recording["raw_contact_point0_w"])[order].astype(np.float32)
    point1_w = np.asarray(recording["raw_contact_point1_w"])[order].astype(np.float32)
    normal_w = np.asarray(recording["raw_contact_normal_w"])[order].astype(np.float32)
    force_w = np.asarray(recording["raw_contact_force_w"])[order].astype(np.float32)

    out_count = np.zeros_like(counts)
    out_shape0 = np.full_like(shape0, -1)
    out_shape1 = np.full_like(shape1, -1)
    out_body0 = np.full_like(body0, -1)
    out_body1 = np.full_like(body1, -1)
    out_point0_w = np.zeros_like(point0_w)
    out_point1_w = np.zeros_like(point1_w)
    out_normal_w = np.zeros_like(normal_w)
    out_force_w = np.zeros_like(force_w)

    for frame_idx in range(order.size):
        write_idx = 0
        n_contacts = max(0, min(int(counts[frame_idx]), body0.shape[1]))
        for contact_idx in range(n_contacts):
            b0 = int(body0[frame_idx, contact_idx])
            b1 = int(body1[frame_idx, contact_idx])
            label0 = newton_body_labels[b0] if 0 <= b0 < len(newton_body_labels) else ""
            label1 = newton_body_labels[b1] if 0 <= b1 < len(newton_body_labels) else ""
            if _label_env_id(label0) != selected_env_id and _label_env_id(label1) != selected_env_id:
                continue
            out_shape0[frame_idx, write_idx] = shape0[frame_idx, contact_idx]
            out_shape1[frame_idx, write_idx] = shape1[frame_idx, contact_idx]
            out_body0[frame_idx, write_idx] = body0[frame_idx, contact_idx]
            out_body1[frame_idx, write_idx] = body1[frame_idx, contact_idx]
            out_point0_w[frame_idx, write_idx] = point0_w[frame_idx, contact_idx]
            out_point1_w[frame_idx, write_idx] = point1_w[frame_idx, contact_idx]
            out_normal_w[frame_idx, write_idx] = normal_w[frame_idx, contact_idx]
            out_force_w[frame_idx, write_idx] = force_w[frame_idx, contact_idx]
            write_idx += 1
        out_count[frame_idx] = write_idx

    max_selected_contacts = int(out_count.max(initial=0))
    width = max(max_selected_contacts, 1)

    return {
        "raw_contact_count": out_count,
        "raw_contact_shape0": out_shape0[:, :width],
        "raw_contact_shape1": out_shape1[:, :width],
        "raw_contact_body0": out_body0[:, :width],
        "raw_contact_body1": out_body1[:, :width],
        "raw_contact_point0_w": out_point0_w[:, :width],
        "raw_contact_point1_w": out_point1_w[:, :width],
        "raw_contact_normal_w": out_normal_w[:, :width],
        "raw_contact_force_w": out_force_w[:, :width],
        "raw_contact_max_count": np.asarray(max_selected_contacts, dtype=np.int32),
        "raw_contact_selected_env_id": np.asarray(selected_env_id, dtype=np.int32),
    }


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quat.shape}")
    return quat[..., [3, 0, 1, 2]]


def _require(recording: dict[str, Any], key: str, order: np.ndarray) -> np.ndarray:
    if key not in recording:
        raise ValueError(f"Recording is missing required key for rollout-ref extraction: {key}")
    return np.asarray(recording[key])[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, type=Path, help="Original motion npz, used only for frame-count metadata.")
    parser.add_argument("--recording", required=True, type=Path, help="Complete eval recording npz.")
    parser.add_argument("--output", required=True, type=Path, help="Output rollout-ref motion npz.")
    parser.add_argument("--threshold", type=float, default=10.0, help="Contact force mask threshold in Newtons.")
    parser.add_argument(
        "--mask-off-threshold",
        type=float,
        default=5.0,
        help="Contact mask release threshold in Newtons; does not alter force values.",
    )
    parser.add_argument(
        "--mask-close-gap-frames",
        type=int,
        default=2,
        help="Close internal false gaps up to this many 50 Hz frames.",
    )
    parser.add_argument(
        "--mask-min-on-frames",
        type=int,
        default=6,
        help="Remove contact pulses shorter than this many 50 Hz frames.",
    )
    parser.add_argument(
        "--mask-min-off-frames",
        type=int,
        default=6,
        help="Fill internal release gaps shorter than this many 50 Hz frames.",
    )
    parser.add_argument(
        "--mask-history-reduce",
        choices=("latest", "max"),
        default="max",
        help="Use the latest physics sample or the maximum-magnitude saved history sample for mask extraction.",
    )
    parser.add_argument("--force-reduce", choices=("sum", "max", "mean"), default="sum")
    parser.add_argument(
        "--forces-only",
        action="store_true",
        help="Write canonical rollout state and force channels without contact-point geometry.",
    )
    parser.add_argument("--part-body-names-json", default=None, help="Optional JSON map for contact-force part body names.")
    parser.add_argument("--allow-timeout", action="store_true", help="Allow timeout flags in the recording.")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to replace existing rollout output: {args.output}. Move it to the system trash first."
        )

    motion = _load_npz(args.motion)
    recording = _load_npz(args.recording)
    robot_asset = decode_robot_asset_json(motion.get("robot_asset_json"), context=f"motion {args.motion}")

    if "joint_pos" not in motion:
        raise ValueError(f"{args.motion} is missing joint_pos.")
    num_frames = int(np.asarray(motion["joint_pos"]).shape[0])
    recording, selected_env_id = _select_recording_env(recording, num_frames, args.allow_timeout)
    meta = _metadata(recording)
    solver_config = _validate_newton_recording_metadata(meta, args.recording)
    order = _validate_complete_rollout(recording, num_frames, args.allow_timeout)

    dof_names = [str(name) for name in meta.get("dof_names", [])]
    body_names = [str(name) for name in meta.get("body_names", [])]
    contact_sensor_body_names = [str(name) for name in meta.get("contact_sensor_body_names", [])]
    if not dof_names:
        raise ValueError("Recording metadata is missing dof_names.")
    if not body_names:
        raise ValueError("Recording metadata is missing body_names.")
    fps = float(meta.get("fps", motion.get("fps", 50)))
    kinematics_provenance = _rollout_kinematics_provenance(
        args.recording,
        fps=fps,
        body_names=body_names,
        dof_names=dof_names,
    )

    joint_pos = np.concatenate(
        [_require(recording, "root_pos", order), _xyzw_to_wxyz(_require(recording, "root_quat_xyzw", order)), _require(recording, "dof_pos", order)],
        axis=1,
    ).astype(np.float32)
    joint_vel = np.concatenate(
        [_require(recording, "root_lin_vel", order), _require(recording, "root_ang_vel", order), _require(recording, "dof_vel", order)],
        axis=1,
    ).astype(np.float32)

    body_pos_w = _require(recording, "body_pos_w", order).astype(np.float32)
    body_quat_w = _xyzw_to_wxyz(_require(recording, "body_quat_xyzw", order)).astype(np.float32)
    body_lin_vel_w = _require(recording, "body_lin_vel_w", order).astype(np.float32)
    body_ang_vel_w = _require(recording, "body_ang_vel_w", order).astype(np.float32)

    part_body_names = _parse_part_body_names(args.part_body_names_json)
    part_force = _part_forces(
        recording,
        body_names,
        part_body_names,
        args.force_reduce,
        contact_sensor_body_names,
    )[order]
    part_force_history = _part_force_history(
        recording,
        body_names,
        part_body_names,
        args.force_reduce,
        contact_sensor_body_names,
    )
    if part_force_history is not None:
        part_force_history = part_force_history[order]
        if not np.allclose(part_force_history[:, 0], part_force, rtol=1e-5, atol=1e-5):
            raise ValueError("Latest contact force does not match contact_force_part_history_w[:, 0].")
    if not np.isfinite(part_force).all():
        raise ValueError("Computed contact_force_part_w contains NaN or Inf.")
    mask_force = part_force
    if part_force_history is not None and args.mask_history_reduce == "max":
        history_magnitude = np.linalg.norm(part_force_history, axis=-1)
        history_index = history_magnitude.argmax(axis=1)
        mask_force = np.take_along_axis(
            part_force_history,
            history_index[:, None, :, None],
            axis=1,
        ).squeeze(1)
    raw_part_mask, part_mask = _stable_contact_mask(
        mask_force,
        on_threshold=float(args.threshold),
        off_threshold=float(args.mask_off_threshold),
        close_gap_frames=int(args.mask_close_gap_frames),
        min_on_frames=int(args.mask_min_on_frames),
        min_off_frames=int(args.mask_min_off_frames),
    )
    force_support_raw, force_support_mask = raw_part_mask, part_mask
    from somaforge_core.newton_contacts import extract_recording_contacts
    from somaforge_core.contact_schema import CONTACT_FORCE_PART_NAMES
    observed = extract_recording_contacts(recording, meta,
        int(selected_env_id if selected_env_id is not None else meta.get('env_id', 0)),
        parts=CONTACT_FORCE_PART_NAMES)
    if observed['unallocated'].any():
        raise ValueError('Unallocated active Newton constraints in recording')
    part_mask = observed['active'][order]
    raw_part_mask = part_mask.copy()
    part_positions = None if args.forces_only else (observed['position_w'][order], part_mask)

    out = {
        "fps": np.asarray(fps),
        "joint_names": np.asarray(dof_names),
        "body_names": np.asarray(body_names),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "contact_force_part_w": part_force,
        "force_support_mask": force_support_mask,
        "force_support_mask_raw": force_support_raw,
        "contact_part_mask": part_mask,
        "contact_semantics_json": np.asarray(json.dumps(meta['solver_contact_semantics'])),
        "contact_force_part_mask": part_mask,
        "contact_force_part_mask_raw": raw_part_mask,
        "contact_force_part_order": np.asarray(PART_ORDER),
        "contact_force_part_body_names_json": np.asarray(json.dumps(part_body_names)),
        "rollout_ref_source_recording": np.asarray(str(args.recording)),
        "rollout_ref_source_env_id": np.asarray(-1 if selected_env_id is None else selected_env_id),
        "rollout_ref_source_motion": np.asarray(str(args.motion)),
        "contact_force_demo_threshold": np.asarray(np.float32(args.threshold)),
        "contact_force_demo_off_threshold": np.asarray(np.float32(args.mask_off_threshold)),
        "contact_force_demo_close_gap_frames": np.asarray(np.int32(args.mask_close_gap_frames)),
        "contact_force_demo_min_on_frames": np.asarray(np.int32(args.mask_min_on_frames)),
        "contact_force_demo_min_off_frames": np.asarray(np.int32(args.mask_min_off_frames)),
        "contact_force_demo_mask_history_reduce": np.asarray(str(args.mask_history_reduce)),
        "contact_force_sample_semantics": np.asarray("latest_physics_step_time_aligned_with_raw_contacts"),
        "contact_mask_sample_semantics": np.asarray(
            "Newton solver activation and allocation; no force threshold or temporal label editing"
        ),
        "robot_asset_json": np.asarray(encode_robot_asset_json(robot_asset)),
        "kinematics_provenance_json": np.asarray(encode_kinematics_provenance(kinematics_provenance)),
    }
    contact_provenance = newton_contact_provenance(
        solver_config=solver_config,
        source_recording=str(args.recording),
        force_reduce=args.force_reduce,
        threshold_n=args.threshold,
        mask_off_threshold_n=args.mask_off_threshold,
        mask_close_gap_frames=args.mask_close_gap_frames,
        history_sample_count=(int(part_force_history.shape[1]) if part_force_history is not None else None),
    )
    contact_provenance.update(
        {
            "contact_mask_history_reduce": str(args.mask_history_reduce),
            "contact_mask_min_on_frames": int(args.mask_min_on_frames),
            "contact_mask_min_off_frames": int(args.mask_min_off_frames),
        }
    )
    out["contact_force_provenance_json"] = np.asarray(
        encode_contact_force_provenance(contact_provenance)
    )
    if part_force_history is not None:
        out["contact_force_part_history_w"] = part_force_history
        out["contact_force_part_history_latest_index"] = np.asarray(np.int32(0))
        out["contact_force_part_history_dt"] = np.asarray(np.float32(meta.get("sim_dt", 0.0)))
        out["contact_force_part_history_complete_control_interval"] = np.asarray(
            bool(part_force_history.shape[1] >= int(meta.get("control_decimation", 0)))
        )
    if part_positions is not None:
        contact_position_w, contact_position_valid = part_positions
        out["contact_force_part_position_w"] = contact_position_w
        out["contact_force_part_position_valid"] = contact_position_valid
        out["contact_force_part_position_source"] = np.asarray("newton_raw_rigid_contacts_latest_physics_step")
        out.update(_raw_contacts(recording, meta, order))
        if "raw_contact_count" in out:
            out["raw_contact_source"] = np.asarray("newton_raw_rigid_contacts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(
        f"Wrote {args.output} frames={num_frames} "
        f"joint_pos={joint_pos.shape} body_pos={body_pos_w.shape} "
        f"force_shape={part_force.shape} contact_ratio={part_mask.mean():.4f}"
    )


if __name__ == "__main__":
    main()
