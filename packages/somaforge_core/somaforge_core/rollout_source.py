from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from somaforge_core.contact_schema import (
    CONTACT_FORCE_PART_BODY_NAMES,
    CONTACT_FORCE_PART_ORDER,
)


_ENV_RE = re.compile(r"/envs/env_(\d+)/", re.IGNORECASE)


def stable_contact_masks(
    force: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    close_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and hysteresis-stabilized masks for ``[T,P,3]`` force."""

    if not 0.0 <= off_threshold <= on_threshold:
        raise ValueError("contact thresholds must satisfy 0 <= off <= on")
    if close_gap_frames < 0:
        raise ValueError("close_gap_frames must be nonnegative")
    values = np.asarray(force)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("force must have shape [T,P,3]")
    magnitude = np.linalg.norm(values, axis=-1)
    raw = magnitude > on_threshold
    stable = np.zeros_like(raw)
    active = np.zeros(raw.shape[1], dtype=bool)
    for frame in range(raw.shape[0]):
        active = np.where(active, magnitude[frame] > off_threshold, raw[frame])
        stable[frame] = active
    if close_gap_frames:
        for part in range(stable.shape[1]):
            part_mask = stable[:, part]
            start = 0
            while start < part_mask.size:
                end = start + 1
                while end < part_mask.size and part_mask[end] == part_mask[start]:
                    end += 1
                if (
                    not part_mask[start]
                    and start > 0
                    and end < part_mask.size
                    and end - start <= close_gap_frames
                ):
                    part_mask[start:end] = True
                start = end
    return raw, stable


def reduce_sensor_force_parts(
    sensor_force: np.ndarray,
    sensor_body_names: Sequence[str],
) -> np.ndarray:
    """Reduce sensor-force samples to the canonical eight parts."""

    values = np.asarray(sensor_force)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("sensor force must end in [body,xyz]")
    names = tuple(str(name) for name in sensor_body_names)
    if values.shape[-2] != len(names):
        raise ValueError(
            "sensor force body axis does not match sensor_body_names"
        )
    name_to_index = {name: index for index, name in enumerate(names)}
    output = np.zeros(
        (*values.shape[:-2], len(CONTACT_FORCE_PART_ORDER), 3),
        dtype=values.dtype,
    )
    for part_index, part in enumerate(CONTACT_FORCE_PART_ORDER):
        indices = [
            name_to_index[name]
            for name in CONTACT_FORCE_PART_BODY_NAMES[part]
            if name in name_to_index
        ]
        if indices:
            output[..., part_index, :] = values[..., indices, :].sum(axis=-2)
    return output


def complete_rollout_env_ids(
    motion_time_step: np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    values = np.asarray(motion_time_step)
    if values.ndim != 2:
        raise ValueError("motion_time_step must have shape [T,E]")
    expected = np.arange(frame_count, dtype=values.dtype)
    return np.asarray(
        [
            env_id
            for env_id in range(values.shape[1])
            if values.shape[0] >= frame_count
            and np.array_equal(values[:frame_count, env_id], expected)
        ],
        dtype=np.int32,
    )


def merge_qpos(
    *,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    root = np.asarray(root_pos, dtype=np.float64)
    quaternion = np.asarray(root_quat_xyzw, dtype=np.float64)
    joints = np.asarray(dof_pos, dtype=np.float64)
    if root.ndim != 3 or root.shape[-1] != 3:
        raise ValueError("root_pos must have shape [T,E,3]")
    if quaternion.shape != (*root.shape[:2], 4):
        raise ValueError("root_quat_xyzw must have shape [T,E,4]")
    if joints.ndim != 3 or joints.shape[:2] != root.shape[:2]:
        raise ValueError("dof_pos must have shape [T,E,J]")
    reference = quaternion[:, :1]
    quaternion = np.where(
        np.sum(quaternion * reference, axis=-1, keepdims=True) < 0.0,
        -quaternion,
        quaternion,
    )
    quaternion = np.median(quaternion, axis=1)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True),
        1.0e-12,
    )
    output = np.zeros((root.shape[0], 7 + joints.shape[-1]), dtype=np.float32)
    output[:, :3] = np.median(root, axis=1)
    output[:, 3:7] = quaternion[:, [3, 0, 1, 2]]
    output[:, 7:] = np.median(joints, axis=1)
    return output


def merge_part_force(
    sensor_force: np.ndarray,
    sensor_body_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_env = reduce_sensor_force_parts(sensor_force, sensor_body_names)
    force = np.median(per_env, axis=1).astype(np.float32)
    raw_mask, stable_mask = stable_contact_masks(
        force,
        on_threshold=10.0,
        off_threshold=5.0,
        close_gap_frames=2,
    )
    return force, raw_mask, stable_mask


def merge_raw_contacts(
    recording: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    frame_count: int,
    env_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Fuse external raw contacts by stable body/shape labels.

    Robot contacts from every selected environment are rewritten into a
    synthetic single-scene namespace. One robust median contact is retained per
    robot-body/shape and counterpart-shape pair per frame.
    """

    body_labels = [str(value) for value in metadata["newton_body_labels"]]
    shape_labels = [str(value) for value in metadata["newton_shape_labels"]]
    selected = {int(value) for value in env_ids}
    counts = np.asarray(recording["raw_contact_count"], dtype=np.int64)
    arrays = {
        name: np.asarray(recording[name])
        for name in (
            "raw_contact_shape0",
            "raw_contact_shape1",
            "raw_contact_body0",
            "raw_contact_body1",
            "raw_contact_point0_w",
            "raw_contact_point1_w",
            "raw_contact_normal_w",
            "raw_contact_force_w",
        )
    }
    canonical_body_ids = _canonical_label_ids(body_labels, selected)
    canonical_shape_ids = _canonical_label_ids(shape_labels, selected)
    frames: list[list[dict[str, Any]]] = []
    for frame in range(frame_count):
        groups: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
        count = max(0, min(int(counts[frame]), arrays["raw_contact_body0"].shape[1]))
        for index in range(count):
            body0 = int(arrays["raw_contact_body0"][frame, index])
            body1 = int(arrays["raw_contact_body1"][frame, index])
            label0 = body_labels[body0] if 0 <= body0 < len(body_labels) else ""
            label1 = body_labels[body1] if 0 <= body1 < len(body_labels) else ""
            env0 = _label_env_id(label0)
            env1 = _label_env_id(label1)
            robot_is_0 = env0 in selected and env1 is None
            robot_is_1 = env1 in selected and env0 is None
            if robot_is_0 == robot_is_1:
                continue
            if robot_is_0:
                robot_body = _stable_label(label0)
                robot_shape = _shape_label(
                    arrays["raw_contact_shape0"][frame, index], shape_labels
                )
                counterpart_shape = _shape_label(
                    arrays["raw_contact_shape1"][frame, index], shape_labels
                )
                point_robot = arrays["raw_contact_point0_w"][frame, index]
                point_counterpart = arrays["raw_contact_point1_w"][frame, index]
                normal = arrays["raw_contact_normal_w"][frame, index]
                force = arrays["raw_contact_force_w"][frame, index]
            else:
                robot_body = _stable_label(label1)
                robot_shape = _shape_label(
                    arrays["raw_contact_shape1"][frame, index], shape_labels
                )
                counterpart_shape = _shape_label(
                    arrays["raw_contact_shape0"][frame, index], shape_labels
                )
                point_robot = arrays["raw_contact_point1_w"][frame, index]
                point_counterpart = arrays["raw_contact_point0_w"][frame, index]
                normal = -arrays["raw_contact_normal_w"][frame, index]
                force = -arrays["raw_contact_force_w"][frame, index]
            if not robot_body or not robot_shape:
                continue
            counterpart_shape = counterpart_shape or "/World/static_environment"
            groups[(robot_body, robot_shape, counterpart_shape)].append(
                {
                    "point_robot": np.asarray(point_robot, dtype=np.float64),
                    "point_counterpart": np.asarray(
                        point_counterpart, dtype=np.float64
                    ),
                    "normal": np.asarray(normal, dtype=np.float64),
                    "force": np.asarray(force, dtype=np.float64),
                }
            )
        fused: list[dict[str, Any]] = []
        for (robot_body, robot_shape, counterpart_shape), samples in sorted(groups.items()):
            normal = np.median(
                np.stack([sample["normal"] for sample in samples]), axis=0
            )
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            fused.append(
                {
                    "robot_body": robot_body,
                    "robot_shape": robot_shape,
                    "counterpart_shape": counterpart_shape,
                    "point_robot": np.median(
                        np.stack([sample["point_robot"] for sample in samples]),
                        axis=0,
                    ),
                    "point_counterpart": np.median(
                        np.stack(
                            [sample["point_counterpart"] for sample in samples]
                        ),
                        axis=0,
                    ),
                    "normal": normal,
                    "force": np.median(
                        np.stack([sample["force"] for sample in samples]), axis=0
                    ),
                }
            )
        frames.append(fused)

    used_bodies = {contact["robot_body"] for frame in frames for contact in frame}
    used_shapes = {
        label
        for frame in frames
        for contact in frame
        for label in (contact["robot_shape"], contact["counterpart_shape"])
    }
    missing_bodies = sorted(used_bodies - canonical_body_ids.keys())
    missing_shapes = sorted(used_shapes - canonical_shape_ids.keys())
    if missing_bodies or missing_shapes:
        raise ValueError(
            "could not preserve stable Newton IDs for consensus contacts: "
            f"bodies={missing_bodies}, shapes={missing_shapes}"
        )
    stable_bodies = np.full(len(body_labels), "", dtype=object)
    stable_shapes = np.full(len(shape_labels), "", dtype=object)
    for label in used_bodies:
        stable_bodies[canonical_body_ids[label]] = label
    for label in used_shapes:
        stable_shapes[canonical_shape_ids[label]] = label
    width = max(max((len(frame) for frame in frames), default=0), 1)
    output = {
        "raw_contact_count": np.asarray(
            [len(frame) for frame in frames], dtype=np.int32
        ),
        "raw_contact_shape0": np.full((frame_count, width), -1, dtype=np.int32),
        "raw_contact_shape1": np.full((frame_count, width), -1, dtype=np.int32),
        "raw_contact_body0": np.full((frame_count, width), -1, dtype=np.int32),
        "raw_contact_body1": np.full((frame_count, width), -1, dtype=np.int32),
        "raw_contact_point0_w": np.zeros(
            (frame_count, width, 3), dtype=np.float32
        ),
        "raw_contact_point1_w": np.zeros(
            (frame_count, width, 3), dtype=np.float32
        ),
        "raw_contact_normal_w": np.zeros(
            (frame_count, width, 3), dtype=np.float32
        ),
        "raw_contact_force_w": np.zeros(
            (frame_count, width, 3), dtype=np.float32
        ),
        "raw_contact_max_count": np.asarray(
            max((len(frame) for frame in frames), default=0), dtype=np.int32
        ),
        "raw_contact_source": np.asarray(
            "multi_rollout_stable_label_median_external_contacts"
        ),
        "raw_contact_point_pairing": np.asarray(
            "body0_point0_body1_point1"
        ),
        "newton_body_labels": stable_bodies.astype(str),
        "newton_shape_labels": stable_shapes.astype(str),
    }
    for frame_index, frame in enumerate(frames):
        for contact_index, contact in enumerate(frame):
            output["raw_contact_body0"][frame_index, contact_index] = canonical_body_ids[
                contact["robot_body"]
            ]
            output["raw_contact_shape0"][frame_index, contact_index] = canonical_shape_ids[
                contact["robot_shape"]
            ]
            output["raw_contact_shape1"][frame_index, contact_index] = canonical_shape_ids[
                contact["counterpart_shape"]
            ]
            output["raw_contact_point0_w"][frame_index, contact_index] = contact[
                "point_robot"
            ]
            output["raw_contact_point1_w"][frame_index, contact_index] = contact[
                "point_counterpart"
            ]
            output["raw_contact_normal_w"][frame_index, contact_index] = contact[
                "normal"
            ]
            output["raw_contact_force_w"][frame_index, contact_index] = contact[
                "force"
            ]
    return output


def _label_env_id(label: str) -> int | None:
    match = _ENV_RE.search(str(label))
    return None if match is None else int(match.group(1))


def _leaf(label: str) -> str:
    return str(label).rstrip("/").split("/")[-1]


def _stable_label(label: str) -> str:
    return _ENV_RE.sub("/", str(label))


def _canonical_label_ids(
    labels: Sequence[str],
    selected_env_ids: set[int],
) -> dict[str, int]:
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, raw_label in enumerate(labels):
        label = str(raw_label)
        env_id = _label_env_id(label)
        if env_id is not None and env_id not in selected_env_ids:
            continue
        stable = _stable_label(label)
        candidates[stable].append((-1 if env_id is None else env_id, index))
    return {
        label: min(values)[1]
        for label, values in candidates.items()
        if label
    }


def _shape_label(value: Any, labels: Sequence[str]) -> str:
    shape_id = int(value)
    if not 0 <= shape_id < len(labels):
        return ""
    return _stable_label(labels[shape_id])


def source_metadata_json(
    *,
    source_recording: str,
    checkpoint: str,
    env_count: int,
) -> np.ndarray:
    return np.asarray(
        json.dumps(
            {
                "schema": "somaforge_multi_rollout_consensus_v1",
                "source_recording": str(source_recording),
                "checkpoint": str(checkpoint),
                "successful_rollout_count": int(env_count),
                "time_alignment": "motion_time_step_exact",
                "kinematic_aggregation": "coordinatewise_median",
                "quaternion_aggregation": (
                    "hemisphere_aligned_component_median_then_normalize"
                ),
                "force_aggregation": "world_vector_coordinatewise_median",
                "raw_contact_aggregation": (
                    "stable_label_external_contact_group_median"
                ),
                "runtime_env_identity_persisted": False,
                "dynamic_replay_used": False,
            },
            sort_keys=True,
        )
    )


__all__ = [
    "complete_rollout_env_ids",
    "merge_part_force",
    "merge_qpos",
    "merge_raw_contacts",
    "source_metadata_json",
]
