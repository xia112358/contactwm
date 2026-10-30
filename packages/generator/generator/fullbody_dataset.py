"""Build full-body event segments directly from canonical edited motions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import decode_robot_asset_json

from somaforge_core.motion_contracts import BODY_NAMES
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def inherit_active_contact_descriptors(
    active: np.ndarray,
    surface: np.ndarray,
    uv: np.ndarray,
    previous_active: np.ndarray,
    previous_surface: np.ndarray,
    previous_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carry the last boundary anchor forward for an uninterrupted contact."""

    output_surface = np.asarray(surface, dtype=np.int64).copy()
    output_uv = np.asarray(uv, dtype=np.float32).copy()
    active = np.asarray(active, dtype=bool)
    previous_active = np.asarray(previous_active, dtype=bool)
    previous_surface = np.asarray(previous_surface, dtype=np.int64)
    previous_uv = np.asarray(previous_uv, dtype=np.float32)
    missing = active & (output_surface < 0)
    inherited = missing & previous_active & (previous_surface >= 0)
    output_surface[inherited] = previous_surface[inherited]
    output_uv[inherited] = previous_uv[inherited]
    return output_surface, output_uv, missing & ~inherited


def _collapse_contact(value: np.ndarray) -> np.ndarray:
    if value.shape[-1] == 6:
        return value.astype(bool)
    return np.stack(
        (
            value[..., 0] | value[..., 1],
            value[..., 2] | value[..., 3],
            value[..., 4],
            value[..., 5],
            value[..., 6],
            value[..., 7],
        ),
        axis=-1,
    )


def _debounce(value: np.ndarray, stable_frames: int = 3) -> np.ndarray:
    output = np.empty_like(value, dtype=bool)
    for channel in range(value.shape[1]):
        current = bool(value[0, channel])
        output[0, channel] = current
        candidate = None
        for frame in range(1, len(value)):
            state = bool(value[frame, channel])
            if state == current:
                if candidate is not None:
                    output[candidate:frame, channel] = current
                    candidate = None
                output[frame, channel] = current
                continue
            candidate = frame if candidate is None else candidate
            output[frame, channel] = current
            if frame - candidate + 1 >= stable_frames:
                output[candidate : frame + 1, channel] = state
                current = state
                candidate = None
        if candidate is not None:
            output[candidate:, channel] = current
    return output


def _touchdown_frames(contact: np.ndarray, merge_gap: int = 4) -> list[int]:
    from somaforge_core.newton_contact_data import touchdown_events
    return [frame for frame,_ in touchdown_events(contact, merge_gap=merge_gap)]


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    # Stable branch-free conversion for the yaw-localized floating root.
    result = np.empty(matrix.shape[:-2] + (4,), dtype=np.float64)
    flat_matrix = matrix.reshape(-1, 3, 3)
    flat_result = result.reshape(-1, 4)
    for index, value in enumerate(flat_matrix):
        trace = float(np.trace(value))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            flat_result[index] = (
                0.25 * scale,
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
                flat_result[index] = (
                    (value[2, 1] - value[1, 2]) / scale,
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                )
            elif axis == 1:
                scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
                flat_result[index] = (
                    (value[0, 2] - value[2, 0]) / scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                )
            else:
                scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
                flat_result[index] = (
                    (value[1, 0] - value[0, 1]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                )
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)
    return result.astype(np.float32)


def _phase_resample(value: np.ndarray, frames: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(value))
    target = np.linspace(0.0, 1.0, frames)
    flat = value.reshape(len(value), -1)
    output = np.stack([np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])], axis=-1)
    return output.reshape((frames,) + value.shape[1:]).astype(np.float32)


def _resample_rotation6d(value: np.ndarray, frames: int) -> np.ndarray:
    """Resample rotations on SO(3), rather than interpolating 6D columns."""

    matrix = value.reshape(value.shape[:-1] + (3, 2)).copy()
    first = matrix[..., 0]
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1.0e-12)
    second = matrix[..., 1] - np.sum(first * matrix[..., 1], axis=-1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1.0e-12)
    third = np.cross(first, second)
    quaternion = _matrix_quaternion_wxyz(np.stack((first, second, third), axis=-1))

    source = np.linspace(0.0, 1.0, len(value))
    target = np.linspace(0.0, 1.0, frames)
    indices = np.searchsorted(source, target, side="right").clip(1, len(source) - 1)
    left = quaternion[indices - 1].copy()
    right = quaternion[indices].copy()
    right[np.sum(left * right, axis=-1) < 0.0] *= -1.0
    alpha = (target - source[indices - 1]) / np.maximum(source[indices] - source[indices - 1], 1.0e-12)
    dot = np.sum(left * right, axis=-1).clip(-1.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    linear = sine < 1.0e-6
    safe_sine = np.where(linear, 1.0, sine)
    left_weight = np.where(
        linear,
        1.0 - alpha[..., None],
        np.sin((1.0 - alpha[..., None]) * angle) / safe_sine,
    )
    right_weight = np.where(
        linear,
        alpha[..., None],
        np.sin(alpha[..., None] * angle) / safe_sine,
    )
    output_quaternion = left_weight[..., None] * left + right_weight[..., None] * right
    output_quaternion /= np.maximum(np.linalg.norm(output_quaternion, axis=-1, keepdims=True), 1.0e-12)
    output_matrix = _quaternion_matrix_wxyz(output_quaternion)
    output = output_matrix[..., :2].reshape((frames,) + value.shape[1:])
    output[0] = value[0]
    output[-1] = value[-1]
    return output.astype(np.float32)


def _resample_qpos(value: np.ndarray, frames: int) -> np.ndarray:
    output = _phase_resample(value, frames)
    source = np.linspace(0.0, 1.0, len(value))
    target = np.linspace(0.0, 1.0, frames)
    indices = np.searchsorted(source, target, side="right").clip(1, len(source) - 1)
    first = value[indices - 1, 3:7].copy()
    second = value[indices, 3:7].copy()
    second[np.sum(first * second, axis=-1) < 0.0] *= -1.0
    alpha = ((target - source[indices - 1]) / np.maximum(source[indices] - source[indices - 1], 1.0e-12))[:, None]
    quaternion = (1.0 - alpha) * first + alpha * second
    output[:, 3:7] = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1.0e-12)
    output[0] = value[0]
    output[-1] = value[-1]
    return output


def _source_contact(plan: dict, frame_count: int) -> np.ndarray:
    from somaforge_core.newton_contact_data import load_contact_labels
    return load_contact_labels(plan['source_motion_path'], plan.get('newton_contact_file'),
                               frame_count=frame_count)['contact_part_mask']


def _local_box(plan: dict, origin: np.ndarray, world_to_local: np.ndarray) -> tuple[np.ndarray, ...]:
    records = [json.loads(line) for line in Path(plan["metadata"]["target_surface_catalog"]).read_text().splitlines()]
    from somaforge_core.contact_face_selection import ground_top_catalog
    selected = ground_top_catalog(records)
    ground, top = selected[0], selected[1]
    polygon = np.asarray(top["metadata"]["polygon_world"], dtype=np.float64)
    first_edge = polygon[1] - polygon[0]
    second_edge = polygon[2] - polygon[1]
    first_length = float(np.linalg.norm(first_edge))
    second_length = float(np.linalg.norm(second_edge))
    axis_x = first_edge / first_length
    axis_y = second_edge / second_length
    axis_z = np.asarray(top["normal"], dtype=np.float64)
    top_center = polygon.mean(axis=0)
    ground_z = float(ground["metadata"]["ground_z"])
    height = float(top_center[2] - ground_z)
    center_world = top_center - 0.5 * height * axis_z
    rotation_world = np.stack((axis_x, axis_y, axis_z), axis=-1)
    return (
        (world_to_local @ (center_world - origin)).astype(np.float32),
        (world_to_local @ rotation_world).astype(np.float32),
        np.asarray((0.5 * first_length, 0.5 * second_length, 0.5 * height), dtype=np.float32),
        np.asarray(ground_z - origin[2], dtype=np.float32),
    )


def condition_from_privileged_geometry(geometry: np.ndarray, duration: np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert the 30D exact terrain contract to the 14D OBB trajectory contract."""

    origin = geometry[:, :3]
    basis = geometry[:, 3:12].reshape(-1, 3, 3)
    polygon = geometry[:, 12:20].reshape(-1, 4, 2)
    height = geometry[:, 28]
    ground = geometry[:, 29]
    first_edge = polygon[:, 1] - polygon[:, 0]
    second_edge = polygon[:, 2] - polygon[:, 1]
    first_length = np.linalg.norm(first_edge, axis=-1)
    second_length = np.linalg.norm(second_edge, axis=-1)
    first_uv = first_edge / first_length[:, None]
    second_uv = second_edge / second_length[:, None]
    axis_x = np.einsum("bij,bj->bi", basis[:, :, :2], first_uv)
    axis_y = np.einsum("bij,bj->bi", basis[:, :, :2], second_uv)
    axis_z = basis[:, :, 2]
    top_center = origin + np.einsum("bij,bj->bi", basis[:, :, :2], polygon.mean(axis=1))
    center = top_center - 0.5 * height[:, None] * axis_z
    rotation = np.stack((axis_x, axis_y, axis_z), axis=-1).astype(np.float32)
    half_extents = np.stack((0.5 * first_length, 0.5 * second_length, 0.5 * height), axis=-1).astype(np.float32)
    condition = np.concatenate(
        (
            center,
            rotation[:, :, :2].reshape(-1, 6),
            half_extents,
            ground[:, None],
            duration[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    return (
        condition,
        center.astype(np.float32),
        rotation,
        half_extents,
        ground.astype(np.float32),
    )


@dataclass(frozen=True)
class FullBodyEventDataset:
    condition: torch.Tensor
    position: torch.Tensor
    rotation6d: torch.Tensor
    contact: torch.Tensor
    qpos: torch.Tensor
    q_valid: torch.Tensor
    box_center: torch.Tensor
    box_rotation: torch.Tensor
    box_half_extents: torch.Tensor
    ground_height: torch.Tensor
    motion_id: torch.Tensor
    event_index: torch.Tensor

    def __len__(self) -> int:
        return len(self.condition)


def build_fullbody_event_dataset(manifest_path: str | Path, phase_frames: int = 32) -> FullBodyEventDataset:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fk = CanonicalG1ForwardKinematics()
    rows: dict[str, list[np.ndarray | int]] = {
        name: []
        for name in (
            "condition",
            "position",
            "rotation6d",
            "contact",
            "qpos",
            "q_valid",
            "box_center",
            "box_rotation",
            "box_half_extents",
            "ground_height",
            "motion_id",
            "event_index",
        )
    }
    for motion_id, entry in enumerate(manifest["motion_files"]):
        plan = json.loads(Path(entry["edit_plan_file"]).read_text(encoding="utf-8"))
        with np.load(Path(entry["motion_file"]), allow_pickle=False) as loaded:
            qpos = np.asarray(loaded["joint_pos"], dtype=np.float32)
            joint_names = tuple(loaded["joint_names"].astype(str))
            asset_json = str(np.asarray(loaded["robot_asset_json"]).item())
            body_names = tuple(loaded["body_names"].astype(str))
            body_position = np.asarray(loaded["body_pos_w"], dtype=np.float32)
            body_quaternion = np.asarray(loaded["body_quat_w"], dtype=np.float32)
            fps = float(np.asarray(loaded["fps"]).item())
        if qpos.shape[1] != 36 or joint_names != G1_29DOF_JOINT_ORDER:
            raise ValueError(f"non-canonical qpos in {entry['motion_file']}")
        decode_robot_asset_json(asset_json, context=str(entry["motion_file"]))
        torso_index = body_names.index(BODY_NAMES[0])
        from somaforge_core.newton_contact_data import load_entry_contacts
        contact = load_entry_contacts(entry, len(qpos))['contact_part_mask']
        current = 0
        for event_index, target in enumerate(_touchdown_frames(contact)):
            if target - current < 5:
                continue
            origin = body_position[current, torso_index].astype(np.float64)
            torso_rotation = _quaternion_matrix_wxyz(body_quaternion[current, torso_index].astype(np.float64))
            yaw = math.atan2(float(torso_rotation[1, 0]), float(torso_rotation[0, 0]))
            cosine, sine = math.cos(yaw), math.sin(yaw)
            world_to_local = np.asarray(((cosine, sine, 0), (-sine, cosine, 0), (0, 0, 1)), dtype=np.float64)
            segment_qpos = qpos[current : target + 1].copy()
            segment_qpos[:, :3] = np.einsum("ij,tj->ti", world_to_local, segment_qpos[:, :3] - origin)
            root_rotation = _quaternion_matrix_wxyz(segment_qpos[:, 3:7].astype(np.float64))
            segment_qpos[:, 3:7] = _matrix_quaternion_wxyz(np.einsum("ij,tjk->tik", world_to_local, root_rotation))
            segment_qpos = _resample_qpos(segment_qpos, phase_frames)
            segment_contact = _phase_resample(contact[current : target + 1].astype(np.float32), phase_frames) >= 0.5
            with torch.no_grad():
                position, rotation = fk(torch.from_numpy(segment_qpos))
            center, box_rotation, half_extents, ground_height = _local_box(plan, origin, world_to_local)
            duration = float((target - current) / fps)
            condition = np.concatenate(
                (
                    center,
                    box_rotation[:, :2].reshape(-1),
                    half_extents,
                    np.atleast_1d(ground_height),
                    np.asarray((duration,), dtype=np.float32),
                )
            )
            rows["condition"].append(condition.astype(np.float32))
            rows["position"].append(position.numpy())
            rows["rotation6d"].append(rotation.numpy())
            rows["contact"].append(segment_contact.astype(np.float32))
            rows["qpos"].append(segment_qpos)
            rows["q_valid"].append(1)
            rows["box_center"].append(center)
            rows["box_rotation"].append(box_rotation)
            rows["box_half_extents"].append(half_extents)
            rows["ground_height"].append(ground_height)
            rows["motion_id"].append(motion_id)
            rows["event_index"].append(event_index)
            current = target
    if not rows["condition"]:
        raise RuntimeError("manifest produced no full-body touchdown segments")
    float_names = set(rows) - {"motion_id", "event_index", "q_valid"}
    tensors = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32 if name in float_names else np.int64))
        for name, value in rows.items()
    }
    return FullBodyEventDataset(**tensors)


def build_generated_fullbody_event_dataset(
    teacher_directory: str | Path, phase_frames: int = 32
) -> FullBodyEventDataset:
    """Load event segments from a generated teacher that already contains qpos."""

    directory = Path(teacher_directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    with np.load(directory / "teacher_events.npz", allow_pickle=False) as loaded:
        events = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows: dict[str, list[np.ndarray | int]] = {
        name: []
        for name in (
            "condition",
            "position",
            "rotation6d",
            "contact",
            "qpos",
            "q_valid",
            "box_center",
            "box_rotation",
            "box_half_extents",
            "ground_height",
            "motion_id",
            "event_index",
        )
    }
    for motion_id, report_row in enumerate(report["samples"]):
        with np.load(Path(report_row["motion_file"]), allow_pickle=False) as loaded:
            motion = {name: np.asarray(loaded[name]) for name in loaded.files}
        has_q = "joint_pos" in motion
        if has_q and tuple(motion["joint_names"].astype(str)) != G1_29DOF_JOINT_ORDER:
            raise ValueError(f"generated teacher {motion_id} has non-canonical joint order")
        decode_robot_asset_json(
            str(np.asarray(motion["robot_asset_json"]).item()),
            context=str(report_row["motion_file"]),
        )
        selected = np.flatnonzero(events["trajectory_id"].astype(np.int64) == motion_id)
        selected = selected[np.argsort(events["event_index"][selected])]
        boundary = motion["boundary_frames"].astype(np.int64)
        conditions = condition_from_privileged_geometry(
            events["geometry"][selected].astype(np.float32),
            events["duration"][selected].astype(np.float32),
        )
        condition, centers, rotations, half_extents, grounds = conditions
        torso_index = tuple(motion["body_names"].astype(str)).index(BODY_NAMES[0])
        for event_index, event_row in enumerate(selected):
            first, last = int(boundary[event_index]), int(boundary[event_index + 1])
            origin = motion["body_pos_w"][first, torso_index].astype(np.float64)
            torso_rotation = _quaternion_matrix_wxyz(motion["body_quat_w"][first, torso_index].astype(np.float64))
            yaw = math.atan2(float(torso_rotation[1, 0]), float(torso_rotation[0, 0]))
            cosine, sine = math.cos(yaw), math.sin(yaw)
            world_to_local = np.asarray(((cosine, sine, 0), (-sine, cosine, 0), (0, 0, 1)), dtype=np.float64)
            position = motion["body_pos_w"][first : last + 1].astype(np.float64)
            quaternion = motion["body_quat_w"][first : last + 1].astype(np.float64)
            rotation = _quaternion_matrix_wxyz(quaternion)
            local_position = np.einsum("ij,tbj->tbi", world_to_local, position - origin)
            local_rotation = np.einsum("ij,tbjk->tbik", world_to_local, rotation)
            if has_q:
                qpos = motion["joint_pos"][first : last + 1].astype(np.float32).copy()
                qpos[:, :3] = np.einsum("ij,tj->ti", world_to_local, qpos[:, :3] - origin)
                q_rotation = _quaternion_matrix_wxyz(qpos[:, 3:7].astype(np.float64))
                qpos[:, 3:7] = _matrix_quaternion_wxyz(np.einsum("ij,tjk->tik", world_to_local, q_rotation))
                resampled_qpos = _resample_qpos(qpos, phase_frames)
            else:
                resampled_qpos = np.zeros((phase_frames, 36), dtype=np.float32)
                resampled_qpos[:, 3] = 1.0
            rows["condition"].append(condition[event_index])
            rows["position"].append(_phase_resample(local_position, phase_frames))
            rows["rotation6d"].append(
                _resample_rotation6d(
                    local_rotation[..., :, :2].reshape(len(local_rotation), len(BODY_NAMES), 6), phase_frames
                )
            )
            from somaforge_core.newton_contact_data import load_contact_labels
            verified=load_contact_labels(directory/'teacher_motion.npz', report.get('newton_contact_file'))
            label_frames=np.rint(np.linspace(first,last,phase_frames)).astype(int)
            rows['contact'].append(verified['contact_part_mask'][label_frames])
            rows["qpos"].append(resampled_qpos)
            rows["q_valid"].append(int(has_q))
            rows["box_center"].append(centers[event_index])
            rows["box_rotation"].append(rotations[event_index])
            rows["box_half_extents"].append(half_extents[event_index])
            rows["ground_height"].append(grounds[event_index])
            rows["motion_id"].append(motion_id)
            rows["event_index"].append(int(events["event_index"][event_row]))
    float_names = set(rows) - {"motion_id", "event_index", "q_valid"}
    tensors = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32 if name in float_names else np.int64))
        for name, value in rows.items()
    }
    return FullBodyEventDataset(**tensors)
