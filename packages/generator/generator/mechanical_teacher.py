"""Lift a sparse wide-start teacher onto canonical G1 joint trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import decode_robot_asset_json

from somaforge_core.motion_contracts import BODY_NAMES
from generator.fullbody_dataset import _matrix_quaternion_wxyz, _quaternion_matrix_wxyz, _resample_qpos, _source_contact, _touchdown_frames
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def _yaw(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _yaw_matrix(angle: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    zero = np.zeros_like(cosine)
    one = np.ones_like(cosine)
    return np.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), axis=-1).reshape(-1, 3, 3)


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(first, -1, 0)
    bw, bx, by, bz = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quaternion_power_wxyz(quaternion: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    value = quaternion.astype(np.float64).copy()
    if value[0] < 0.0:
        value *= -1.0
    value /= max(float(np.linalg.norm(value)), 1.0e-12)
    half_angle = math.acos(float(np.clip(value[0], -1.0, 1.0)))
    sine = math.sin(half_angle)
    if abs(sine) < 1.0e-10:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = value[1:] / sine
    scaled = exponent * half_angle
    return np.concatenate((np.cos(scaled)[:, None], np.sin(scaled)[:, None] * axis[None]), axis=-1)


def _placed_q_segment(
    fk: CanonicalG1ForwardKinematics,
    source_qpos: np.ndarray,
    desired_position: np.ndarray,
    desired_rotation: np.ndarray,
    desired_contact: np.ndarray,
    frames: int,
    previous_qpos: np.ndarray | None,
) -> np.ndarray:
    qpos = _resample_qpos(source_qpos, frames)
    phase = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    if previous_qpos is not None:
        delta = previous_qpos[7:] - qpos[0, 7:]
        qpos[:, 7:] += (1.0 - smooth_phase[:, None]) * delta

    with torch.no_grad():
        source_position, source_rotation6d = fk(torch.from_numpy(qpos.astype(np.float32)))
    source_position = source_position.numpy().astype(np.float64)
    source_rotation6d = source_rotation6d.numpy().reshape(frames, len(BODY_NAMES), 3, 2)
    source_torso_rotation = np.stack(
        (
            source_rotation6d[:, 0, :, 0],
            source_rotation6d[:, 0, :, 1],
            np.cross(source_rotation6d[:, 0, :, 0], source_rotation6d[:, 0, :, 1]),
        ),
        axis=-1,
    )
    source_yaw = np.unwrap(np.asarray([_yaw(value) for value in source_torso_rotation]))
    desired_yaw = np.unwrap(np.asarray([_yaw(value) for value in desired_rotation[:, 0]]))
    start_delta = desired_yaw[0] - source_yaw[0]
    end_delta = desired_yaw[-1] - source_yaw[-1]
    delta_change = math.atan2(math.sin(end_delta - start_delta), math.cos(end_delta - start_delta))
    yaw_delta = start_delta + smooth_phase * delta_change
    yaw_rotation = _yaw_matrix(yaw_delta)
    rotated_position = np.einsum("tij,tbj->tbi", yaw_rotation, source_position)

    # Sparse observations constrain only the two event boundaries.  Following
    # their interior frames injects contact-bit changes and sparse pose noise
    # directly into the floating root, which visibly jitters the whole robot.
    # Fit one support-aware rigid translation at each endpoint and interpolate
    # those two transforms with zero endpoint velocity instead.
    endpoint_translation = []
    for frame in (0, frames - 1):
        weights = np.full(len(BODY_NAMES), 0.05, dtype=np.float64)
        weights[0] = 0.5
        for part, body in enumerate((1, 2, 3, 4, 5, 6)):
            weights[body] += 5.0 * desired_contact[frame, part]
        endpoint_translation.append(
            np.sum(weights[:, None] * (desired_position[frame] - rotated_position[frame]), axis=0)
            / np.sum(weights)
        )
    translation = (
        (1.0 - smooth_phase[:, None]) * endpoint_translation[0]
        + smooth_phase[:, None] * endpoint_translation[1]
    )
    qpos[:, :3] = translation + np.einsum("tij,tj->ti", yaw_rotation, qpos[:, :3])
    root_rotation = _quaternion_matrix_wxyz(qpos[:, 3:7].astype(np.float64))
    qpos[:, 3:7] = _matrix_quaternion_wxyz(np.einsum("tij,tjk->tik", yaw_rotation, root_rotation))
    if previous_qpos is not None:
        current_start = qpos[0].copy()
        inverse_start = current_start[3:7].astype(np.float64).copy()
        inverse_start[1:] *= -1.0
        correction = _quaternion_multiply_wxyz(previous_qpos[3:7], inverse_start)
        correction_phase = _quaternion_power_wxyz(correction, 1.0 - smooth_phase)
        qpos[:, 3:7] = _quaternion_multiply_wxyz(correction_phase, qpos[:, 3:7])
        qpos[:, 3:7] /= np.maximum(np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True), 1.0e-12)
        qpos[:, :3] += (1.0 - smooth_phase[:, None]) * (previous_qpos[:3] - current_start[:3])
    return qpos.astype(np.float32)


def _local_keyframe(
    position: np.ndarray, rotation: np.ndarray, origin: np.ndarray, world_to_local: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local_position = np.einsum("ij,bj->bi", world_to_local, position - origin)
    local_rotation = np.einsum("ij,bjk->bik", world_to_local, rotation)
    return local_position.astype(np.float32), local_rotation[..., :2].reshape(len(BODY_NAMES), 6).astype(np.float32)


def _match_segment_boundary_velocities(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Make root/joint boundary velocities C1 without moving event keyframes."""

    if len(segments) < 2:
        return segments
    columns = np.asarray((*range(3), *range(7, 36)), dtype=np.int64)
    start_velocity = [segment[1, columns] - segment[0, columns] for segment in segments]
    end_velocity = [segment[-1, columns] - segment[-2, columns] for segment in segments]
    boundary_velocity = [start_velocity[0]]
    boundary_velocity.extend(
        0.5 * (end_velocity[index - 1] + start_velocity[index]) for index in range(1, len(segments))
    )
    boundary_velocity.append(end_velocity[-1])
    output = []
    for index, segment in enumerate(segments):
        frames = len(segment)
        phase = np.linspace(0.0, 1.0, frames, dtype=np.float64)
        h10 = phase**3 - 2.0 * phase**2 + phase
        h11 = phase**3 - phase**2
        start_delta = boundary_velocity[index] - start_velocity[index]
        end_delta = boundary_velocity[index + 1] - end_velocity[index]
        finite_difference_basis = np.asarray(
            (
                (h10[1] - h10[0], h11[1] - h11[0]),
                (h10[-1] - h10[-2], h11[-1] - h11[-2]),
            )
        )
        coefficients = np.linalg.solve(
            finite_difference_basis, np.stack((start_delta, end_delta), axis=0)
        )
        correction = h10[:, None] * coefficients[0] + h11[:, None] * coefficients[1]
        corrected = segment.copy()
        corrected[:, columns] += correction.astype(np.float32)
        output.append(corrected)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = json.loads((args.source / "report.json").read_text(encoding="utf-8"))
    nominal = report["nominal_motion"]
    nominal_path = Path(nominal["motion_file"])
    # Canonical motions contain legacy object-scalar metadata.  The path is
    # pinned by the teacher report and the decoded asset/joint order are
    # validated immediately below before any trajectory data are accepted.
    with np.load(nominal_path, allow_pickle=True) as loaded:
        source_motion = {name: np.asarray(loaded[name]) for name in loaded.files}
    decode_robot_asset_json(str(source_motion["robot_asset_json"].item()), context=str(nominal_path))
    if tuple(source_motion["joint_names"].astype(str)) != G1_29DOF_JOINT_ORDER:
        raise ValueError("nominal motion does not use canonical G1 joint order")
    plan = json.loads(Path(nominal["edit_plan_file"]).read_text(encoding="utf-8"))
    contact = _source_contact(plan, len(source_motion["joint_pos"]))
    raw_touchdowns = np.asarray(_touchdown_frames(contact), dtype=np.int64)
    first_top = int(report["first_top_contact_event"])
    fk = CanonicalG1ForwardKinematics()

    with np.load(args.source / "teacher_events.npz", allow_pickle=False) as loaded:
        events = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
    first_rows = np.flatnonzero(events["trajectory_id"].astype(np.int64) == 0)
    first_rows = first_rows[np.argsort(events["event_index"][first_rows])]
    first_approach_events = int(report["samples"][0]["approach_events"])
    gate_frame = int(report["gate_frame"])
    # The source event builder rejects touchdowns whose contact anchors cannot
    # be resolved.  Therefore raw touchdown indices are not source-event
    # indices.  Reconstruct the exact accepted nominal timeline used by the
    # wide teacher: two settling events, the last two accepted approach events
    # ending at the gate, then the saved (unscaled) climb-event durations.
    prefix = raw_touchdowns[:2]
    cycle = raw_touchdowns[raw_touchdowns <= gate_frame][-2:]
    source_boundaries_list = [0, *prefix.tolist(), *cycle.tolist()]
    if len(source_boundaries_list) != first_top + 1 or source_boundaries_list[-1] != gate_frame:
        raise ValueError("could not reconstruct the nominal accepted approach timeline")
    fps = float(np.asarray(source_motion["fps"]).item())
    for duration in events["duration"][first_rows[first_approach_events:]]:
        source_boundaries_list.append(source_boundaries_list[-1] + round(float(duration) * fps))
    source_boundaries = np.asarray(source_boundaries_list, dtype=np.int64)
    if source_boundaries[-1] >= len(source_motion["joint_pos"]):
        raise ValueError("reconstructed nominal event timeline exceeds the canonical motion")
    output_rows = []
    maximum_joint_seam = 0.0
    for trajectory_id, row in enumerate(report["samples"]):
        with np.load(Path(row["motion_file"]), allow_pickle=False) as loaded:
            sparse = {name: np.asarray(loaded[name]) for name in loaded.files}
        target_boundaries = sparse["boundary_frames"].astype(np.int64)
        approach_events = int(row["approach_events"])
        q_segments = []
        output_boundaries = [0]
        previous_qpos = None
        for event in range(len(target_boundaries) - 1):
            if event < approach_events:
                source_event = event if event < 2 else 2 + (event - 2) % 2
            else:
                source_event = first_top + event - approach_events
            source_first = int(source_boundaries[source_event])
            source_last = int(source_boundaries[source_event + 1])
            target_first = int(target_boundaries[event])
            target_last = int(target_boundaries[event + 1])
            frames = target_last - target_first + 1
            segment = _placed_q_segment(
                fk,
                source_motion["joint_pos"][source_first : source_last + 1].astype(np.float32),
                sparse["body_pos_w"][target_first : target_last + 1].astype(np.float64),
                _quaternion_matrix_wxyz(
                    sparse["body_quat_w"][target_first : target_last + 1].astype(np.float64)
                ),
                sparse["contact_force_part_mask"][target_first : target_last + 1].astype(np.float64),
                frames,
                previous_qpos,
            )
            if q_segments:
                maximum_joint_seam = max(
                    maximum_joint_seam, float(np.abs(segment[0, 7:] - q_segments[-1][-1, 7:]).max())
                )
            q_segments.append(segment)
            previous_qpos = segment[-1].copy()
            output_boundaries.append(output_boundaries[-1] + frames - 1)
        q_segments = _match_segment_boundary_velocities(q_segments)
        qpos = np.concatenate((q_segments[0], *(segment[1:] for segment in q_segments[1:])))
        with torch.no_grad():
            position, rotation6d = fk(torch.from_numpy(qpos))
        position = position.numpy()
        rotation6d = rotation6d.numpy()
        rotation = rotation6d.reshape(len(qpos), len(BODY_NAMES), 3, 2)
        rotation = np.stack((rotation[..., 0], rotation[..., 1], np.cross(rotation[..., 0], rotation[..., 1])), axis=-1)
        quaternion = _matrix_quaternion_wxyz(rotation)
        output_path = args.output / f"teacher_{trajectory_id:04d}.npz"
        np.savez_compressed(
            output_path,
            body_names=np.asarray(BODY_NAMES),
            body_pos_w=position.astype(np.float32),
            body_quat_w=quaternion,
            contact_force_part_mask=sparse["contact_force_part_mask"],
            contact_force_part_order=sparse["contact_force_part_order"],
            joint_pos=qpos,
            joint_names=np.asarray(G1_29DOF_JOINT_ORDER),
            fps=sparse["fps"],
            boundary_frames=np.asarray(output_boundaries, dtype=np.int64),
            robot_asset_json=source_motion["robot_asset_json"],
            generation_json=np.asarray(json.dumps({"schema": "climb00_canonical_q_template_transfer_v1"})),
        )
        selected = np.flatnonzero(events["trajectory_id"].astype(np.int64) == trajectory_id)
        selected = selected[np.argsort(events["event_index"][selected])]
        for event, event_row in enumerate(selected):
            first, last = output_boundaries[event], output_boundaries[event + 1]
            torso_rotation = rotation[first, 0]
            yaw = _yaw(torso_rotation)
            cosine, sine = math.cos(yaw), math.sin(yaw)
            world_to_local = np.asarray(((cosine, sine, 0), (-sine, cosine, 0), (0, 0, 1)))
            start_position, start_rotation = _local_keyframe(
                position[first], rotation[first], position[first, 0], world_to_local
            )
            end_position, end_rotation = _local_keyframe(
                position[last], rotation[last], position[first, 0], world_to_local
            )
            events["start_position"][event_row] = start_position
            events["start_rotation"][event_row] = start_rotation
            events["end_position"][event_row] = end_position
            events["end_rotation"][event_row] = end_rotation
        output_row = dict(row)
        output_row["motion_file"] = str(output_path.resolve())
        output_row["fullbody_qpos"] = True
        output_rows.append(output_row)

    np.savez_compressed(args.output / "teacher_events.npz", **events)
    output_report = dict(report)
    output_report["samples"] = output_rows
    output_report["mechanical_lift"] = {
        "schema": "climb00_canonical_q_template_transfer_v1",
        "nominal_motion": str(nominal_path.resolve()),
        "ik": False,
        "maximum_joint_c0_seam_rad": maximum_joint_seam,
    }
    (args.output / "report.json").write_text(json.dumps(output_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_report["mechanical_lift"]), flush=True)


if __name__ == "__main__":
    main()
