"""Canonical G1 FK and full-geometry collision gate for motion trajectories."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from somaforge_core.newton_collision import DirectNewtonCollisionScene
from numpy.typing import NDArray
from somaforge_core import CONTACT_BODY_NAMES_BY_PART, G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import canonical_g1_urdf_path

from somaforge_core.motion_contracts import BODY_NAMES, SparseKeyframe
from somaforge_core.motion_trajectory import ConstrainedKeypointResult, FullBodyCollisionAudit, FullBodyTrajectory, KeypointTrajectory

_CONTACT_BODY_NAMES = (
    CONTACT_BODY_NAMES_BY_PART["left_foot"],
    CONTACT_BODY_NAMES_BY_PART["right_foot"],
    CONTACT_BODY_NAMES_BY_PART["left_hand"],
    CONTACT_BODY_NAMES_BY_PART["right_hand"],
    CONTACT_BODY_NAMES_BY_PART["left_knee"],
    CONTACT_BODY_NAMES_BY_PART["right_knee"],
)


class MechanicalConstraintError(RuntimeError):
    """Raised when an internally generated motion is not executable by G1."""




def _rotation6d_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-12)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    matrix = np.stack(
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
    ).reshape(quat.shape[:-1] + (3, 3))
    return matrix[..., :, :2].reshape(quat.shape[:-1] + (6,)).astype(np.float32)


def _joint_limits() -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(canonical_g1_urdf_path()).getroot()  # noqa: S314 - canonical local asset
    joints = {joint.get("name", ""): joint for joint in root.findall("joint")}
    lower, upper = [], []
    for name in G1_29DOF_JOINT_ORDER:
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"canonical G1 URDF is missing joint {name}")
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"canonical G1 joint has no finite limits: {name}")
        lower.append(float(limit.get("lower", "nan")))
        upper.append(float(limit.get("upper", "nan")))
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def _interpolate_qpos(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    result = (1.0 - fraction) * first + fraction * second
    qa, qb = first[3:7], second[3:7]
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    quaternion = (1.0 - fraction) * qa + fraction * qb
    result[3:7] = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-12)
    return result.astype(np.float32)


class G1MechanicalProjector:
    """Project private G1 qpos to keypoints while auditing real collision geometry."""

    def __init__(
        self,
        terrain_mesh: str | Path | None,
        *,
        device: str = "cpu",
        swept_substeps: int = 2,
        ground_height_m: float = 0.0,
    ) -> None:
        if swept_substeps < 0:
            raise ValueError("swept_substeps must be non-negative")
        self.scene = DirectNewtonCollisionScene(
            terrain_mesh,
            device=device,
            ground_height_m=ground_height_m,
        )
        self.swept_substeps = int(swept_substeps)
        self.lower, self.upper = _joint_limits()

    def _validate_limits(self, trajectory: FullBodyTrajectory) -> None:
        joint = trajectory.qpos[:, 7:]
        below = self.lower[None] - joint
        above = joint - self.upper[None]
        violation = float(np.maximum(np.maximum(below, above), 0.0).max(initial=0.0))
        if violation > 1.0e-6:
            raise MechanicalConstraintError(f"G1 joint-limit violation: {violation:.6f} rad")

    def project(
        self,
        trajectory: FullBodyTrajectory,
        contact: NDArray[np.bool_],
    ) -> ConstrainedKeypointResult:
        self._validate_limits(trajectory)
        contacts = np.asarray(contact, dtype=bool)
        if contacts.shape != (len(trajectory.qpos), 6):
            raise ValueError(f"contact must have shape [T,6], got {contacts.shape}")

        positions, quaternions = [], []
        environment_depth, planned_contact_depth, self_depth = [], [], []
        sample_frame, sample_fraction = [], []
        environment_bodies: list[tuple[str, ...]] = []
        self_pairs: list[tuple[tuple[str, str], ...]] = []

        for frame in range(len(trajectory.qpos)):
            fractions = (
                (0.0,)
                if frame == len(trajectory.qpos) - 1
                else tuple(substep / (self.swept_substeps + 1) for substep in range(self.swept_substeps + 1))
            )
            for fraction in fractions:
                qpos = (
                    trajectory.qpos[frame]
                    if fraction == 0.0
                    else _interpolate_qpos(trajectory.qpos[frame], trajectory.qpos[frame + 1], fraction)
                )
                collision = self.scene.query_qpos(qpos)
                if fraction == 0.0:
                    body_position, body_quaternion = self.scene.body_poses(BODY_NAMES)
                    positions.append(body_position)
                    quaternions.append(body_quaternion)

                active_contact = contacts[frame]
                if fraction > 0.0:
                    active_contact = active_contact | contacts[frame + 1]
                allowed_contact_bodies = {
                    body_name
                    for part, is_active in enumerate(active_contact)
                    if is_active
                    for body_name in _CONTACT_BODY_NAMES[part]
                }

                first_robot = collision.body0 >= 0
                second_robot = collision.body1 >= 0
                penetration = np.maximum(-collision.geometry_distance_m, 0.0)
                environment_mask = first_robot ^ second_robot
                self_mask = first_robot & second_robot
                environment_indices = np.flatnonzero(environment_mask)
                environment_body_by_index = {
                    int(contact_index): body_name
                    for contact_index, body_name in zip(
                        collision.robot_body_indices,
                        collision.robot_body_names,
                        strict=True,
                    )
                }
                planned_mask = np.asarray(
                    [
                        index in environment_body_by_index
                        and environment_body_by_index[index] in allowed_contact_bodies
                        for index in range(len(penetration))
                    ],
                    dtype=bool,
                )
                disallowed_environment_mask = environment_mask & ~planned_mask
                environment_depth.append(float(penetration[disallowed_environment_mask].max(initial=0.0)))
                planned_contact_depth.append(float(penetration[planned_mask].max(initial=0.0)))
                self_depth.append(float(penetration[self_mask].max(initial=0.0)))
                environment_bodies.append(
                    tuple(
                        sorted(
                            {
                                environment_body_by_index[int(index)]
                                for index in environment_indices
                                if penetration[index] > 0.0 and not planned_mask[index]
                            }
                        )
                    )
                )
                pairs = {
                    tuple(sorted((self.scene.body_names[int(a)], self.scene.body_names[int(b)])))
                    for a, b, depth in zip(collision.body0, collision.body1, penetration, strict=True)
                    if a >= 0 and b >= 0 and depth > 0.0
                }
                self_pairs.append(tuple(sorted(pairs)))
                sample_frame.append(frame)
                sample_fraction.append(float(fraction))

        keypoints = KeypointTrajectory(
            position_w=np.asarray(positions, dtype=np.float32),
            rotation6d_w=_rotation6d_wxyz(np.asarray(quaternions, dtype=np.float32)),
            contact=contacts,
            fps=trajectory.fps,
        )
        audit = FullBodyCollisionAudit(
            environment_penetration_m=np.asarray(environment_depth, dtype=np.float32),
            planned_contact_penetration_m=np.asarray(planned_contact_depth, dtype=np.float32),
            self_penetration_m=np.asarray(self_depth, dtype=np.float32),
            sample_frame=np.asarray(sample_frame, dtype=np.int64),
            sample_fraction=np.asarray(sample_fraction, dtype=np.float32),
            environment_bodies=tuple(environment_bodies),
            self_body_pairs=tuple(self_pairs),
        )
        return ConstrainedKeypointResult(keypoints, audit)
