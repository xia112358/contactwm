"""Internal full-body representation for mechanically constrained trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import decode_robot_asset_json

from somaforge_core.motion_contracts import BODY_NAMES, CONTACT_PARTS


@dataclass(frozen=True)
class FullBodyTrajectory:
    """Private generator state: floating root plus canonical 29-DOF G1 pose."""

    qpos: NDArray[np.float32]
    fps: float
    robot_asset_json: str
    joint_names: tuple[str, ...] = G1_29DOF_JOINT_ORDER

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36:
            raise ValueError(f"full-body qpos must have shape [T,36], got {self.qpos.shape}")
        if self.qpos.shape[0] < 2:
            raise ValueError("full-body trajectory must contain at least two frames")
        if tuple(self.joint_names) != G1_29DOF_JOINT_ORDER:
            raise ValueError("full-body trajectory does not use canonical G1 joint order")
        if not np.isfinite(self.qpos).all():
            raise ValueError("full-body qpos contains non-finite values")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {self.fps}")
        quaternion_norm = np.linalg.norm(self.qpos[:, 3:7], axis=-1)
        if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4):
            raise ValueError("root quaternion must be normalized WXYZ")
        decode_robot_asset_json(self.robot_asset_json, context="full-body trajectory")


@dataclass(frozen=True)
class KeypointTrajectory:
    """Public trajectory output; no private full-body coordinates are exposed."""

    position_w: NDArray[np.float32]
    rotation6d_w: NDArray[np.float32]
    contact: NDArray[np.bool_]
    fps: float

    def __post_init__(self) -> None:
        frame_count = self.position_w.shape[0]
        if self.position_w.shape != (frame_count, len(BODY_NAMES), 3):
            raise ValueError(f"invalid keypoint position shape: {self.position_w.shape}")
        if self.rotation6d_w.shape != (frame_count, len(BODY_NAMES), 6):
            raise ValueError(f"invalid keypoint rotation shape: {self.rotation6d_w.shape}")
        if self.contact.shape != (frame_count, len(CONTACT_PARTS)):
            raise ValueError(f"invalid keypoint contact shape: {self.contact.shape}")


@dataclass(frozen=True)
class FullBodyCollisionAudit:
    """Collision measurements include original frames and swept subframes."""

    environment_penetration_m: NDArray[np.float32]
    planned_contact_penetration_m: NDArray[np.float32]
    self_penetration_m: NDArray[np.float32]
    sample_frame: NDArray[np.int64]
    sample_fraction: NDArray[np.float32]
    environment_bodies: tuple[tuple[str, ...], ...]
    self_body_pairs: tuple[tuple[tuple[str, str], ...], ...]

    @property
    def maximum_environment_penetration_m(self) -> float:
        return float(self.environment_penetration_m.max(initial=0.0))

    @property
    def maximum_self_penetration_m(self) -> float:
        return float(self.self_penetration_m.max(initial=0.0))

    @property
    def maximum_planned_contact_penetration_m(self) -> float:
        return float(self.planned_contact_penetration_m.max(initial=0.0))

    @property
    def collision_free(self) -> bool:
        return self.maximum_environment_penetration_m <= 0.0 and self.maximum_self_penetration_m <= 0.0


@dataclass(frozen=True)
class ConstrainedKeypointResult:
    """Public result from full-body generation and collision validation."""

    trajectory: KeypointTrajectory
    audit: FullBodyCollisionAudit


@dataclass(frozen=True)
class FullBodyTeacherSegment:
    """Private full-body supervision paired with the public contact sequence."""

    trajectory: FullBodyTrajectory
    contact: NDArray[np.bool_]

    def __post_init__(self) -> None:
        expected = (len(self.trajectory.qpos), len(CONTACT_PARTS))
        if self.contact.shape != expected:
            raise ValueError(
                "full-body teacher contact does not match the qpos trajectory: "
                f"expected {expected}, got {self.contact.shape}"
            )
