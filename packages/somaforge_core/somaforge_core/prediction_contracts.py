from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryPrediction:
    """One forward result: public command and private constraint state."""

    keypoint_position: Tensor
    keypoint_rotation6d: Tensor
    contact_logits: Tensor
    auxiliary_qpos: Tensor



@dataclass(frozen=True)
class UnifiedInteractionPrediction:
    qpos: Tensor
    q_velocity: Tensor
    stationary_contact: Tensor
    keypoint_position: Tensor
    keypoint_rotation6d: Tensor
    contact_logits: Tensor
    active_contact: Tensor
    touchdown_logits: Tensor
    touchdown: Tensor
    persistent_logits: Tensor
    persistent_support: Tensor
    surface_logits: Tensor
    contact_surface: Tensor
    contact_local_offset: Tensor
    contact_position: Tensor
    contact_velocity: Tensor
    duration: Tensor

    def persistent_from(self, current_contact: Tensor) -> Tensor:
        return self.persistent_support & (current_contact > 0.5)

    def touchdown_from(self, current_contact: Tensor) -> Tensor:
        del current_contact
        return self.touchdown

    def liftoff_from(self, current_contact: Tensor) -> Tensor:
        return (current_contact > 0.5) & ~self.persistent_support

    def swing_from(self, current_contact: Tensor) -> Tensor:
        return (current_contact <= 0.5) & ~self.active_contact
