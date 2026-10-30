"""Autonomous next-interaction predictor without spatial binding inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from generator.conditioned_pose_predictor import ConditionedHeightmapPosePredictor, ConditionedPosePrediction, conditioned_pose_objective


@dataclass(frozen=True)
class FullInteractionPrediction:
    qpos: Tensor
    role_logits: Tensor
    surface_logits: Tensor
    conditioned_contact: Tensor
    conditioned_surface: Tensor
    teacher_conditioned: Tensor

    @property
    def role(self) -> Tensor:
        return self.role_logits.argmax(-1)

    @property
    def contact(self) -> Tensor:
        return self.role != 0

    @property
    def surface(self) -> Tensor:
        return self.surface_logits.argmax(-1)


class FullHeightmapInteractionPredictor(ConditionedHeightmapPosePredictor):
    """Jointly predict the next contact plan and its nominal endpoint pose.

    At inference the pose decoder consumes the hard plan emitted by these same
    role/surface heads.  Optional teacher conditioning exists only for the
    supervised curriculum and is reported per sample in the output.
    """

    def __init__(self, width: int = 192, layers: int = 3) -> None:
        super().__init__(width, layers)
        self.role_head = nn.Linear(width, 4)
        self.next_surface_head = nn.Linear(width, 2)

    def forward(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_anchor: Tensor,
        current_surface: Tensor,
        heightmap: Tensor,
        *,
        teacher_role: Tensor | None = None,
        teacher_surface: Tensor | None = None,
        teacher_mask: Tensor | None = None,
    ) -> FullInteractionPrediction:
        terrain, basis, yaw_quaternion, body, part = self._encode_observation(
            current_q, current_contact, current_anchor, current_surface, heightmap
        )
        role_logits = self.role_head(part)
        surface_logits = self.next_surface_head(part)
        predicted_role = role_logits.argmax(-1)
        predicted_contact = predicted_role != 0
        predicted_surface = surface_logits.argmax(-1)
        batch = len(current_q)
        if teacher_mask is None:
            teacher_mask = torch.zeros(batch, dtype=torch.bool, device=current_q.device)
        if teacher_mask.shape != (batch,) or teacher_mask.dtype != torch.bool:
            raise ValueError("teacher_mask must be bool [B]")
        if bool(teacher_mask.any()):
            if teacher_role is None or teacher_surface is None:
                raise ValueError("teacher-conditioned samples require role and surface labels")
            if teacher_role.shape != (batch, 6) or teacher_surface.shape != (batch, 6):
                raise ValueError("teacher role/surface must be [B,6]")
            teacher_contact = teacher_role != 0
            conditioned_contact = torch.where(
                teacher_mask[:, None], teacher_contact, predicted_contact
            )
            conditioned_surface = torch.where(
                teacher_mask[:, None], teacher_surface, predicted_surface
            )
        else:
            conditioned_contact = predicted_contact
            conditioned_surface = predicted_surface
        hard_plan = torch.stack((
            conditioned_contact & (conditioned_surface == 0),
            conditioned_contact & (conditioned_surface == 1),
            ~conditioned_contact,
        ), -1).to(role_logits)
        # Contact is the primary decision: geometry must make the pose obey
        # that decision, never move the categorical head toward an easier
        # contact.  The hard plan therefore conditions the pose decoder while
        # role/surface heads remain exclusively teacher-supervised.
        qpos = self._decode_with_plan_weights(
            current_q, terrain, basis, yaw_quaternion, body, part,
            hard_plan,
        )
        return FullInteractionPrediction(
            qpos=qpos,
            role_logits=role_logits,
            surface_logits=surface_logits,
            conditioned_contact=conditioned_contact,
            conditioned_surface=conditioned_surface,
            teacher_conditioned=teacher_mask,
        )


def full_interaction_objective(
    model: FullHeightmapInteractionPredictor,
    prediction: FullInteractionPrediction,
    target: dict[str, Tensor],
    scene: dict[str, Tensor],
    *,
    missing_pair_approach_weight: float = 0.10,
    query_override=None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Classify every plan, but regress pose only under a compatible plan."""

    role = F.cross_entropy(
        prediction.role_logits.transpose(1, 2), target["role"], reduction="none"
    ).mean(-1)
    active = target["role"] != 0
    count = active.sum(-1).clamp_min(1)
    surface = F.cross_entropy(
        prediction.surface_logits.transpose(1, 2),
        target["planned_surface"].clamp_min(0),
        reduction="none",
    )
    surface = (surface * active).sum(-1) / count
    predicted_plan_exact = (
        (prediction.contact == active).all(-1)
        & ((prediction.surface == target["planned_surface"]) | ~active).all(-1)
    )
    pose_allowed = prediction.teacher_conditioned | predicted_plan_exact
    _, pose_metrics = conditioned_pose_objective(
        model,
        ConditionedPosePrediction(prediction.qpos),
        target,
        scene,
        realization_contact=prediction.conditioned_contact,
        realization_surface=prediction.conditioned_surface,
        missing_pair_approach_weight=missing_pair_approach_weight,
        structure_joint_max_weight=1.0,  # Historical non-endpoint objective.
        query_override=query_override,
    )
    # A wrong autonomous plan has no compatible pose target.  In particular,
    # Newton realization/safety gradients from that plan must not rewrite the
    # demonstrated pose prior. Such samples train only the categorical heads;
    # pose supervision is supplied by a separate GT-conditioned pass.
    loss = (
        role
        + surface
        + pose_allowed * (
            pose_metrics["imitation_loss"]
            + missing_pair_approach_weight
            * pose_metrics["missing_pair_approach_loss"]
            + pose_metrics["own_plan_realization_loss"]
            + pose_metrics["safety_loss"]
        )
    )
    metrics = dict(pose_metrics)
    metrics.update(
        loss=loss,
        role_loss=role,
        surface_loss=surface,
        role_exact=(prediction.role == target["role"]).all(-1).float(),
        contact_exact=(prediction.contact == active).all(-1).float(),
        plan_exact=predicted_plan_exact.float(),
        pose_supervised=pose_allowed.float(),
        teacher_conditioned=prediction.teacher_conditioned.float(),
    )
    return loss, metrics


def interaction_plan_objective(
    prediction: FullInteractionPrediction,
    target: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Teacher-supervise only topology/surface classification.

    This objective is deliberately independent of qpos and therefore cannot
    use a wrong proposed plan to alter the pose decoder.
    """

    role = F.cross_entropy(
        prediction.role_logits.transpose(1, 2), target["role"], reduction="none"
    ).mean(-1)
    active = target["role"] != 0
    count = active.sum(-1).clamp_min(1)
    surface = F.cross_entropy(
        prediction.surface_logits.transpose(1, 2),
        target["planned_surface"].clamp_min(0),
        reduction="none",
    )
    surface = (surface * active).sum(-1) / count
    exact = (
        (prediction.role == target["role"]).all(-1)
        & ((prediction.surface == target["planned_surface"]) | ~active).all(-1)
    )
    return role + surface, {
        "role_loss": role,
        "surface_loss": surface,
        "role_exact": (prediction.role == target["role"]).all(-1).float(),
        "contact_exact": (prediction.contact == active).all(-1).float(),
        "plan_exact": exact.float(),
    }


__all__ = [
    "FullHeightmapInteractionPredictor",
    "FullInteractionPrediction",
    "full_interaction_objective",
    "interaction_plan_objective",
]
