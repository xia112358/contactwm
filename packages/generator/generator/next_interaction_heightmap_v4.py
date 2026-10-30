"""Contact-bound one-pass predictor from a root-yaw height map.

Each contactable body part selects one concrete spatial cell.  Contact
classification and whole-body pose decoding share that selection, so the pose
cannot independently reinterpret the terrain after predicting an intention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from generator.next_interaction import NextInteraction
from somaforge_core.heightmap import heightmap_grid
from somaforge_core.heightmap import _root_yaw_basis, render_root_yaw_box_heightmaps
from generator.next_interaction_heightmap_v3 import DenseHeightmapInteractionPredictor
from somaforge_core.g1_kinematics import _matrix_from_rotation6d, _rotation6d


@dataclass
class BoundNextInteraction(NextInteraction):
    binding_points_local: torch.Tensor
    binding_logits: torch.Tensor
    binding_confidence: torch.Tensor
    binding_grid_local: torch.Tensor


class BoundHeightmapInteractionPredictor(DenseHeightmapInteractionPredictor):
    """Predict a topology and a pose through the same per-part spatial binding."""

    def __init__(self, width: int = 192, layers: int = 3, *, joint_constraint_pose: bool = False):
        super().__init__(width, layers, couple_contact_pose=True)
        self.joint_constraint_pose = bool(joint_constraint_pose)
        grid = torch.from_numpy(heightmap_grid()[::2, ::2]).reshape(-1, 2)
        self.register_buffer("binding_xy", grid, persistent=False)
        self.binding_query = nn.Linear(width, width, bias=False)
        self.binding_key = nn.Linear(width, width, bias=False)
        self.binding_geometry = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.binding_norm = nn.LayerNorm(width)
        self.binding_scale = nn.Parameter(torch.zeros(()))
        self.part_pose_head = nn.Linear(width, 36, bias=False)
        nn.init.zeros_(self.part_pose_head.weight)
        if self.joint_constraint_pose:
            constraint_layer = nn.TransformerEncoderLayer(
                width, 6, 2 * width, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.constraint_decoder = nn.TransformerEncoder(
                constraint_layer, 2, enable_nested_tensor=False
            )
            self.constraint_pose_head = nn.Linear(width, 36, bias=False)
            nn.init.zeros_(self.constraint_pose_head.weight)

    @torch.no_grad()
    def initialize_binding_from_pose_attention(self) -> None:
        """Seed spatial binding from the warm-started terrain attention."""
        projection = self.pose_terrain.attention.in_proj_weight
        width = projection.shape[1]
        self.binding_query.weight.copy_(projection[:width])
        self.binding_key.weight.copy_(projection[width:2 * width])

    def set_binding_pretrain(self, enabled: bool) -> None:
        """Freeze the inherited predictor while learning spatial assignment."""
        self.binding_pretrain_active = bool(enabled)
        for parameter in self.parameters():
            parameter.requires_grad_(not enabled)
        if enabled:
            for module in (self.binding_query, self.binding_key):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    @torch.no_grad()
    def activate_binding(self, strength: float = 0.10) -> None:
        if not 0.0 < strength < 1.0:
            raise ValueError("binding strength must be in (0,1)")
        self.binding_scale.fill_(math.atanh(strength))

    def _spatial_binding(self, part: torch.Tensor, terrain: torch.Tensor, heightmap: torch.Tensor):
        logits = torch.einsum(
            "bpw,bsw->bps", self.binding_query(part), self.binding_key(terrain)
        ) / math.sqrt(part.shape[-1])
        soft = logits.softmax(-1)
        hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft)
        # Exact local selection in the forward pass, soft spatial gradient in backward.
        weight = hard + soft - soft.detach()
        z = heightmap[:, ::2, ::2].reshape(len(heightmap), -1, 1)
        xy = self.binding_xy.to(heightmap)[None].expand(len(heightmap), -1, -1)
        geometry = torch.cat((xy, z), -1)
        point = torch.einsum("bps,bsd->bpd", weight, geometry)
        feature = torch.einsum("bps,bsw->bpw", weight, terrain)
        return logits, point, feature, soft.amax(-1), geometry

    def forward(self, current_q, current_contact, current_anchor, current_surface, heightmap):
        terrain = self.height(heightmap)
        basis, yaw_quaternion = _root_yaw_basis(current_q)
        positions, rotations6d = self.fk(current_q[:, None])
        positions = positions[:, 0, 1:7]
        rotations = _matrix_from_rotation6d(rotations6d[:, 0, 1:7])
        local_position = self._to_local(positions, current_q, basis)
        local_rotation = basis[:, None].transpose(-1, -2) @ rotations
        local_anchor = self._to_local(current_anchor, current_q, basis)
        surface_index = torch.where(current_contact, current_surface.clamp(0, 1), 2)
        part = self.part_encoder(torch.cat((
            local_position,
            _rotation6d(local_rotation),
            local_anchor * current_contact[..., None],
            current_contact[..., None].float(),
        ), -1))
        part = part + self.part_identity.weight[None] + self.surface_embedding(surface_index)
        state = self.global_encoder(self._local_state(current_q, basis))[:, None]
        queries = torch.cat((state, part), 1)
        for block in self.terrain_reasoning:
            queries = block(queries, terrain)
        tokens = self.shared(queries)
        body, part = self.norm(tokens[:, :1]), self.norm(tokens[:, 1:])

        binding_logits, binding_point, binding_feature, binding_confidence, binding_grid = self._spatial_binding(
            part, terrain, heightmap
        )
        binding_update = self.binding_norm(binding_feature + self.binding_geometry(binding_point))
        binding_strength = self.binding_scale.tanh()
        # V4 keeps the residual experiment.  V5 predicts intent on the stable
        # inherited path and lets one global decoder reconcile all constraints.
        bound_part = part + binding_strength * binding_update
        categorical_part = part if self.joint_constraint_pose else bound_part
        role_logits = self.role_head(categorical_part)
        surface_logits = self.surface_head(categorical_part)
        role_probability, surface_probability = role_logits.softmax(-1), surface_logits.softmax(-1)
        active_probability = 1.0 - role_probability[..., :1]
        base_contact = part + self.role_embedding(role_probability)
        base_contact = base_contact + self.predicted_surface_embedding(surface_probability)
        interaction = self.pose_terrain(torch.cat((body, base_contact), 1), terrain)
        interaction = self.interaction_decoder(interaction)
        shared = self.norm(interaction[:, 0] + interaction[:, 1:].mean(1))
        raw = self.pose_head(shared)
        if self.joint_constraint_pose:
            constraint = (
                part + self.role_embedding(role_probability)
                + self.predicted_surface_embedding(surface_probability)
                + binding_update * active_probability
            )
            joint = self.constraint_decoder(torch.cat((shared[:, None], constraint), 1))
            # One coordinated 36D correction, never a sum of independent limb deltas.
            raw = raw + self.constraint_pose_head(self.norm(joint[:, 0]))
        else:
            constraint_delta = self.part_pose_head(self.norm(interaction[:, 1:]))
            raw = raw + binding_strength * (
                (constraint_delta * active_probability).sum(1)
                / active_probability.sum(1).clamp_min(1.0)
            )
        q = self._decode_pose(raw, current_q, basis, yaw_quaternion)
        duration = F.softplus(self.duration_head(shared)[:, 0])
        return BoundNextInteraction(
            q, role_logits, surface_logits, duration, binding_point, binding_logits,
            binding_confidence, binding_grid
        )


def objective(model, prediction, target, scene, *, binding_realization_weight: float = 0.25, **kwargs):
    """Teacher correctness plus predicted-intent realization and spatial binding."""
    from generator.next_interaction_surface import objective as newton_objective

    loss, metrics = newton_objective(
        model,
        prediction,
        target,
        scene,
        predicted_intent_mode="all",
        intent_consistency_weight=0.25,
        **kwargs,
    )
    basis, _ = _root_yaw_basis(target["current_q"])
    target_local = torch.einsum(
        "bij,bpj->bpi",
        basis.transpose(1, 2),
        target["anchor"] - target["current_q"][:, None, :3],
    )
    active = target["role"] != 0
    error = (prediction.binding_points_local - target_local).norm(dim=-1)
    grid_distance_sq = (
        prediction.binding_grid_local[:, None] - target_local[:, :, None]
    ).square().sum(-1)
    target_distribution = (-grid_distance_sq / (2 * 0.04 ** 2)).softmax(-1)
    binding_ce = -(target_distribution * prediction.binding_logits.log_softmax(-1)).sum(-1)
    binding_ce = (binding_ce * active).sum(-1) / active.sum(-1).clamp_min(1)
    target_cell = grid_distance_sq.argmin(-1)
    predicted_cell = prediction.binding_logits.argmax(-1)
    batch = torch.arange(len(predicted_cell), device=predicted_cell.device)[:, None]
    part = torch.arange(6, device=predicted_cell.device)[None]
    predicted_cell_error = (
        prediction.binding_grid_local[batch, predicted_cell]
        - prediction.binding_grid_local[batch, target_cell]
    ).norm(dim=-1)
    normalized = (error / 0.10).square() * active
    binding = normalized.sum(-1) / active.sum(-1).clamp_min(1) + 0.25 * normalized.amax(-1)
    # A predicted contact must be reachable by an actual collision-surface
    # sample of that body part.  This still has a useful gradient when Newton
    # has no narrow-phase candidate for a badly displaced initial proposal.
    binding_world = target["current_q"][:, None, :3] + torch.einsum(
        "bij,bpj->bpi", basis, prediction.binding_points_local
    )
    cloud, point_parts = model.geometry(model.fk, prediction.qpos[:, None])
    cloud = cloud[:, 0]
    realization = []
    for part_index in range(6):
        part_cloud = cloud[:, point_parts == part_index]
        realization.append((part_cloud - binding_world[:, part_index, None]).norm(dim=-1).amin(-1))
    realization = torch.stack(realization, -1)
    intended = prediction.contact.detach()
    # A diffuse/random binding must not destroy a good warm-started pose.  Its
    # own anchor supervision remains active; realization turns on as the
    # spatial distribution becomes confident.
    reliable = (
        torch.exp(-(error.detach() / 0.10).square())
        * active
        * intended
    )
    realization_normalized = (realization / 0.02).square() * reliable
    realization_loss = (
        realization_normalized.sum(-1) / reliable.sum(-1).clamp_min(1)
        + 0.25 * realization_normalized.amax(-1)
    )
    realization_weight = (
        0.0 if getattr(model, "binding_pretrain_active", False)
        else float(binding_realization_weight)
    )
    loss = loss + binding_ce + 0.05 * binding + realization_weight * realization_loss
    metrics.update(
        binding_contact_cm=100 * (error * active).sum(-1) / active.sum(-1).clamp_min(1),
        binding_contact_max_cm=100 * (error * active).amax(-1),
        binding_loss=binding,
        binding_cross_entropy=binding_ce,
        binding_cell_error_cm=100 * (predicted_cell_error * active).sum(-1) / active.sum(-1).clamp_min(1),
        binding_realization_cm=100 * (realization * intended).sum(-1) / intended.sum(-1).clamp_min(1),
        binding_realization_max_cm=100 * (realization * intended).amax(-1),
        binding_realization_loss=realization_loss,
        binding_realization_weight=prediction.qpos.new_full((len(prediction.qpos),), realization_weight),
        binding_reliable_parts=reliable.sum(-1),
        loss=loss,
    )
    return loss, metrics


class JointBoundHeightmapInteractionPredictor(BoundHeightmapInteractionPredictor):
    """Use one global decoder to reconcile all learned contact constraints."""

    def __init__(self, width: int = 192, layers: int = 3):
        super().__init__(width, layers, joint_constraint_pose=True)


def joint_objective(model, prediction, target, scene, **kwargs):
    # Newton teacher contact, predicted-intent consistency, full-body collision,
    # and pose supervision train the coordinated decoder.  Do not reintroduce
    # the failed independent point-pulling objective.
    return objective(
        model, prediction, target, scene,
        binding_realization_weight=0.0, **kwargs,
    )


__all__ = [
    "BoundHeightmapInteractionPredictor",
    "JointBoundHeightmapInteractionPredictor",
    "BoundNextInteraction",
    "joint_objective",
    "render_root_yaw_box_heightmaps",
]
