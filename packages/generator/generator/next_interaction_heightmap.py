"""Dense egocentric height-map next-interaction predictor.

The height map is an observation.  It never defines contact truth: role,
surface, realization, and penetration supervision remain current Newton data.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from generator.next_interaction import NextInteraction
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d, _quaternion_matrix_wxyz


from somaforge_core.heightmap import HEIGHTMAP_FORWARD_MIN_M
from somaforge_core.heightmap import HEIGHTMAP_FORWARD_MAX_M
from somaforge_core.heightmap import HEIGHTMAP_LATERAL_MIN_M
from somaforge_core.heightmap import HEIGHTMAP_LATERAL_MAX_M
from somaforge_core.heightmap import HEIGHTMAP_RESOLUTION_M
from somaforge_core.heightmap import HEIGHTMAP_ROWS
from somaforge_core.heightmap import HEIGHTMAP_COLS
from somaforge_core.heightmap import HEIGHTMAP_CLIP_M


from somaforge_core.heightmap import heightmap_grid


from somaforge_core.heightmap import render_box_heightmaps


class HeightmapEncoder(nn.Module):
    """Preserve spatial terrain tokens while reducing 4331 raw samples."""

    def __init__(self, width: int):
        super().__init__()
        grid = torch.from_numpy(heightmap_grid()).permute(2, 0, 1)
        grid[0] = 2 * (grid[0] - HEIGHTMAP_FORWARD_MIN_M) / (
            HEIGHTMAP_FORWARD_MAX_M - HEIGHTMAP_FORWARD_MIN_M
        ) - 1
        grid[1] = 2 * (grid[1] - HEIGHTMAP_LATERAL_MIN_M) / (
            HEIGHTMAP_LATERAL_MAX_M - HEIGHTMAP_LATERAL_MIN_M
        ) - 1
        self.register_buffer("coordinates", grid, persistent=False)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, heightmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if heightmap.shape[1:] != (HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
            raise ValueError(
                f"heightmap must be [B,{HEIGHTMAP_ROWS},{HEIGHTMAP_COLS}], got {tuple(heightmap.shape)}"
            )
        coordinates = self.coordinates.to(heightmap).expand(len(heightmap), -1, -1, -1)
        feature = self.conv(torch.cat((heightmap[:, None] / HEIGHTMAP_CLIP_M, coordinates), 1))
        tokens = self.norm(feature.flatten(2).transpose(1, 2))
        return tokens, feature


class HeightmapInteractionPredictor(nn.Module):
    """Contacts first, then whole-body pose, with learned terrain feedback."""

    def __init__(self, width: int = 192, layers: int = 3, refinements: int = 2):
        super().__init__()
        self.fk = CanonicalG1ForwardKinematics()
        self.geometry = CanonicalG1CollisionPoints(64)
        self.height = HeightmapEncoder(width)
        self.global_encoder = nn.Linear(38, width)
        self.part_encoder = nn.Linear(13, width)
        self.surface_embedding = nn.Embedding(3, width)
        self.part_identity = nn.Embedding(6, width)
        layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.shared = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.role_head = nn.Linear(width, 4)
        self.surface_head = nn.Linear(width, 2)
        self.role_embedding = nn.Linear(4, width, bias=False)
        self.predicted_surface_embedding = nn.Linear(2, width, bias=False)
        interaction_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.interaction_decoder = nn.TransformerEncoder(
            interaction_layer, 1, enable_nested_tensor=False
        )
        self.pose_head = nn.Linear(width, 36)
        self.duration_head = nn.Linear(width, 1)
        self.refinements = int(refinements)
        self.refine_state = nn.Linear(38, width)
        self.refine_part = nn.Linear(17, width)
        refine_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.refine_decoder = nn.TransformerEncoder(refine_layer, 1, enable_nested_tensor=False)
        self.refine_head = nn.Linear(width, 36)
        nn.init.normal_(self.pose_head.weight, std=0.001)
        nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.refine_head.weight)
        nn.init.zeros_(self.refine_head.bias)
        with torch.no_grad():
            self.pose_head.bias[3] = 1.0

    @staticmethod
    def _state(q: torch.Tensor) -> torch.Tensor:
        rotation = _quaternion_matrix_wxyz(q[:, 3:7])
        return torch.cat((q[:, :3], rotation[:, :, :2].flatten(1), q[:, 7:] / math.pi), -1)

    @staticmethod
    def decode_pose(raw: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(raw[:, 3:7], dim=-1)
        return torch.cat((current_q[:, :3] + raw[:, :3], quat, raw[:, 7:]), -1)

    @staticmethod
    def _sample_height(heightmap: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        lateral = 2 * (xy[..., 1] - HEIGHTMAP_LATERAL_MIN_M) / (
            HEIGHTMAP_LATERAL_MAX_M - HEIGHTMAP_LATERAL_MIN_M
        ) - 1
        forward = 2 * (xy[..., 0] - HEIGHTMAP_FORWARD_MIN_M) / (
            HEIGHTMAP_FORWARD_MAX_M - HEIGHTMAP_FORWARD_MIN_M
        ) - 1
        grid = torch.stack((lateral, forward), -1)[:, :, None]
        return F.grid_sample(
            heightmap[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[:, 0, :, 0]

    def _refine(
        self,
        q: torch.Tensor,
        body: torch.Tensor,
        role_logits: torch.Tensor,
        surface_logits: torch.Tensor,
        heightmap: torch.Tensor,
    ) -> torch.Tensor:
        position, rotation6d = self.fk(q[:, None])
        position, rotation6d = position[:, 0, 1:7], rotation6d[:, 0, 1:7]
        sampled = self._sample_height(heightmap, position[:, :, :2])
        clearance = position[:, :, 2] - sampled
        feature = torch.cat(
            (
                position,
                rotation6d,
                sampled[..., None],
                clearance[..., None],
                role_logits.softmax(-1),
                surface_logits.softmax(-1),
            ),
            -1,
        )
        part = self.refine_part(feature) + self.part_identity.weight[None]
        tokens = self.refine_decoder(torch.cat(((body + self.refine_state(self._state(q)))[:, None], part), 1))
        delta = self.refine_head(self.norm(tokens[:, 0] + tokens[:, 1:].mean(1)))
        quaternion = F.normalize(q[:, 3:7] + 0.1 * delta[:, 3:7], dim=-1)
        return torch.cat((q[:, :3] + delta[:, :3], quaternion, q[:, 7:] + delta[:, 7:]), -1)

    def forward(
        self,
        current_q: torch.Tensor,
        current_contact: torch.Tensor,
        current_anchor: torch.Tensor,
        current_surface: torch.Tensor,
        heightmap: torch.Tensor,
    ) -> NextInteraction:
        terrain, _ = self.height(heightmap)
        positions, rotations = self.fk(current_q[:, None])
        position, rotation6d = positions[:, 0, 1:7], rotations[:, 0, 1:7]
        state = self.global_encoder(self._state(current_q))[:, None]
        surface_index = torch.where(current_contact, current_surface.clamp(0, 1), 2)
        part = self.part_encoder(
            torch.cat(
                (position, rotation6d, current_anchor * current_contact[..., None], current_contact[..., None].float()),
                -1,
            )
        )
        part = part + self.part_identity.weight[None] + self.surface_embedding(surface_index)
        tokens = self.shared(torch.cat((state, part, terrain), 1))
        body = self.norm(tokens[:, 0] + tokens[:, 1:7].mean(1) + tokens[:, 7:].mean(1))
        part = self.norm(tokens[:, 1:7])
        role_logits = self.role_head(part)
        surface_logits = self.surface_head(part)
        contact = (
            part
            + self.role_embedding(role_logits.softmax(-1))
            + self.predicted_surface_embedding(surface_logits.softmax(-1))
        )
        interaction = self.interaction_decoder(torch.cat((body[:, None], contact), 1))
        shared = self.norm(interaction[:, 0] + interaction[:, 1:].mean(1))
        q = self.decode_pose(self.pose_head(shared), current_q)
        for _ in range(self.refinements):
            q = self._refine(q, shared, role_logits, surface_logits, heightmap)
        return NextInteraction(q, role_logits, surface_logits, F.softplus(self.duration_head(shared)[:, 0]))


def objective(
    model: HeightmapInteractionPredictor,
    prediction: NextInteraction,
    target: dict[str, torch.Tensor],
    scene: dict[str, torch.Tensor],
    *,
    invalid_witness_policy: str = "error",
    witness_audit_path=None,
):
    """Newton objective plus a batch-tail penalty for rare deep collisions."""
    from generator.next_interaction_surface import objective as newton_objective

    loss, metrics = newton_objective(
        model,
        prediction,
        target,
        scene,
        invalid_witness_policy=invalid_witness_policy,
        witness_audit_path=witness_audit_path,
    )
    depth = metrics["newton_fullbody_penetration_cm"] / 100.0
    # The old per-sample quadratic did not prevent a few 15--22 cm failures
    # from being diluted by hundreds of clear validation poses.  Keep its
    # ordinary gradients and add a smooth quartic term, emphasizing the worst
    # fifth of each batch without inventing a new collision definition.
    tail = depth.detach() >= torch.quantile(depth.detach(), 0.8)
    barrier = 0.02 * (depth / 0.01).pow(4) * (1.0 + 3.0 * tail.to(depth.dtype))
    metrics["penetration_tail_barrier"] = barrier
    metrics["loss"] = loss + barrier
    return loss + barrier, metrics


__all__ = [
    "HEIGHTMAP_COLS",
    "HEIGHTMAP_FORWARD_MAX_M",
    "HEIGHTMAP_FORWARD_MIN_M",
    "HEIGHTMAP_LATERAL_MAX_M",
    "HEIGHTMAP_LATERAL_MIN_M",
    "HEIGHTMAP_RESOLUTION_M",
    "HEIGHTMAP_ROWS",
    "HeightmapInteractionPredictor",
    "heightmap_grid",
    "objective",
    "render_box_heightmaps",
]
