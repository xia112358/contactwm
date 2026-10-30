"""Dense latent terrain reasoning for the observation-only interaction predictor.

The height map remains the only terrain input. Surface identity and geometric
relations are never supplied as analytic features or auxiliary labels.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from generator.next_interaction import NextInteraction
from somaforge_core.heightmap import HEIGHTMAP_CLIP_M, HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MAX_M, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MAX_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_ROWS, heightmap_grid
from somaforge_core.heightmap import _root_yaw_basis, render_root_yaw_box_heightmaps
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d, _quaternion_matrix_wxyz, _quaternion_multiply_wxyz, _rotation6d


class DenseHeightmapEncoder(nn.Module):
    """Encode the 2 cm observation into a 4 cm spatial memory."""

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
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, heightmap: torch.Tensor) -> torch.Tensor:
        if heightmap.shape[1:] != (HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
            raise ValueError(
                f"heightmap must be [B,{HEIGHTMAP_ROWS},{HEIGHTMAP_COLS}], got {tuple(heightmap.shape)}"
            )
        coordinates = self.coordinates.to(heightmap).expand(len(heightmap), -1, -1, -1)
        feature = self.conv(torch.cat((heightmap[:, None] / HEIGHTMAP_CLIP_M, coordinates), 1))
        return self.norm(feature.flatten(2).transpose(1, 2))


class TerrainCrossBlock(nn.Module):
    """Update a small set of body queries without modifying terrain memory."""

    def __init__(self, width: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 6, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        query = query + self.attention(
            normalized_query, normalized_memory, normalized_memory, need_weights=False
        )[0]
        return query + self.ffn(self.ffn_norm(query))


class DenseHeightmapInteractionPredictor(nn.Module):
    """Infer contact and pose directly from dense latent terrain memory."""

    def __init__(self, width: int = 192, layers: int = 3, *, couple_contact_pose: bool = False):
        super().__init__()
        self.couple_contact_pose = bool(couple_contact_pose)
        self.fk = CanonicalG1ForwardKinematics()
        self.geometry = CanonicalG1CollisionPoints(64)
        self.height = DenseHeightmapEncoder(width)
        self.global_encoder = nn.Linear(38, width)
        self.part_encoder = nn.Linear(13, width)
        self.part_identity = nn.Embedding(6, width)
        self.surface_embedding = nn.Embedding(3, width)
        self.terrain_reasoning = nn.ModuleList((TerrainCrossBlock(width), TerrainCrossBlock(width)))
        layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.shared = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.role_head = nn.Linear(width, 4)
        self.surface_head = nn.Linear(width, 2)
        self.role_embedding = nn.Linear(4, width, bias=False)
        self.predicted_surface_embedding = nn.Linear(2, width, bias=False)
        self.pose_terrain = TerrainCrossBlock(width)
        interaction = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.interaction_decoder = nn.TransformerEncoder(interaction, 1, enable_nested_tensor=False)
        self.pose_head = nn.Linear(width, 36)
        self.duration_head = nn.Linear(width, 1)
        nn.init.normal_(self.pose_head.weight, std=0.001)
        nn.init.zeros_(self.pose_head.bias)
        with torch.no_grad():
            self.pose_head.bias[3] = 1.0

    @staticmethod
    def _to_local(points: torch.Tensor, q: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bpj->bpi", basis.transpose(1, 2), points - q[:, None, :3])

    @staticmethod
    def _local_state(q: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        local_rotation = basis.transpose(1, 2) @ _quaternion_matrix_wxyz(q[:, 3:7])
        return torch.cat(
            (torch.zeros_like(q[:, :3]), _rotation6d(local_rotation), q[:, 7:] / math.pi), -1
        )

    @staticmethod
    def _decode_pose(raw, current_q, basis, yaw_quaternion):
        world_delta = torch.einsum("bij,bj->bi", basis, raw[:, :3])
        world_quaternion = _quaternion_multiply_wxyz(
            yaw_quaternion, F.normalize(raw[:, 3:7], dim=-1)
        )
        return torch.cat((current_q[:, :3] + world_delta, world_quaternion, raw[:, 7:]), -1)

    def encode_pose(self, target_q: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
        basis, yaw_quaternion = _root_yaw_basis(current_q)
        local_delta = torch.einsum(
            "bij,bj->bi", basis.transpose(1, 2), target_q[:, :3] - current_q[:, :3]
        )
        inverse_yaw = yaw_quaternion.clone()
        inverse_yaw[:, 1:] *= -1
        local_quaternion = _quaternion_multiply_wxyz(inverse_yaw, target_q[:, 3:7])
        return torch.cat((local_delta, local_quaternion, target_q[:, 7:]), -1)

    def decode_pose(self, raw: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
        basis, yaw_quaternion = _root_yaw_basis(current_q)
        return self._decode_pose(raw, current_q, basis, yaw_quaternion)

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
        part = self.part_encoder(
            torch.cat(
                (
                    local_position,
                    _rotation6d(local_rotation),
                    local_anchor * current_contact[..., None],
                    current_contact[..., None].float(),
                ),
                -1,
            )
        )
        part = part + self.part_identity.weight[None] + self.surface_embedding(surface_index)
        state = self.global_encoder(self._local_state(current_q, basis))[:, None]
        queries = torch.cat((state, part), 1)
        for block in self.terrain_reasoning:
            queries = block(queries, terrain)
        tokens = self.shared(queries)
        body, part = self.norm(tokens[:, :1]), self.norm(tokens[:, 1:])
        role_logits, surface_logits = self.role_head(part), self.surface_head(part)
        role_probability = role_logits.softmax(-1)
        surface_probability = surface_logits.softmax(-1)
        if not self.couple_contact_pose:
            role_probability = role_probability.detach()
            surface_probability = surface_probability.detach()
        contact = part + self.role_embedding(role_probability)
        contact = contact + self.predicted_surface_embedding(surface_probability)
        interaction = self.pose_terrain(torch.cat((body, contact), 1), terrain)
        interaction = self.interaction_decoder(interaction)
        shared = self.norm(interaction[:, 0] + interaction[:, 1:].mean(1))
        q = self._decode_pose(self.pose_head(shared), current_q, basis, yaw_quaternion)
        return NextInteraction(q, role_logits, surface_logits, F.softplus(self.duration_head(shared)[:, 0]))


def objective(model, prediction, target, scene, **kwargs):
    from generator.next_interaction_surface import objective as newton_objective

    return newton_objective(model, prediction, target, scene, **kwargs)


__all__ = [
    "DenseHeightmapEncoder",
    "DenseHeightmapInteractionPredictor",
    "render_root_yaw_box_heightmaps",
]
