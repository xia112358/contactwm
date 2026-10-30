"""Observation-only predictor whose single output is the next interaction.

The network does not emit a separate contact plan and does not condition a
second pose head on a predicted/teacher plan.  It emits one embodied action
(``qpos``); the interaction is the contact transition actually realized by
that action.  Newton labels supervise realization but never enter forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from somaforge_core.heightmap import HEIGHTMAP_COLS, HEIGHTMAP_FORWARD_MAX_M, HEIGHTMAP_FORWARD_MIN_M, HEIGHTMAP_LATERAL_MAX_M, HEIGHTMAP_LATERAL_MIN_M, HEIGHTMAP_ROWS, heightmap_grid
from somaforge_core.heightmap import _root_yaw_basis
from generator.next_interaction_heightmap_v3 import DenseHeightmapEncoder, DenseHeightmapInteractionPredictor, TerrainCrossBlock
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d, _rotation6d


GRID_CELLS = HEIGHTMAP_ROWS * HEIGHTMAP_COLS


class RelationalTerrainCrossBlock(nn.Module):
    """Cross-attend through body-to-terrain geometry, not terrain alone."""

    def __init__(self, width: int, heads: int = 6):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by relational attention heads")
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.relation_bias = nn.Sequential(
            nn.Linear(4, width // 2), nn.GELU(), nn.Linear(width // 2, heads)
        )
        self.relation_gate = nn.Sequential(nn.Linear(width + 1, width), nn.Sigmoid())
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        query_position: Tensor,
        terrain_position: Tensor,
        query_contact: Tensor,
    ) -> Tensor:
        if query_position.shape != (*query.shape[:2], 3):
            raise ValueError("query_position must be [B,Q,3]")
        if terrain_position.shape != (*memory.shape[:2], 3):
            raise ValueError("terrain_position must be [B,N,3]")
        if query_contact.shape != query.shape[:2]:
            raise ValueError("query_contact must be [B,Q]")
        relative = terrain_position[:, None] - query_position[:, :, None]
        relation = torch.cat((
            relative,
            torch.linalg.vector_norm(relative[..., :2], dim=-1, keepdim=True),
        ), -1)
        bias = self.relation_bias(relation).permute(0, 3, 1, 2).flatten(0, 1)
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        update = self.attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
            attn_mask=bias,
            need_weights=False,
        )[0]
        gate = self.relation_gate(torch.cat((
            normalized_query, query_contact[..., None].to(query)
        ), -1))
        query = query + gate * update
        return query + self.ffn(self.ffn_norm(query))


class RelationalHeightmapEncoder(nn.Module):
    """Encode terrain appearance without an absolute raster-position shortcut."""

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, heightmap: Tensor) -> Tensor:
        if heightmap.shape[1:] != (HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
            raise ValueError(
                f"heightmap must be [B,{HEIGHTMAP_ROWS},{HEIGHTMAP_COLS}]"
            )
        feature = self.conv(heightmap[:, None])
        return self.norm(feature.flatten(2).transpose(1, 2))


def contact_points_to_heightmap_cells(
    current_q: Tensor,
    contact: Tensor,
    points_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rasterize actual Newton contact points into nearest 2 cm cells.

    This function only changes coordinates.  ``contact`` must already be the
    authoritative Newton/MJWarp activation result after primary-face
    selection; distance to the height map never creates a contact label.

    Returns ``(cell, valid)`` with shape ``[B,6]``.  Inactive and out-of-view
    entries use cell zero and have ``valid=False``.
    """

    batch = len(current_q)
    if current_q.shape != (batch, 36):
        raise ValueError("current_q must be [B,36]")
    if contact.shape != (batch, 6) or contact.dtype != torch.bool:
        raise ValueError("contact must be bool [B,6]")
    if points_world.shape != (batch, 6, 3):
        raise ValueError("points_world must be [B,6,3]")
    basis, _ = _root_yaw_basis(current_q)
    local = torch.einsum(
        "bij,bpj->bpi",
        basis.transpose(1, 2),
        points_world - current_q[:, None, :3],
    )
    forward, lateral = local[..., 0], local[..., 1]
    valid = (
        contact
        & (forward >= HEIGHTMAP_FORWARD_MIN_M)
        & (forward <= HEIGHTMAP_FORWARD_MAX_M)
        & (lateral >= HEIGHTMAP_LATERAL_MIN_M)
        & (lateral <= HEIGHTMAP_LATERAL_MAX_M)
    )
    row = ((forward - HEIGHTMAP_FORWARD_MIN_M) / (
        HEIGHTMAP_FORWARD_MAX_M - HEIGHTMAP_FORWARD_MIN_M
    ) * (HEIGHTMAP_ROWS - 1)).round().long().clamp(0, HEIGHTMAP_ROWS - 1)
    column = ((lateral - HEIGHTMAP_LATERAL_MIN_M) / (
        HEIGHTMAP_LATERAL_MAX_M - HEIGHTMAP_LATERAL_MIN_M
    ) * (HEIGHTMAP_COLS - 1)).round().long().clamp(0, HEIGHTMAP_COLS - 1)
    cell = row * HEIGHTMAP_COLS + column
    return torch.where(valid, cell, torch.zeros_like(cell)), valid


def contact_cells_to_map(cell: Tensor, valid: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Return six one-hot contact maps without inventing missing contacts."""

    if cell.shape != valid.shape or cell.ndim != 2 or cell.shape[1] != 6:
        raise ValueError("cell and valid must be [B,6]")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be bool")
    result = F.one_hot(cell.clamp(0, GRID_CELLS - 1), GRID_CELLS).to(dtype)
    return (result * valid[..., None]).reshape(
        len(cell), 6, HEIGHTMAP_ROWS, HEIGHTMAP_COLS
    )


def contact_points_to_heightmap_map(
    current_q: Tensor,
    contact: Tensor,
    points_world: Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience wrapper returning ``(map, cell, valid)``."""

    cell, valid = contact_points_to_heightmap_cells(current_q, contact, points_world)
    if dtype is None:
        dtype = current_q.dtype
    return contact_cells_to_map(cell, valid, dtype=dtype), cell, valid


def contact_points_to_observed_heightmap_map(
    current_q: Tensor, contact: Tensor, points_world: Tensor, heightmap: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Locate known contact points in observed 3D geometry, without face IDs.

    XY rounding can select the lower level beside a ledge. Nearest 3D scan
    samples preserve the observed height. The supplied Newton mask remains
    authoritative; this function only encodes location, never contact truth.
    """
    _, valid = contact_points_to_heightmap_cells(current_q, contact, points_world)
    basis, _ = _root_yaw_basis(current_q)
    local = torch.einsum("bij,bpj->bpi", basis.transpose(1, 2),
                         points_world - current_q[:, None, :3])
    grid = torch.as_tensor(heightmap_grid(), device=current_q.device, dtype=current_q.dtype).reshape(-1, 2)
    cells = []
    for start in range(0, len(current_q), 64):
        heights = heightmap[start:start + 64].flatten(1)
        geometry = torch.cat((grid[None].expand(len(heights), -1, -1), heights[..., None]), -1)
        cells.append(torch.cdist(local[start:start + 64], geometry).argmin(-1))
    cell = torch.where(valid, torch.cat(cells), torch.zeros_like(valid, dtype=torch.long))
    return contact_cells_to_map(cell, valid, dtype=current_q.dtype), cell, valid


@dataclass(frozen=True)
class ContactLocationPrediction:
    qpos: Tensor


class HeightmapContactLocationPredictor(nn.Module):
    """Predict one embodied next interaction directly from observations."""

    def __init__(
        self,
        width: int = 192,
        layers: int = 3,
        location_width: int = 32,
        *,
        joint_residual_output: bool = False,
        relational_terrain: bool = False,
    ):
        super().__init__()
        self.joint_residual_output = bool(joint_residual_output)
        self.relational_terrain = bool(relational_terrain)
        self.fk = CanonicalG1ForwardKinematics()
        self.geometry = CanonicalG1CollisionPoints(64)
        self.height = (
            RelationalHeightmapEncoder(width)
            if self.relational_terrain else DenseHeightmapEncoder(width)
        )
        self.global_encoder = nn.Linear(38, width)
        self.part_encoder = nn.Linear(13, width)
        self.part_identity = nn.Embedding(6, width)

        grid = torch.from_numpy(heightmap_grid()).reshape(-1, 2)
        self.register_buffer("grid_xy", grid, persistent=False)
        terrain_grid = torch.from_numpy(heightmap_grid()[::2, ::2]).reshape(-1, 2)
        self.register_buffer("terrain_grid_xy", terrain_grid, persistent=False)
        normalized = grid.clone()
        normalized[:, 0] = 2.0 * (
            normalized[:, 0] - HEIGHTMAP_FORWARD_MIN_M
        ) / (HEIGHTMAP_FORWARD_MAX_M - HEIGHTMAP_FORWARD_MIN_M) - 1.0
        normalized[:, 1] = 2.0 * (
            normalized[:, 1] - HEIGHTMAP_LATERAL_MIN_M
        ) / (HEIGHTMAP_LATERAL_MAX_M - HEIGHTMAP_LATERAL_MIN_M) - 1.0
        self.register_buffer("normalized_grid_xy", normalized, persistent=False)
        location_channels = 1 if self.relational_terrain else 3
        self.location_key = nn.Sequential(
            nn.Conv2d(location_channels, location_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(location_width, location_width, 3, padding=1),
        )
        self.current_location_feature = nn.Linear(location_width, width, bias=False)

        terrain_block = RelationalTerrainCrossBlock if self.relational_terrain else TerrainCrossBlock
        self.terrain_reasoning = nn.ModuleList((terrain_block(width), terrain_block(width)))
        state_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.shared = nn.TransformerEncoder(
            state_layer, layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.pose_terrain = terrain_block(width)
        pose_layer = nn.TransformerEncoderLayer(
            width, 6, 2 * width, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.interaction_decoder = nn.TransformerEncoder(
            pose_layer, 2, enable_nested_tensor=False
        )
        # The action readout retains an explicit view of the current body.
        self.pose_state_encoder = nn.Sequential(
            nn.Linear(38, width), nn.GELU(), nn.Linear(width, width)
        )
        # Preserve the fixed state/limb token identities.  Mean pooling made
        # distinct multi-contact plans collide in the same pose bottleneck.
        self.pose_token_readout = nn.Sequential(
            nn.LayerNorm(7 * width),
            nn.Linear(7 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.pose_head = nn.Linear(width, 36)
        nn.init.normal_(self.pose_head.weight, std=0.001)
        nn.init.zeros_(self.pose_head.bias)
        with torch.no_grad():
            self.pose_head.bias[3] = 1.0

    def load_observation_encoder(
        self, state: dict[str, Tensor]
    ) -> tuple[list[str], list[str]]:
        """Warm-start only features whose observable meaning is unchanged.

        In particular, no old surface embedding/head or future-plan decoder
        parameter is accepted.  The contact raster and location pathway remain
        newly initialized.
        """

        prefixes = (
            "height.", "global_encoder.", "part_encoder.", "part_identity.",
            "terrain_reasoning.", "shared.", "norm.",
        )
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        incompatible = self.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"unexpected observation warm-start keys: {incompatible.unexpected_keys}"
            )
        return sorted(selected), sorted(incompatible.missing_keys)

    def _grid_geometry(self, heightmap: Tensor) -> Tensor:
        xy = self.grid_xy.to(heightmap)[None].expand(len(heightmap), -1, -1)
        return torch.cat((xy, heightmap.flatten(1)[..., None]), -1)

    def _terrain_geometry(self, heightmap: Tensor) -> Tensor:
        xy = self.terrain_grid_xy.to(heightmap)[None].expand(len(heightmap), -1, -1)
        return torch.cat((xy, heightmap[:, ::2, ::2].flatten(1)[..., None]), -1)

    def _location_keys(self, heightmap: Tensor) -> Tensor:
        if self.relational_terrain:
            return self.location_key(heightmap[:, None]).flatten(2).transpose(1, 2)
        coordinates = self.normalized_grid_xy.to(heightmap).T.reshape(
            1, 2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS
        ).expand(len(heightmap), -1, -1, -1)
        values = torch.cat((heightmap[:, None], coordinates), 1)
        return self.location_key(values).flatten(2).transpose(1, 2)

    def _encode_observation(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_contact_map: Tensor,
        heightmap: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = len(current_q)
        if current_contact.shape != (batch, 6) or current_contact.dtype != torch.bool:
            raise ValueError("current_contact must be bool [B,6]")
        if current_contact_map.shape != (batch, 6, HEIGHTMAP_ROWS, HEIGHTMAP_COLS):
            raise ValueError(
                f"current_contact_map must be [B,6,{HEIGHTMAP_ROWS},{HEIGHTMAP_COLS}]"
            )
        terrain = self.height(heightmap)
        keys = self._location_keys(heightmap)
        geometry = self._grid_geometry(heightmap)
        terrain_geometry = self._terrain_geometry(heightmap)
        basis, yaw_quaternion = _root_yaw_basis(current_q)
        positions, rotations6d = self.fk(current_q[:, None])
        positions = positions[:, 0, 1:7]
        rotations = _matrix_from_rotation6d(rotations6d[:, 0, 1:7])
        local_position = torch.einsum(
            "bij,bpj->bpi", basis.transpose(1, 2), positions - current_q[:, None, :3]
        )
        local_rotation = basis[:, None].transpose(-1, -2) @ rotations

        contact_weight = current_contact_map.flatten(2).clamp_min(0.0)
        contact_weight = contact_weight / contact_weight.sum(-1, keepdim=True).clamp_min(1.0e-8)
        current_point = torch.einsum("bpn,bnd->bpd", contact_weight, geometry)
        current_feature = torch.einsum("bpn,bnd->bpd", contact_weight, keys)
        part = self.part_encoder(torch.cat((
            local_position,
            _rotation6d(local_rotation),
            current_point * current_contact[..., None],
            current_contact[..., None].float(),
        ), -1))
        part = (
            part
            + self.part_identity.weight[None]
            + self.current_location_feature(current_feature) * current_contact[..., None]
        )
        state = self.global_encoder(
            DenseHeightmapInteractionPredictor._local_state(current_q, basis)
        )[:, None]
        queries = torch.cat((state, part), 1)
        query_position = torch.cat((
            torch.zeros_like(local_position[:, :1]), local_position
        ), 1)
        query_contact = torch.cat((
            torch.ones_like(current_contact[:, :1]), current_contact
        ), 1)
        for block in self.terrain_reasoning:
            if self.relational_terrain:
                queries = block(
                    queries, terrain, query_position, terrain_geometry, query_contact
                )
            else:
                queries = block(queries, terrain)
        encoded = self.shared(queries)
        body = self.norm(encoded[:, :1])
        part = self.norm(encoded[:, 1:])
        return terrain, keys, geometry, basis, yaw_quaternion, body, part

    def interaction_state(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_contact_map: Tensor,
        heightmap: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the embodied interaction state used by the single q readout."""

        terrain, keys, geometry, basis, yaw_quaternion, body, part = self._encode_observation(
            current_q, current_contact, current_contact_map, heightmap
        )
        return self._decode_interaction_state(
            current_q, current_contact, heightmap, terrain, basis,
            yaw_quaternion, body, part,
        )

    def _decode_interaction_state(
        self, current_q, current_contact, heightmap, terrain, basis,
        yaw_quaternion, body, part,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tokens = torch.cat((body, part), 1)
        if self.relational_terrain:
            positions, _ = self.fk(current_q[:, None])
            local_position = torch.einsum(
                "bij,bpj->bpi",
                basis.transpose(1, 2),
                positions[:, 0, 1:7] - current_q[:, None, :3],
            )
            query_position = torch.cat((
                torch.zeros_like(local_position[:, :1]), local_position
            ), 1)
            query_contact = torch.cat((
                torch.ones_like(current_contact[:, :1]), current_contact
            ), 1)
            interaction = self.pose_terrain(
                tokens,
                terrain,
                query_position,
                self._terrain_geometry(heightmap),
                query_contact,
            )
        else:
            interaction = self.pose_terrain(tokens, terrain)
        interaction = self.interaction_decoder(interaction)
        pose_state = self.pose_state_encoder(
            DenseHeightmapInteractionPredictor._local_state(current_q, basis)
        )
        pooled = self.norm(
            self.pose_token_readout(interaction.flatten(1)) + pose_state
        )
        return pooled, basis, yaw_quaternion

    def forward(
        self,
        current_q: Tensor,
        current_contact: Tensor,
        current_contact_map: Tensor,
        heightmap: Tensor,
    ) -> ContactLocationPrediction:
        pooled, basis, yaw_quaternion = self.interaction_state(
            current_q, current_contact, current_contact_map, heightmap
        )
        raw_pose = self.pose_head(pooled)
        qpos = DenseHeightmapInteractionPredictor._decode_pose(
            raw_pose, current_q, basis, yaw_quaternion
        )
        if self.joint_residual_output:
            qpos = torch.cat((qpos[:, :7], current_q[:, 7:] + raw_pose[:, 7:]), -1)
        return ContactLocationPrediction(qpos=qpos)

    encode_pose = DenseHeightmapInteractionPredictor.encode_pose
    _decode_pose = staticmethod(DenseHeightmapInteractionPredictor._decode_pose)
    decode_pose = DenseHeightmapInteractionPredictor.decode_pose


__all__ = [
    "ContactLocationPrediction",
    "HeightmapContactLocationPredictor",
    "contact_cells_to_map",
    "contact_points_to_heightmap_cells",
    "contact_points_to_heightmap_map",
    "contact_points_to_observed_heightmap_map",
]
