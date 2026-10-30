"""Paper Table S3 networks and Eq. (2) supervision."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mlp(widths, activation=nn.ReLU, final=None):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(activation())
    if final is not None:
        layers.append(final())
    return nn.Sequential(*layers)


class ContactPlanner(nn.Module):
    """Nine body queries over 187 measured terrain points; no free XYZ head.

    The default 23-joint model follows Table S3. The canonical SomaForge
    29-joint variant changes only the proprioceptive input width.
    """
    def __init__(self, num_joints=23, top_k=1, temperature=1.0):
        super().__init__()
        if not 1 <= top_k <= 187 or temperature <= 0:
            raise ValueError("Invalid contact decoder settings")
        self.num_joints, self.top_k, self.temperature = num_joints, top_k, temperature
        self.history_dim = 5 * (6 + 3 * num_joints)
        self.point_embedding = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.height_encoder = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ELU(),
                                            nn.Conv2d(16, 48, 5, padding=2), nn.ELU())
        self.terrain_projection = nn.Linear(64, 96)
        self.context_encoder = mlp([96, 64, 64], nn.ELU)
        self.state_encoder = mlp([self.history_dim + 3 + 64, 128, 128, 96])
        self.body_embedding = nn.Embedding(9, 96)
        self.film = mlp([96, 64, 192])
        self.attention = nn.MultiheadAttention(96, 32, dropout=0.0, batch_first=True)
        self.occupancy = mlp([160, 512, 512, 512, 1])

    def forward(self, history, goal, terrain, valid=None):
        if history.shape[-1] != self.history_dim or terrain.shape[1:] != (187, 3):
            raise ValueError("Planner expects five-step proprioception and 17x11 XYZ terrain")
        if not torch.isfinite(terrain).all():
            raise ValueError("Terrain must contain finite coordinates, with a separate validity mask")
        if valid is None:
            valid = torch.ones(terrain.shape[:2], dtype=torch.bool, device=terrain.device)
        if valid.shape != terrain.shape[:2] or not valid.any(-1).all():
            raise ValueError("Each planner scan must contain an observed terrain point")
        height = self.height_encoder(terrain[..., 2].reshape(-1, 1, 17, 11)).flatten(2).transpose(1, 2)
        features = self.terrain_projection(torch.cat((self.point_embedding(terrain), height), -1))
        context = self.context_encoder(features).masked_fill(~valid[..., None], -torch.inf).amax(1)
        state = self.state_encoder(torch.cat((history, goal, context), -1))
        scale, shift = self.film(self.body_embedding.weight).chunk(2, -1)
        query = state[:, None] * (1 + scale[None]) + shift[None]
        attended, weights = self.attention(query, features, features, key_padding_mask=~valid,
                                           need_weights=True, average_attn_weights=True)
        occupancy_logits = self.occupancy(torch.cat((attended, context[:, None].expand(-1, 9, -1)), -1)).squeeze(-1)
        values, indices = weights.topk(self.top_k, -1)
        selected = terrain[:, None].expand(-1, 9, -1, -1).gather(2, indices[..., None].expand(-1, -1, -1, 3))
        decoder_weights = (values.clamp_min(1e-30).log() / self.temperature).softmax(-1)
        positions = (decoder_weights[..., None] * selected).sum(2)
        occupancy = occupancy_logits.sigmoid() >= 0.5
        return {"positions": positions * occupancy[..., None], "ungated_positions": positions,
                "occupancy": occupancy, "occupancy_logits": occupancy_logits,
                "affordance": weights, "valid": valid}


def planner_loss(prediction, terrain, target, occupancy, known=None, bandwidth=0.1, normalizers=None):
    """Masked Gaussian affordance CE plus nine-body occupancy BCE, Eq. (2)."""
    if bandwidth <= 0:
        raise ValueError("Gaussian bandwidth must be positive")
    known = torch.ones_like(occupancy, dtype=torch.bool) if known is None else known.bool()
    logits = -((terrain[:, None] - target[:, :, None]) ** 2).sum(-1) / (2 * bandwidth**2)
    logits = logits.masked_fill(~prediction["valid"][:, None], -torch.inf)
    distribution = logits.softmax(-1)
    present = occupancy.bool() & known
    ce = -(distribution * prediction["affordance"].clamp_min(1e-30).log()).sum(-1)
    if normalizers is None:
        normalizers = (present.sum().clamp_min(1), known.sum().clamp_min(1))
    affordance = (ce * present).sum() / normalizers[0]
    bce = F.binary_cross_entropy_with_logits(prediction["occupancy_logits"], occupancy.float(), reduction="none")
    occupancy_loss = (bce * known).sum() / normalizers[1]
    return {"loss": affordance + occupancy_loss, "affordance": affordance, "occupancy": occupancy_loss}


def contact_schedule(iteration, start, duration):
    if duration <= 0:
        raise ValueError("Contact transition duration must be positive")
    return min(1.0, max(0.0, (iteration - start) / duration))
