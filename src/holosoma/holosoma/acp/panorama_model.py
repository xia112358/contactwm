"""Experimental panoramic ACP planner; independent of the paper height-map model.

Metric full-sphere depth is read at a gravity-aligned virtual viewpoint. History
and goal use the base-yaw frame. Geometry is never inferred for unknown pixels.
Output contacts are intentions, not measured/verified physical contacts.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from holosoma.acp.models import mlp


class SphericalConv(nn.Module):
    """Wrap longitude only; the two elevation poles are not adjacent rows."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        return F.elu(self.conv(F.pad(x, (0, 0, 1, 1))))


class PanoramaContactPlanner(nn.Module):
    """Six body queries by default select actual observed pixels, without XYZ averaging.

    Inputs: history [B,5*(6+3*J)], goal [B,3], depth/valid/age [B,H,W],
    base_position_w/view_origin_w [B,3], yaw [B] radians. Metric depth and age
    are metres and seconds. Confidence [B,H,W] is optional, heuristic [0,1].
    The caller supplies the elevated view origin; roll/pitch are not inputs.
    Invalid depth/age placeholders may be NaN/inf and are masked before encoding.
    """
    def __init__(self, num_joints=29, height=64, width=128, max_range=2.5, max_age=15., num_contacts=6):
        super().__init__()
        if num_contacts not in (6, 9):
            raise ValueError("num_contacts must be 6 or 9")
        self.num_contacts = num_contacts
        self.contact_names = ("left_foot", "right_foot", "left_knee", "right_knee",
                              "left_hand", "right_hand", "left_elbow", "right_elbow", "pelvis")[:num_contacts]
        if num_joints < 1 or height < 2 or width < 4 or max_range <= 0 or max_age <= 0:
            raise ValueError("Invalid panorama model configuration")
        self.config = dict(num_joints=num_joints, height=height, width=width,
                           max_range=max_range, max_age=max_age, num_contacts=num_contacts)
        self.history_dim = 5*(6+3*num_joints)
        az = -math.pi+(torch.arange(width)+.5)*2*math.pi/width
        el = math.pi/2-(torch.arange(height)+.5)*math.pi/height
        el, az = torch.meshgrid(el, az, indexing="ij")
        self.register_buffer("directions", torch.stack((el.cos()*az.cos(), el.cos()*az.sin(), el.sin()), -1))
        # Depth, known mask, age, confidence; no implicit imputation of unknown.
        self.image_encoder = nn.Sequential(SphericalConv(4, 32), SphericalConv(32, 48))
        self.point_encoder = nn.Sequential(nn.Linear(3, 32), nn.ELU())
        self.projection = nn.Linear(80, 96)
        self.context_encoder = mlp([96, 64, 64], nn.ELU)
        self.state_encoder = mlp([self.history_dim+3+64, 128, 128, 96])
        self.body_embedding = nn.Embedding(num_contacts, 96)
        self.film = mlp([96, 64, 192])
        self.attention = nn.MultiheadAttention(96, 8, dropout=0., batch_first=True)
        self.occupancy_head = mlp([160, 256, 256, 1])

    def forward(self, history, goal, depth, valid, age, base_position_w, view_origin_w, yaw, confidence=None):
        b = history.shape[0]
        shape = (b, self.config['height'], self.config['width'])
        if (history.shape != (b, self.history_dim) or goal.shape != (b, 3)
                or depth.shape != shape or valid.shape != shape or age.shape != shape
                or base_position_w.shape != (b, 3) or view_origin_w.shape != (b, 3) or yaw.shape != (b,)):
            raise ValueError("Invalid panorama/state/pose input shape")
        if valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean mask")
        if not all(torch.isfinite(x).all() for x in (history, goal, base_position_w, view_origin_w, yaw)):
            raise ValueError("State and poses must be finite")
        if confidence is None:
            confidence = torch.ones_like(depth)
        if confidence.shape != shape:
            raise ValueError("Invalid confidence shape")
        bad = (~torch.isfinite(depth) | (depth <= 0) | (depth > self.config['max_range'])
               | ~torch.isfinite(age) | (age < 0) | ~torch.isfinite(confidence)
               | (confidence < 0) | (confidence > 1))
        if (bad & valid).any():
            raise ValueError("Known pixels must have finite, in-range depth, age and confidence")
        valid = valid & (age <= self.config['max_age'])
        depth = torch.where(valid, depth, 0.)
        age = torch.where(valid, age, 0.)
        confidence = torch.where(valid, confidence, 0.)
        mask = valid.flatten(1)
        available = mask.any(-1)
        c, s = yaw.cos(), yaw.sin()
        rays = depth[..., None]*self.directions
        delta = view_origin_w-base_position_w
        offset = torch.stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1], delta[:, 2]), -1)
        points = (rays+offset[:, None, None]).flatten(1, 2)
        points = torch.where(mask[..., None], points, 0.)
        world = torch.stack((c[:, None]*points[..., 0]-s[:, None]*points[..., 1],
                             s[:, None]*points[..., 0]+c[:, None]*points[..., 1], points[..., 2]), -1)+base_position_w[:, None]
        world = torch.where(mask[..., None], world, 0.)
        image = torch.stack((depth/self.config['max_range'], valid.to(depth.dtype),
                             age/self.config['max_age'], confidence), 1)
        features = self.image_encoder(image).flatten(2).transpose(1, 2)
        features = self.projection(torch.cat((features, self.point_encoder(points/self.config['max_range'])), -1))
        features = features*mask[..., None]
        context = self.context_encoder(features).masked_fill(~mask[..., None], -torch.inf).amax(1)
        context = torch.where(available[:, None], context, 0.)
        state = self.state_encoder(torch.cat((history, goal, context), -1))
        scale, shift = self.film(self.body_embedding.weight).chunk(2, -1)
        query = state[:, None]*(1+scale[None])+shift[None]
        # One zero sentinel keeps empty-scan attention finite. It cannot be selected.
        safe = mask.clone()
        safe[:, 0] |= ~available
        attended, weights = self.attention(query, features, features, key_padding_mask=~safe,
                                           need_weights=True, average_attn_weights=True)
        weights = weights*mask[:, None]
        logits = self.occupancy_head(torch.cat((attended, context[:, None].expand(-1, self.num_contacts, -1)), -1)).squeeze(-1)
        probability = logits.sigmoid()*available[:, None]
        occupied = (probability >= .5) & available[:, None]
        indices = weights.argmax(-1)
        gather = indices[..., None].expand(-1, -1, 3)
        selected = points.gather(1, gather)
        selected_w = world.gather(1, gather)
        return dict(positions=selected*occupied[..., None], positions_w=selected_w*occupied[..., None],
                    ungated_positions=selected, ungated_positions_w=selected_w,
                    occupancy=occupied, occupancy_probability=probability, occupancy_logits=logits,
                    affordance=weights, valid=mask, available=available,
                    selected_indices=indices.masked_fill(~available[:, None], -1),
                    terrain=points, terrain_w=world)


def panorama_planner_loss(prediction, target_w, occupancy, known, bandwidth=.1, support_radius=.12):
    """Gaussian surface attention CE plus occupancy BCE.

    Positive targets lacking a nearby observed surface are masked, not snapped
    to an unrelated surface or relabelled negative. Unknown native labels and
    all-empty observations are masked. Targets use world coordinates.
    """
    if bandwidth <= 0 or support_radius <= 0:
        raise ValueError("Loss scales must be positive")
    shape = prediction['occupancy_logits'].shape
    if target_w.shape != (*shape, 3) or occupancy.shape != shape or known.shape != shape:
        raise ValueError("Invalid supervision shape")
    positive = occupancy.bool()
    known = known.bool()
    if (~torch.isfinite(target_w).all(-1) & positive & known).any():
        raise ValueError("Known positive targets must be finite")
    target = torch.where((positive & known)[..., None], target_w, 0.)
    distance = (prediction['terrain_w'][:, None]-target[:, :, None]).square().sum(-1)
    valid = prediction['valid'][:, None]
    nearest = distance.masked_fill(~valid, torch.inf).amin(-1)
    supported = nearest <= support_radius**2
    supervised = known & prediction['available'][:, None] & (~positive | supported)
    present = supervised & positive
    scores = (-distance/(2*bandwidth**2)).masked_fill(~valid, -torch.inf)
    scores = torch.where(prediction['available'][:, None, None], scores, torch.zeros_like(scores))
    distribution = scores.softmax(-1)*valid
    ce = -(distribution*prediction['affordance'].clamp_min(1e-30).log()).sum(-1)
    affordance = (ce*present).sum()/present.sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(prediction['occupancy_logits'], positive.float(), reduction='none')
    occupancy_loss = (bce*supervised).sum()/supervised.sum().clamp_min(1)
    return dict(loss=affordance+occupancy_loss, affordance=affordance, occupancy=occupancy_loss,
                supervised=supervised, supported=supported, positive_count=present.sum())
