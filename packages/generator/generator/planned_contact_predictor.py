"""Explicit six-part contact intentions conditioning the next endpoint pose.

Raster locations are intentions in the observed heightmap, never contact truth.
Only Newton activation/allocation after primary-face selection establishes the
contacts actually realized by qpos. No analytic scene enters this network.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from generator.contact_location_predictor import HeightmapContactLocationPredictor
from generator.next_interaction_heightmap_v3 import DenseHeightmapInteractionPredictor
from somaforge_core.heightmap import HEIGHTMAP_COLS, HEIGHTMAP_ROWS, HEIGHTMAP_RESOLUTION_M
from somaforge_core.heightmap import _root_yaw_basis


@dataclass(frozen=True)
class PlannedContactPrediction:
    qpos: Tensor
    role_logits: Tensor
    location_logits: Tensor
    contact_points_local: Tensor
    conditioned_contact: Tensor
    conditioned_cell: Tensor
    conditioned_points_local: Tensor

    @property
    def role(self) -> Tensor:
        return self.role_logits.argmax(-1)

    @property
    def contact(self) -> Tensor:
        return self.role != 0

    @property
    def cell(self) -> Tensor:
        return self.location_logits.argmax(-1)


class PlannedHeightmapContactPredictor(HeightmapContactLocationPredictor):
    """Choose a hard contact plan before decoding the endpoint pose."""

    PLAN_PREFIXES = ("role_head.", "location_query.", "location_relation.", "plan_fusion.")

    def __init__(self, width=192, layers=3, location_width=32, **kwargs):
        super().__init__(width, layers, location_width, **kwargs)
        self.role_head = nn.Linear(width, 4)
        self.location_query = nn.Linear(width, location_width)
        # Per-part geometry distinguishes cells with the same terrain height.
        self.location_relation = nn.Sequential(
            nn.Linear(width + 3, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.plan_fusion = nn.Linear(4, width, bias=False)
        # A migrated checkpoint initially retains its exact pose function.
        nn.init.zeros_(self.plan_fusion.weight)

    def load_embodied(self, state: dict[str, Tensor]) -> tuple[list[str], list[str]]:
        own = self.state_dict()
        missing = sorted(set(own) - set(state))
        expected = sorted(k for k in own if k.startswith(self.PLAN_PREFIXES))
        if missing != expected or set(state) - set(own):
            raise ValueError("warm start must contain the complete compatible embodied model")
        self.load_state_dict(state, strict=False)
        return sorted(state), missing

    def _predict(self, current_q, current_contact, current_contact_map, heightmap,
                 planned_contact=None, planned_cell=None):
        terrain, keys, geometry, basis, yaw, body, part = self._encode_observation(
            current_q, current_contact, current_contact_map, heightmap
        )
        # Train the newly introduced decision heads without overwriting the
        # pretrained pose representation through classification gradients.
        plan_part, plan_keys = part.detach(), keys.detach()
        role_logits = self.role_head(plan_part)
        location_logits = torch.einsum(
            "bpd,bnd->bpn", self.location_query(plan_part), plan_keys
        ) / keys.shape[-1] ** 0.5
        # Chunk parts to avoid materializing [B,6,4331,width] at once.
        relation = []
        for index in range(6):
            token = plan_part[:, index:index + 1].expand(-1, geometry.shape[1], -1)
            relation.append(self.location_relation(torch.cat((token, geometry), -1)).squeeze(-1))
        location_logits = location_logits + torch.stack(relation, 1)
        predicted_cell = location_logits.argmax(-1)
        points = geometry.gather(1, predicted_cell[..., None].expand(-1, -1, 3))
        if planned_contact is None:
            contact, cell = role_logits.argmax(-1) != 0, predicted_cell
        else:
            if planned_contact.dtype != torch.bool or planned_contact.shape != role_logits.shape[:2]:
                raise ValueError("planned_contact must be bool [B,6]")
            if planned_cell is None or planned_cell.shape != planned_contact.shape:
                raise ValueError("planned_cell must be [B,6]")
            if bool(((planned_cell < 0) | (planned_cell >= geometry.shape[1])).any()):
                raise ValueError("planned_cell is outside the observation raster")
            contact, cell = planned_contact, planned_cell
        conditioned_point = geometry.gather(1, cell[..., None].expand(-1, -1, 3))
        plan = torch.cat((contact[..., None].to(part),
                          conditioned_point * contact[..., None]), -1).detach()
        pooled, basis, yaw = self._decode_interaction_state(
            current_q, current_contact, heightmap, terrain, basis, yaw,
            body, part + self.plan_fusion(plan),
        )
        raw = self.pose_head(pooled)
        qpos = DenseHeightmapInteractionPredictor._decode_pose(raw, current_q, basis, yaw)
        if self.joint_residual_output:
            qpos = torch.cat((qpos[:, :7], current_q[:, 7:] + raw[:, 7:]), -1)
        return PlannedContactPrediction(qpos, role_logits, location_logits, points, contact, cell,
                                        conditioned_point)

    def forward(self, current_q, current_contact, current_contact_map, heightmap):
        return self._predict(current_q, current_contact, current_contact_map, heightmap)

    def forward_with_plan(self, current_q, current_contact, current_contact_map, heightmap,
                          *, planned_contact, planned_cell):
        """Training/diagnostic conditioning; deployment uses forward only."""
        return self._predict(current_q, current_contact, current_contact_map, heightmap,
                             planned_contact, planned_cell)


def contact_plan_objective(prediction, target):
    if not bool(torch.isfinite(prediction.role_logits.gather(-1, target['role'][..., None])).all()):
        raise ValueError('Supervised role is outside the actual-contact action space')
    from generator.limb_contact import role_objective
    role = role_objective(prediction, target['role'])
    valid = target["contact_cell_valid"] & (target["role"] != 0)
    if getattr(prediction, 'keep_uses_current_point', False):
        valid &= target['role'] != 2
    locations = (prediction.location_logits_for_roles(target['role'])
        if hasattr(prediction, 'location_logits_for_roles') else prediction.location_logits)
    safe_cells = torch.where(valid, target['contact_cell'], locations.argmax(-1))
    if bool((valid & ~torch.isfinite(locations.gather(-1, safe_cells[..., None]).squeeze(-1))).any()):
        raise ValueError('Supervised keep location is outside the observed contact relation')
    cells = F.cross_entropy(locations.transpose(1, 2), safe_cells, reduction="none")
    region_metrics = {}
    if 'location_region_points_local' in target:
        # Training-only proxy: probability mass on nearby observed points of the
        # demonstrated surface. Surface IDs never enter the learned score.
        candidates = prediction.candidate_points_local
        surfaces = prediction.candidate_surfaces
        observed = prediction.candidate_valid
        if candidates is None or surfaces is None or observed is None:
            raise ValueError('Regional location supervision requires registered scan candidates')
        radius = target['location_region_radius_m']
        if radius.shape != (len(locations),) or not bool(torch.isfinite(radius).all()) or bool((radius <= 0).any()):
            raise ValueError('Regional location radius must be finite positive [B]')
        distance = torch.cdist(target['location_region_points_local'].detach(), candidates.detach())
        positive = (observed[:, None] & torch.isfinite(locations)
            & (surfaces[:, None] == target['location_region_surface'][..., None])
            & (distance <= radius[:, None, None]))
        eligible = (target['role'] != 0) & (target['role'] != 2)
        covered = positive.any(-1)
        valid = valid & eligible & covered
        # Empty/excluded rows use a finite dummy support to avoid inf*0/NaN
        # gradients. They contribute neither location loss nor accuracy.
        fallback = torch.zeros_like(positive).scatter_(-1, locations.argmax(-1)[..., None], True)
        support = torch.where(valid[..., None], positive, fallback)
        cells = (torch.logsumexp(locations, -1)
            - torch.logsumexp(locations.masked_fill(~support, -torch.inf), -1))
        selected_positive = positive.gather(-1, prediction.cell[..., None]).squeeze(-1)
        region_metrics = {
            'plan_region_exact': ((selected_positive | ~valid).all(-1) & valid.any(-1)).float(),
            'plan_region_supervised_parts': valid.sum(-1).float(),
            'plan_region_uncovered_parts': (eligible & ~covered).sum(-1).float(),
            'plan_region_positive_candidates': (positive.sum(-1)*valid).sum(-1).float()/valid.sum(-1).clamp_min(1),
        }
    location = (cells * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    exact_contact = (prediction.contact == (target["role"] != 0)).all(-1)
    candidates = getattr(prediction, 'candidate_points_local', None)
    if candidates is not None:
        reference = candidates.gather(1, safe_cells[..., None].expand(-1, -1, 3))
        predicted = candidates.gather(1, prediction.cell[..., None].expand(-1, -1, 3))
        location_error = 100*(predicted-reference).norm(dim=-1)
    else:
        row_error = prediction.cell // HEIGHTMAP_COLS - target["contact_cell"] // HEIGHTMAP_COLS
        col_error = prediction.cell % HEIGHTMAP_COLS - target["contact_cell"] % HEIGHTMAP_COLS
        location_error = (row_error.square() + col_error.square()).float().sqrt() * (100 * HEIGHTMAP_RESOLUTION_M)
    return role + location, {
        **region_metrics,
        "plan_role_loss": role,
        "plan_location_loss": location,
        "plan_contact_exact": exact_contact.float(),
        "plan_role_exact": (prediction.role == target["role"]).all(-1).float(),
        "plan_location_error_cm": (location_error * valid).sum(-1) / valid.sum(-1).clamp_min(1),
        "plan_cell_exact": (exact_contact & ((prediction.cell == target["contact_cell"]) | ~valid).all(-1)).float(),
    }


def plan_points_in_pose_frame(current_q: Tensor, points_local: Tensor) -> Tensor:
    """Lift observation-root-yaw locations into the frame containing q."""
    basis, _ = _root_yaw_basis(current_q)
    return current_q[:, None, :3] + torch.einsum("bij,bpj->bpi", basis, points_local)


def translated_observed_plan_cells(cells, active, heightmap, row_shift, column_shift):
    """Counterfactual spatial intentions on the same observed height levels.

    This only selects training intentions. It makes no claim about realized
    contacts or feasibility; every generated pose still requires Newton.
    """
    rows = cells // HEIGHTMAP_COLS + row_shift[:, None]
    columns = cells % HEIGHTMAP_COLS + column_shift[:, None]
    inside = (rows >= 0) & (rows < HEIGHTMAP_ROWS) & (columns >= 0) & (columns < HEIGHTMAP_COLS)
    moved = rows.clamp(0, HEIGHTMAP_ROWS - 1) * HEIGHTMAP_COLS + columns.clamp(0, HEIGHTMAP_COLS - 1)
    heights = heightmap.flatten(1)
    same_height = torch.isclose(heights.gather(1, cells), heights.gather(1, moved), atol=1e-6, rtol=0)
    eligible = ((inside & same_height) | ~active).all(-1) & active.any(-1)
    eligible &= (row_shift != 0) | (column_shift != 0)
    return torch.where(eligible[:, None], moved, cells), eligible


def audit_intended_surfaces(points_world: Tensor, active: Tensor, observed: dict) -> Tensor:
    """Associate spatial intentions with upward Newton faces, outside forward.

    Nearest-plane association describes the intent only. It neither declares
    contact nor converts a side-face collision into a top-face collision.
    """
    if observed.get('schema') == 'newton_device_witness_batch_v1':
        from contact_solver.device_contact_objective import intended_surfaces
        return intended_surfaces(points_world, active, observed)
    from somaforge_core.contact_face_selection import upward_face_mask
    catalogs = observed.get("surface_catalog_by_sample")
    if catalogs is None:
        catalogs = [observed["surface_catalog"]] * len(points_world)
    if len(catalogs) != len(points_world):
        raise ValueError("Newton catalog batch mismatch")
    result = torch.full_like(active, -1, dtype=torch.long)
    for row, catalog in enumerate(catalogs):
        keep = upward_face_mask([face["normal_w"] for face in catalog])
        faces = [face for face, valid in zip(catalog, keep, strict=True) if valid]
        if not faces:
            raise ValueError("Newton catalog contains no valid horizontal primary faces")
        normal = points_world.new_tensor([face["normal_w"] for face in faces])
        offset = points_world.new_tensor([face["plane_offset"] for face in faces])
        distance = (points_world[row].detach() @ normal.T - offset).abs()
        ids = result.new_tensor([face["surface"] for face in faces])
        result[row] = torch.where(active[row], ids[distance.argmin(-1)], -1)
    return result
