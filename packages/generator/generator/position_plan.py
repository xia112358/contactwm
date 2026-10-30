"""Hard contact-plan selection shared by versioned predictor architectures.

This changes neither demonstrated Newton labels nor teacher/repair semantics.
Only discrete intent leaves this module; execution never backpropagates to plan heads.
"""
from dataclasses import dataclass
import torch
from somaforge_core.heightmap import heightmap_grid
from generator.planned_contact_predictor import PlannedContactPrediction
from generator.role_validity import (mask_role_logits, conditional_location_logits,
    keep_location_support)


@dataclass(frozen=True)
class Full1000PositionPrediction(PlannedContactPrediction):
    teacher_conditioned: torch.Tensor
    region_logits: torch.Tensor | None = None
    planned_regions: torch.Tensor | None = None
    conditioned_role: torch.Tensor | None = None
    raw_role_logits: torch.Tensor | None = None
    raw_location_logits: torch.Tensor | None = None
    keep_location_allowed: torch.Tensor | None = None
    candidate_points_local: torch.Tensor | None = None
    conditioned_surface: torch.Tensor | None = None
    conditioned_point_id: torch.Tensor | None = None
    keep_uses_current_point: bool = False
    candidate_surfaces: torch.Tensor | None = None
    candidate_valid: torch.Tensor | None = None
    limb_role_logits: torch.Tensor | None = None

    @property
    def role(self):
        if self.limb_role_logits is not None:
            from generator.limb_contact import decode_limb_roles
            return decode_limb_roles(self.limb_role_logits)
        return self.role_logits.argmax(-1)

    def location_logits_for_roles(self, roles):
        raw = self.location_logits if self.raw_location_logits is None else self.raw_location_logits
        return conditional_location_logits(raw, roles, self.keep_location_allowed)


@dataclass(frozen=True)
class PositionPlan:
    role_logits: torch.Tensor
    location_logits: torch.Tensor
    contact_points_local: torch.Tensor
    conditioned_contact: torch.Tensor
    conditioned_cell: torch.Tensor
    conditioned_points_local: torch.Tensor
    teacher_conditioned: torch.Tensor
    region_logits: torch.Tensor | None
    planned_regions: torch.Tensor | None
    conditioned_role: torch.Tensor
    raw_role_logits: torch.Tensor | None = None
    raw_location_logits: torch.Tensor | None = None
    keep_location_allowed: torch.Tensor | None = None
    candidate_points_local: torch.Tensor | None = None
    conditioned_surface: torch.Tensor | None = None
    conditioned_point_id: torch.Tensor | None = None
    keep_uses_current_point: bool = False
    candidate_surfaces: torch.Tensor | None = None
    candidate_valid: torch.Tensor | None = None
    limb_role_logits: torch.Tensor | None = None

    def prediction(self, qpos):
        return Full1000PositionPrediction(qpos, self.role_logits, self.location_logits,
            self.contact_points_local, self.conditioned_contact, self.conditioned_cell,
            self.conditioned_points_local, self.teacher_conditioned, self.region_logits,
            self.planned_regions, self.conditioned_role, self.raw_role_logits,
            self.raw_location_logits, self.keep_location_allowed, self.candidate_points_local,
            self.conditioned_surface, self.conditioned_point_id, self.keep_uses_current_point,
            self.candidate_surfaces, self.candidate_valid, self.limb_role_logits)


def select_position_plan(model, current_q, heightmap, basis, part, *, current_contact,
        current_contact_relations=None, contact_surface_ids=None, heightmap_surface=None,
        contact_relation_points=None,
        contact_patch_points=None, contact_patch_part=None, contact_patch_surface=None, contact_patch_valid=None,
        scan_points_world=None, scan_surface=None, scan_valid=None, scan_point_id=None, scan_spacing_m=None,
        depth_image=None, depth_valid=None, depth_intrinsics=None, depth_camera_rotation_world=None,
        depth_camera_position_world=None, scan_pixel_index=None, image_features=None,
        teacher_contact=None, teacher_cell=None, teacher_mask=None, teacher_role=None, teacher_regions=None,
        repair_mask=None, repair_role=None, repair_points_world=None, repair_regions=None,
        repair_surface=None, repair_point_id=None):
    mesh = getattr(model, 'world_surface_scan', False)
    bound_keep = getattr(model, 'depth_observation', False)
    if mesh:
        if any(v is None for v in (scan_points_world, scan_surface, scan_valid, scan_point_id, scan_spacing_m)):
            raise ValueError('World mesh scan is missing; raster candidate fallback is forbidden')
        if (scan_points_world.ndim != 3 or scan_points_world.shape[0] != len(current_q)
                or scan_points_world.shape[-1] != 3 or scan_surface.shape != scan_points_world.shape[:2]
                or scan_valid.shape != scan_surface.shape or scan_point_id.shape != scan_surface.shape
                or scan_valid.dtype != torch.bool or scan_surface.dtype != torch.long
                or scan_point_id.dtype != torch.long or scan_spacing_m.shape != (len(current_q),)):
            raise ValueError('Invalid registered world mesh scan shapes/dtypes')
        if (not bool(torch.isfinite(scan_points_world).all()) or not bool(scan_valid.any(-1).all())
                or bool((scan_valid & ((scan_surface < 0) | (scan_point_id < 0))).any())
                or not bool(torch.isfinite(scan_spacing_m).all()) or bool((scan_spacing_m <= 0).any())):
            raise ValueError('Unknown/empty world mesh scan or missing surface/node identity')
        geometry = torch.einsum('bji,bnj->bni', basis, scan_points_world-current_q[:, None, :3])
        candidate_surface = scan_surface
    else:
        grid = torch.as_tensor(heightmap_grid(), device=heightmap.device, dtype=heightmap.dtype).reshape(-1, 2)
        geometry = torch.cat((grid[None].expand(len(current_q), -1, -1), heightmap.flatten(1)[..., None]), -1)
        candidate_surface = heightmap_surface
    # Hard plan heads remain supervised only by planning losses.
    raw_role = model.role_head(part)
    validity = getattr(model, 'role_validity', False)
    role = mask_role_logits(raw_role, current_contact) if validity else raw_role
    keep_allowed = (keep_location_support(current_contact, current_contact_relations,
        contact_surface_ids, candidate_surface) if validity else None)
    if mesh:
        keep_allowed &= scan_valid[:, None]
    if bound_keep:
        # Keep has no camera-location decision. Its world anchor is bound below.
        keep_allowed = None
    keys = model.location_key(geometry)
    if getattr(model, 'depth_observation', False):
        if image_features is None or depth_image is None:
            raise ValueError('Depth location selection requires encoded 2D image features')
        from generator.depth_observation import image_candidate_keys
        keys = keys + image_candidate_keys(model.location_scan_key, image_features, scan_pixel_index, depth_image)
    raw_location = torch.einsum('bpd,bnd->bpn', model.location_query(part), keys) / keys.shape[-1] ** .5
    if mesh:
        raw_location = raw_location.masked_fill(~scan_valid[:, None], -torch.inf)
    joint_roles = None
    conditioned_role = role.argmax(-1)
    if getattr(model, "limb_contact_modes", False):
        from generator.limb_contact import limb_logits, decode_limb_roles
        joint_roles = limb_logits(role)
        conditioned_role = decode_limb_roles(joint_roles)
    logits = conditional_location_logits(raw_location, conditioned_role, keep_allowed)
    contact, cell = conditioned_role != 0, logits.argmax(-1)
    predicted_points = geometry.gather(1, cell[..., None].expand(-1, -1, 3))
    if teacher_mask is None:
        teacher_mask = torch.zeros(len(current_q), dtype=torch.bool, device=current_q.device)
    if teacher_mask.shape != (len(current_q),) or teacher_mask.dtype != torch.bool:
        raise ValueError('teacher mask must be boolean [B]')
    if bool(teacher_mask.any()):
        if model.event_roles:
            if teacher_role is None or teacher_role.shape != contact.shape:
                raise ValueError('Event-role conditioning requires explicit teacher roles')
            if teacher_role.dtype != torch.long or bool(((teacher_role < 0) | (teacher_role > 3)).any()):
                raise ValueError('Teacher roles must be long values in [0,3]')
            if teacher_contact is None or not torch.equal((teacher_role != 0)[teacher_mask], teacher_contact[teacher_mask]):
                raise ValueError('Teacher role/contact mismatch')
            conditioned_role = torch.where(teacher_mask[:, None], teacher_role, conditioned_role)
        if teacher_contact is None or teacher_cell is None:
            raise ValueError('teacher samples require observed contact and cell labels')
        if teacher_contact.dtype != torch.bool or teacher_contact.shape != contact.shape or teacher_cell.shape != cell.shape:
            raise ValueError('teacher contact/cell shape mismatch')
        contact = torch.where(teacher_mask[:, None], teacher_contact, contact)
        teacher_locations = teacher_mask[:, None] & (~(conditioned_role == 2) if bound_keep else True)
        cell = torch.where(teacher_locations, teacher_cell, cell)
        if bool(((cell < 0) | (cell >= geometry.shape[1])).any()):
            raise ValueError('Teacher location index outside observed candidates')
        if mesh and bool((teacher_locations & contact & ~scan_valid.gather(1, cell)).any()):
            raise ValueError('Teacher location is not a scanned interaction surface')
        if validity:
            if bool(((conditioned_role == 2) & ~current_contact & teacher_mask[:, None]).any()):
                raise ValueError('Teacher keep requires actual current contact')
            legal_cells = conditional_location_logits(raw_location, conditioned_role, keep_allowed)
            if bool((teacher_locations & contact & ~torch.isfinite(
                    legal_cells.gather(-1, cell[..., None]).squeeze(-1))).any()):
                raise ValueError('Teacher keep requires the same observed contact surface')
    points = geometry.gather(1, cell[..., None].expand(-1, -1, 3))
    region_logits, planned_regions = None, None
    if model.region_plan:
        region_logits = (model.region_head(part)+model.region_prior).masked_fill(~model.region_geometry.valid_subsets, -1e4)
        subsets = model.region_geometry.subsets.to(part)
        hard = subsets[region_logits.argmax(-1)]
        planned_regions = hard.bool() & contact[..., None]
        if bool(teacher_mask.any()):
            if (teacher_regions is None or teacher_regions.dtype != torch.bool
                    or teacher_regions.shape != planned_regions.shape):
                raise ValueError('Regional teacher conditioning requires actual demonstrated regions [B,6,4]')
            selected = teacher_regions[teacher_mask]
            active = contact[teacher_mask]
            if bool((selected & ~model.region_geometry.valid[None]).any()):
                raise ValueError('Teacher regions contain an invalid anatomical region')
            if not torch.equal(selected.any(-1), active):
                raise ValueError('Teacher regions/contact mismatch or missing native regional evidence')
            planned_regions = torch.where(teacher_mask[:, None, None], teacher_regions, planned_regions)

    selected_surface = (scan_surface.gather(1, cell).masked_fill(~contact, -1) if mesh else None)
    selected_point_id = (scan_point_id.gather(1, cell).masked_fill(~contact, -1) if mesh else None)
    if repair_mask is not None:
        if not model.event_roles or repair_mask.dtype != torch.bool or repair_mask.shape != (len(current_q),):
            raise ValueError('Plan repair requires event roles and a boolean batch mask')
        if bool(repair_mask.any()):
            if repair_role is None or repair_points_world is None:
                raise ValueError('Repair requires the previously issued role and world points')
            if repair_role.shape != contact.shape or repair_role.dtype != torch.long or repair_points_world.shape != points.shape:
                raise ValueError('Invalid stored plan shapes/dtypes')
            if bool(((repair_role < 0) | (repair_role > 3)).any()) or not bool(torch.isfinite(repair_points_world).all()):
                raise ValueError('Invalid stored plan values')
            if validity and bool(repair_mask.any()):
                # Stored intent keeps its original surface and world point.
                # Re-observation may alter coverage, never relabel that intent.
                if mesh:
                    if (repair_surface is None or repair_point_id is None
                            or repair_surface.shape != contact.shape or repair_surface.dtype != torch.long
                            or repair_point_id.shape != contact.shape or repair_point_id.dtype != torch.long):
                        raise ValueError('Mesh repair requires stored surface and point identity')
                    active_repair = repair_mask[:, None] & (repair_role != 0)
                    if bound_keep:
                        active_repair &= repair_role != 2
                    if bool((active_repair & ((repair_surface < 0) | (repair_point_id < 0))).any()):
                        raise ValueError('Mesh repair has unknown stored surface/point identity')
                    allowed = scan_valid[:, None] & (scan_surface[:, None] == repair_surface[..., None])
                    distance = torch.cdist(repair_points_world, scan_points_world).square().masked_fill(
                        ~allowed, torch.inf)
                    error, repair_cells = distance.min(-1)
                    visible = torch.isfinite(error) & (error <= (2*scan_spacing_m[:, None]).square())
                    if bool((active_repair & ~visible).any()):
                        raise ValueError('Repair point is not covered by the current world mesh scan')
                    selected_surface = torch.where(repair_mask[:, None], repair_surface, selected_surface)
                    selected_point_id = torch.where(repair_mask[:, None], repair_point_id, selected_point_id)
                else:
                    from generator.contact_location_predictor import contact_points_to_observed_heightmap_map
                    _, repair_cells, visible = contact_points_to_observed_heightmap_map(
                        current_q, repair_role != 0, repair_points_world, heightmap)
                keep = (repair_role == 2) & repair_mask[:, None]
                if bound_keep:
                    keep = torch.zeros_like(keep)
                if bool((keep & (~current_contact | ~visible)).any()):
                    raise ValueError('Repair keep requires visible actual current contact')
                legal = (keep_allowed.gather(-1, repair_cells[..., None]).squeeze(-1)
                    if keep_allowed is not None else torch.ones_like(keep))
                if bool((keep & ~legal).any()):
                    raise ValueError('Repair keep requires the same observed contact surface')
            local = torch.einsum('bji,bpj->bpi', basis,
                repair_points_world.detach()-current_q[:, None, :3])
            conditioned_role = torch.where(repair_mask[:, None], repair_role, conditioned_role)
            contact = conditioned_role != 0
            points = torch.where(repair_mask[:, None, None], local, points)
            cell = torch.where(repair_mask[:, None], -1, cell)
            if model.region_plan:
                if repair_regions is None or repair_regions.shape != planned_regions.shape or repair_regions.dtype != torch.bool:
                    raise ValueError('Regional repair requires stored regional intent')
                planned_regions = torch.where(repair_mask[:, None, None], repair_regions, planned_regions)

    if joint_roles is not None:
        from generator.limb_contact import limb_targets
        limb_targets(conditioned_role)  # Also validate teacher and repair plans.
    if bound_keep:
        keep = conditioned_role == 2
        if bool(keep.any()):
            if contact_relation_points is None:
                raise ValueError('Keep requires actual current contact points')
            if bool((keep & ~current_contact).any()):
                raise ValueError('Keep requires actual current contact')
            # Canonical surface ordering, independent of location scores and scan coverage.
            ids = contact_surface_ids[:, None].expand_as(current_contact_relations)
            relation = ids.masked_fill(~current_contact_relations, torch.iinfo(ids.dtype).max).argmin(-1)
            anchor = contact_relation_points.gather(2, relation[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
            surface = contact_surface_ids.gather(1, relation)
            if not bool(torch.isfinite(anchor[keep]).all()):
                raise ValueError('Keep has nonfinite actual contact point')
            local = torch.einsum('bji,bpj->bpi', basis, anchor.detach()-current_q[:, None, :3])
            points = torch.where(keep[..., None], local, points)
            predicted_points = torch.where(keep[..., None], local, predicted_points)
            selected_surface = torch.where(keep, surface, selected_surface)
            # No scan pixel/node is selected for a held contact.
            cell = cell.masked_fill(keep, -1)
            selected_point_id = selected_point_id.masked_fill(keep, -1)

    if mesh:
        selected_surface = selected_surface.masked_fill(~contact, -1)
        selected_point_id = selected_point_id.masked_fill(~contact, -1)

    return PositionPlan(role, logits, predicted_points, contact, cell, points, teacher_mask,
                        region_logits, planned_regions, conditioned_role,
                        raw_role if validity else None, raw_location if validity else None, keep_allowed,
                        geometry if mesh else None, selected_surface, selected_point_id, bound_keep,
                        scan_surface if mesh else None, scan_valid if mesh else None, joint_roles)
