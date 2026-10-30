"""Historical position scalars for controlled comparisons only.

These intentionally retain the diagnosed frozen-witness gradient and point
attraction. Production uses contact_solver.native_contact_position; do not
use these diagnostic baselines as training or contact acceptance objectives.
"""
import torch
from somaforge_core.heightmap import HEIGHTMAP_RESOLUTION_M


def frozen_contact_layout_diagnostic(model, qpos, points, active, surfaces, rows, *, regions=None):
    """Reproduce the retired frozen-witness XY scalar for comparisons.

    Use fresh Newton geometric witnesses only. Missing pairs contribute no
    layout term and must be established by the separate Newton/finite-face
    approach objective; absence is never counted as successful contact.
    """
    from contact_solver.device_contact_objective import DeviceWitnessRows
    if isinstance(rows, DeviceWitnessRows):
        return rows.frozen_layout_diagnostic(points, active, surfaces, regions=regions)
    from contact_solver.contact_layout import (nearest_spatial_pairs,
        endpoint_position_statistics, endpoint_position_loss)
    active_cpu, surface_cpu = active.detach().cpu().tolist(), surfaces.detach().cpu().tolist()
    spatial_rows = [[dict(pair) for pair, _ in pairs] for pairs in rows]
    if regions is not None:
        all_pairs = [(sample, pair) for sample, pairs in enumerate(spatial_rows) for pair in pairs]
        if all_pairs:
            sample_ids = torch.tensor([s for s, _ in all_pairs], device=qpos.device)
            parts = torch.tensor([p['part'] for _, p in all_pairs], device=qpos.device)
            witnesses = qpos.new_tensor([p['position_w'] for _, p in all_pairs])
            ids = model.region_geometry.witness_regions(model.fk, qpos, sample_ids, parts, witnesses)
            for (_, pair), region in zip(all_pairs, ids.tolist(), strict=True):
                pair['contact_region'] = region
    selected = [nearest_spatial_pairs(pairs, enabled, face, points[sample].detach().cpu(),
                    regions=None if regions is None else regions[sample].detach().cpu())
                for sample, (pairs, enabled, face) in enumerate(zip(spatial_rows, active_cpu, surface_cpu, strict=True))]
    flat = [(sample, part, pair) for sample, row in enumerate(selected)
            for part, pair in enumerate(row) if pair is not None]
    moving = qpos[:, None, :3].expand(-1, 6, -1) * 0
    observed = torch.zeros_like(active)
    if flat:
        names = tuple(sorted({pair['body_name'] for _, _, pair in flat}))
        # A single batched FK instead of a full robot traversal for every limb.
        positions, rotations = model.fk.link_poses(qpos, names)
        lookup = {name: index for index, name in enumerate(names)}
        si = torch.tensor([sample for sample, _, _ in flat], device=qpos.device)
        pi = torch.tensor([part for _, part, _ in flat], device=qpos.device)
        bi = torch.tensor([lookup[pair['body_name']] for _, _, pair in flat], device=qpos.device)
        position, rotation = positions[si, bi], rotations[si, bi]
        witness = qpos.new_tensor([pair['position_w'] for _, _, pair in flat])
        material = torch.bmm(rotation.detach().transpose(1, 2), (witness - position.detach())[..., None])
        values = position + torch.bmm(rotation, material).squeeze(-1)
        moving = moving.index_put((si, pi), values)
        observed[si, pi] = True
    error, count = endpoint_position_statistics(moving, points.detach(), observed)
    return endpoint_position_loss(error), {
        'relative_layout_rms_cm': torch.where(count >= 1, 100 * error.clamp_min(1e-16).sqrt(), torch.zeros_like(error)),
        'relative_layout_observed_parts': count.to(qpos),
    }



def point_target_approach_diagnostic(model, qpos, points, active, configured_margin, *, cell_basis=None):
    """Reproduce the retired point-target approach for historical comparisons.

    This is an optimization residual, never a contact classifier. The normal
    is upward under this task's horizontal-primary-face contract. No semantic
    surface IDs, analytic scene, GT material witnesses, or q projection enter
    this residual. Newton supplies the actual margin and verifies realization.
    """
    from contact_solver.contact_constrained_projector import CanonicalContactCollisionGeometry

    if points.shape != (len(qpos), 6, 3) or active.shape != (len(qpos), 6):
        raise ValueError("spatial intentions must have shapes [B,6,3] and [B,6]")
    if configured_margin.shape != (len(qpos),) or not bool((configured_margin > 0).all()):
        raise ValueError("spatial approach requires actual positive Newton margins")
    geometry = getattr(model, "_spatial_plan_geometry", None)
    if geometry is None:
        geometry = CanonicalContactCollisionGeometry().to(qpos.device)
        model._spatial_plan_geometry = geometry
    points = points.detach()
    margin = configured_margin.detach()
    normal = qpos.new_tensor((0., 0., 1.)).expand(len(qpos), -1)
    tangent_u = (qpos.new_tensor((1., 0., 0.)).expand(len(qpos), -1)
                 if cell_basis is None else cell_basis.detach()[:, :, 0])
    tangent_v = (qpos.new_tensor((0., 1., 0.)).expand(len(qpos), -1)
                 if cell_basis is None else cell_basis.detach()[:, :, 1])
    residuals = []
    for part in range(6):
        # Cell extent is a spatial optimization tolerance, not contact truth.
        region = torch.cat((points[:, part], tangent_u, tangent_v,
                            qpos.new_full((len(qpos), 2), HEIGHTMAP_RESOLUTION_M / 2)), -1)
        residual = geometry.approach_residual(
            model.fk, qpos, part, normal, points[:, part, 2] + 0.05 * margin,
            region, points[:, part], qpos.new_zeros(len(qpos)), patch=False,
        )
        residuals.append(residual.square().sum(-1))
    error_sq = torch.stack(residuals, -1)
    normalized = error_sq / margin[:, None].square() * active
    count = active.sum(-1).clamp_min(1)
    loss = normalized.sum(-1) / count + 0.25 * normalized.amax(-1)
    return loss, {
        "spatial_plan_realization_loss": loss,
        "spatial_plan_residual_cm": 100 * (error_sq.clamp_min(1e-16).sqrt() * active).sum(-1) / count,
    }
