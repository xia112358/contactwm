"""Preserved experiment helpers for regression tests; no training CLI."""
import torch
from contact_solver.collision_geometry import full_geometry_box_ground_penetration

def penetration(points, parts, contact, *, box_center, box_rotation, box_half_extents, ground_height, margin_m=0.0, planned_contact_tolerance_m=0.0):
    if margin_m != 0.0 or planned_contact_tolerance_m != 0.0:
        return full_geometry_box_ground_penetration(points, parts, contact, box_center=box_center, box_rotation=box_rotation, box_half_extents=box_half_extents, ground_height=ground_height, margin_m=margin_m, planned_contact_tolerance_m=planned_contact_tolerance_m)
    if points.ndim != 4 or points.shape[-1] != 3 or contact.shape != (*points.shape[:2], 6):
        raise ValueError('Expected points [B,T,P,3] and contacts [B,T,6]')
    local = torch.einsum('btpji,btpj->btpi', box_rotation[:, None, None], points - box_center[:, None, None])
    box = (-(local.abs() - box_half_extents[:, None, None]).amax(-1)).relu()
    ground = (-(points[..., 2] - ground_height[:, None, None])).relu()
    return (box, ground)
