"""DAgger endpoint labels: verified solver outputs, never proposed witnesses.

Endpoint feasibility does not certify the path from the learner state.
"""
import numpy as np
import torch
from contact_solver.strict_nonpenetration import clearance_status
from somaforge_core.g1_kinematics import _matrix_from_rotation6d

SCHEMA = 'dagger_verified_next_contact_endpoint_v1'


def endpoint_gate(q, active, surface, actual, separation, lower, upper):
    q = np.asarray(q)
    if q.shape != (36,) or not np.isfinite(q).all():
        return False
    if clearance_status(separation) != 'clear':
        return False
    if np.any(q[7:] < lower) or np.any(q[7:] > upper):
        return False
    observed = np.asarray(actual['contact_part_mask'], bool)
    observed_surface = np.asarray(actual['contact_surface'])
    active = np.asarray(active, bool)
    return bool(np.array_equal(observed, active)
                and np.array_equal(observed_surface[active], np.asarray(surface)[active]))


def recovery_roles(active, surface, target_active, target_surface, demonstrated_roles):
    """Absent current contact cannot be a persistent target; no input relabeling."""
    same = active & (surface == target_surface)
    held = (demonstrated_roles == 2) & same & target_active
    touch = target_active & ((demonstrated_roles == 1) | ~same)
    role = torch.where(target_active, 3, 0)
    role = torch.where(held, 2, role)
    return torch.where(touch, 1, role)


@torch.no_grad()
def verified_target(model, template, solved_q, realized_anchor):
    """Replace ALL pose-derived supervision with the exact accepted solution."""
    result = {k: v.detach().clone() for k, v in template.items()}
    result['q'] = solved_q.detach().clone()
    result['anchor'] = realized_anchor.detach().clone()
    position, rotation = model.fk(solved_q[:, None])
    result['body_position'] = position[:, 0]
    rotations = _matrix_from_rotation6d(rotation[:, 0])[:, 1:7]
    result['material_local'] = torch.einsum(
        'bpji,bpj->bpi', rotations, realized_anchor-position[:, 0, 1:7])
    return result
