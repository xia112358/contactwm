"""Legacy surface approach guidance, never contact or face-ownership truth.

Kept for old geometry-only callers. A zero cost does not establish contact.
The predictor's realized-contact objective is newton_witness_loss.
"""
import torch
from contact_solver.surface_geometry import nearest_face

SCHEMA='legacy_surface_approach_v1'


def primary_face_cost(points,parts,faces,margin):
    """Return per-part/face squared residual and diagnostics [B,6,F]."""
    nearest,_=nearest_face(points,faces)
    gap=(points[:,:,None]-nearest).square().sum(-1).clamp_min(1e-16).sqrt()
    costs=[];gaps=[];excesses=[];edges=[]
    for part in range(6):
        d=gap[:,parts==part]
        excess=(d-margin[:,part,None]).relu()
        # Legacy approach proxy only, NOT realized contact or face ownership.
        # Restore closest-face guidance; an entire limb need not lie above a
        # finite support. Actual contact losses use newton_witness_loss instead.
        cost=excess.square()
        index=d.argmin(1,keepdim=True)
        get=lambda a:a.gather(1,index)[:,0]
        costs.append(get(cost));gaps.append(get(d));excesses.append(get(excess))
        edges.append(torch.zeros_like(get(d)))
    return tuple(torch.stack(value,1) for value in (costs,gaps,excesses,edges))
