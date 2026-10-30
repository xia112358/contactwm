from __future__ import annotations
import torch
from contact_solver.endpoint_task import JOINT_LIMIT_SCALE_RAD


def joint_feasibility_loss(joints, lower, upper):
    """Soft physical-limit cost; 0.1 rad is a loss scale, not an output cap.

    The maximum term prevents a single invalid joint being diluted across DOFs.
    """
    violation=(lower-joints).relu()+(joints-upper).relu()
    normalized=(violation/JOINT_LIMIT_SCALE_RAD).square()
    return normalized.mean(-1)+normalized.amax(-1),violation


def nearest_face(points, faces):
    """Closest points on observed rectangular faces or infinite ground planes.

    faces: center3, normal3, edge-axis-u3, edge-axis-v3, half-extents2,
    infinite-plane flag. Rectangle axes follow actual mesh edges, not XY bounds.
    Returns [B,P,F,3] points and signed normal distance. Not contact truth.
    """
    delta=points[:,:,None]-faces[:,None,:,:3]
    u=(delta*faces[:,None,:,6:9]).sum(-1)
    v=(delta*faces[:,None,:,9:12]).sum(-1)
    unlimited=faces[:,None,:,14]>0.5
    u=torch.where(unlimited,u,torch.maximum(torch.minimum(u,faces[:,None,:,12]),-faces[:,None,:,12]))
    v=torch.where(unlimited,v,torch.maximum(torch.minimum(v,faces[:,None,:,13]),-faces[:,None,:,13]))
    nearest=faces[:,None,:,:3]+u[...,None]*faces[:,None,:,6:9]+v[...,None]*faces[:,None,:,9:12]
    return nearest,(delta*faces[:,None,:,3:6]).sum(-1)

