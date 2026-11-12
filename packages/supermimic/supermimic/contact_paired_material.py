"""Material candidates and their paired world goals for stable native contact."""
import torch
from .contact_material import world_points

CONTRACT = dict(version="paired_material_candidates_v1", selection="same native stable row chooses its own local material and world target", completion="a physically matched candidate must reach its paired target")


def matched_material_index(witness,position,rotation,local,valid):
    """Identify the candidate nearest the same-solve positive-load contact row."""
    measured=(rotation.transpose(-1,-2)@(witness-position).unsqueeze(-1)).squeeze(-1)
    distance=(local-measured.unsqueeze(-2)).square().sum(-1)
    return distance.masked_fill(~valid,float('inf')).argmin(-1)


def paired_material_error_sq(position,rotation,witness,stable,local,target,valid,candidate_index=None):
    """Choose a matching material/goal pair; guidance may choose the nearest pair.

    The stable branch selects material by the actual native row in local space;
    it never independently minimizes over the world targets. For unconfirmed
    guidance only, the pair with the smallest fixed-point error is used.
    """
    actual=world_points(position,rotation,local)
    cost=(actual-target).square().sum(-1).masked_fill(~valid,float('inf'))
    matched=(matched_material_index(witness,position,rotation,local,valid) if candidate_index is None else candidate_index.long())
    guided=cost.detach().argmin(-1)
    chosen=torch.where(stable,matched,guided)
    return cost.gather(-1,chosen.unsqueeze(-1)).squeeze(-1),chosen


def paired_raw_guidance_error_sq(position,rotation,raw_point,local,target,valid):
    """Guide one complete material/goal pair toward a real stable native row.

    This is a loss term only: the raw row does not become a completion witness.
    Position, rotation and raw point must be from the same solver substep.
    """
    actual=world_points(position,rotation,local)
    goal=(actual-target).square().sum(-1)
    correspondence=(actual-raw_point.unsqueeze(-2)).square().sum(-1)
    cost=(goal+correspondence).masked_fill(~valid,float('inf'))
    return cost.amin(-1)
