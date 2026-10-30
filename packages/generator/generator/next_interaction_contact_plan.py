"""Experimental explicit contact-plan bottleneck, not a constrained IK solver.

Only observation inputs reach forward(). A plan is an intention, never an
achieved Newton contact. The pose decoder has no access to planner latents.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from generator.next_interaction import NextInteraction, NextInteractionPredictor

SCHEMA='explicit_contact_plan_pose_v1_experimental'


def hard_choice(logits):
    probability=logits.softmax(-1)
    hard=F.one_hot(probability.argmax(-1),probability.shape[-1]).to(probability.dtype)
    # Forward selects ONE face/role. The soft derivative is a training estimator,
    # not a forward average of incompatible geometric surfaces.
    return hard+probability-probability.detach()


@dataclass
class ContactPlan:
    role_weights: torch.Tensor
    surface_weights: torch.Tensor
    point_w: torch.Tensor
    normal_w: torch.Tensor
    tangent_w: torch.Tensor

    @property
    def active(self):return self.role_weights[...,1:].sum(-1)


@dataclass
class PlannedInteraction(NextInteraction):
    plan: ContactPlan


class ContactPlanPredictor(NextInteractionPredictor):
    def __init__(self,width=192,layers=3):
        super().__init__(width,layers)
        self.plan_point_head=nn.Linear(width,2)
        self.plan_heading_head=nn.Linear(width,2)
        self.current_pose_encoder=nn.Linear(36,width)
        self.plan_encoder=nn.Sequential(nn.Linear(13,width),nn.GELU(),nn.Linear(width,width))
        self.plan_decoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width,6,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True),
            2,enable_nested_tensor=False)

    def make_plan(self,part,faces,roles,surfaces):
        selected=hard_choice(surfaces)
        face=torch.einsum('bps,bsd->bpd',selected,faces)
        coordinates=self.plan_point_head(part)
        # Bounds describe actual observed faces, not a cap on body motion.
        uv=torch.where(face[...,14:15]>.5,coordinates,coordinates.tanh()*face[...,12:14])
        point=face[...,:3]+uv[...,:1]*face[...,6:9]+uv[...,1:2]*face[...,9:12]
        heading=self.plan_heading_head(part)
        heading=F.normalize(heading,dim=-1,eps=1e-8)
        fallback=torch.zeros_like(heading);fallback[...,0]=1
        heading=torch.where((heading.norm(dim=-1,keepdim=True)>1e-7),heading,fallback)
        tangent=heading[...,:1]*face[...,6:9]+heading[...,1:2]*face[...,9:12]
        return ContactPlan(hard_choice(roles),selected,point,face[...,3:6],tangent)

    def decode_plan(self,current_q,plan):
        if plan.point_w.shape!=(len(current_q),6,3):raise ValueError('Expected six contact parts')
        active=plan.active[...,None]
        # Inactive parts do not accidentally constrain a surface or location.
        features=torch.cat((plan.role_weights,active*(plan.point_w-current_q[:,None,:3]),
                            active*plan.normal_w,active*plan.tangent_w),-1)
        token=self.plan_encoder(features)+self.part_identity.weight[None]
        state=self.current_pose_encoder(current_q)[:,None]
        decoded=self.plan_decoder(torch.cat((state,token),1))
        hidden=self.norm(decoded[:,0]+decoded[:,1:].mean(1))
        return self.decode_pose(self.pose_head(hidden),current_q),F.softplus(self.duration_head(hidden)[:,0])

    def decode_interaction(self,body,part,face,current_q,faces):
        roles=self.role_head(part)
        surfaces=torch.einsum('bpw,bsw->bps',self.surface_query(part),self.surface_key(face))/part.shape[-1]**.5
        plan=self.make_plan(part,faces,roles,surfaces)
        q,duration=self.decode_plan(current_q,plan)
        return PlannedInteraction(q,roles,surfaces,duration,plan)


def plan_demonstration_loss(prediction,target):
    """Plan supervision only. Does not certify achieved contact or collision-free q.

    Existing Newton contact/penetration checks must still evaluate the resulting
    pose. Target q/material_local never enter forward or decode_plan.
    """
    active=(target['role']!=0).to(prediction.qpos.dtype)
    point=(prediction.plan.point_w-target['anchor']).square().sum(-1)/.02**2
    loss=(point*active).sum(-1)/active.sum(-1).clamp_min(1)
    return loss
