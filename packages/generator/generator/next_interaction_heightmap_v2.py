"""Root-yaw height-map predictor with explicit part-to-space attention."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from generator.next_interaction import NextInteraction
from somaforge_core.heightmap import HEIGHTMAP_CLIP_M, HEIGHTMAP_COLS, HEIGHTMAP_RESOLUTION_M, HEIGHTMAP_ROWS, heightmap_grid
from generator.next_interaction_heightmap import HeightmapEncoder
from contact_solver.collision_geometry import CanonicalG1CollisionPoints
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics, _matrix_from_rotation6d, _quaternion_matrix_wxyz, _quaternion_multiply_wxyz, _rotation6d


from somaforge_core.heightmap import _numpy_root_yaw_basis


from somaforge_core.heightmap import render_root_yaw_box_heightmaps


from somaforge_core.heightmap import heightmap_supersample_for_architecture


from somaforge_core.heightmap import _DEVICE_HEIGHTMAP_GRIDS


from somaforge_core.heightmap import render_root_yaw_box_heightmaps_device


from somaforge_core.heightmap import _root_yaw_basis


class RootYawHeightmapInteractionPredictor(nn.Module):
    """One-pass joint contact/pose predictor using observed spatial queries."""
    def __init__(self,width: int=192,layers: int=3):
        super().__init__();self.fk=CanonicalG1ForwardKinematics();self.geometry=CanonicalG1CollisionPoints(64)
        self.height=HeightmapEncoder(width);self.global_encoder=nn.Linear(38,width);self.part_encoder=nn.Linear(13,width)
        self.part_identity=nn.Embedding(6,width);self.surface_embedding=nn.Embedding(3,width)
        self.part_to_terrain=nn.MultiheadAttention(width,6,batch_first=True)
        layer=nn.TransformerEncoderLayer(width,6,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.shared=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.norm=nn.LayerNorm(width)
        self.role_head=nn.Linear(width,4);self.surface_head=nn.Linear(width,2)
        self.role_embedding=nn.Linear(4,width,bias=False);self.predicted_surface_embedding=nn.Linear(2,width,bias=False)
        interaction=nn.TransformerEncoderLayer(width,6,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.interaction_decoder=nn.TransformerEncoder(interaction,1,enable_nested_tensor=False)
        self.pose_head=nn.Linear(width,36);self.duration_head=nn.Linear(width,1)
        nn.init.normal_(self.pose_head.weight,std=.001);nn.init.zeros_(self.pose_head.bias)
        with torch.no_grad():self.pose_head.bias[3]=1.

    @staticmethod
    def _to_local(points,q,basis):
        return torch.einsum('bij,bpj->bpi',basis.transpose(1,2),points-q[:,None,:3])

    @staticmethod
    def _local_state(q,basis):
        local_rotation=basis.transpose(1,2)@_quaternion_matrix_wxyz(q[:,3:7])
        return torch.cat((torch.zeros_like(q[:,:3]),_rotation6d(local_rotation),q[:,7:]/math.pi),-1)

    @staticmethod
    def _decode_pose(raw,current_q,basis,yaw_quaternion):
        world_delta=torch.einsum('bij,bj->bi',basis,raw[:,:3])
        world_quaternion=_quaternion_multiply_wxyz(yaw_quaternion,F.normalize(raw[:,3:7],dim=-1))
        return torch.cat((current_q[:,:3]+world_delta,world_quaternion,raw[:,7:]),-1)

    def encode_pose(self,target_q,current_q):
        basis,yaw_quaternion=_root_yaw_basis(current_q)
        local_delta=torch.einsum('bij,bj->bi',basis.transpose(1,2),target_q[:,:3]-current_q[:,:3])
        inverse_yaw=yaw_quaternion.clone();inverse_yaw[:,1:]*=-1
        local_quaternion=_quaternion_multiply_wxyz(inverse_yaw,target_q[:,3:7])
        return torch.cat((local_delta,local_quaternion,target_q[:,7:]),-1)

    def decode_pose(self,raw,current_q):
        basis,yaw_quaternion=_root_yaw_basis(current_q)
        return self._decode_pose(raw,current_q,basis,yaw_quaternion)

    def forward(self,current_q,current_contact,current_anchor,current_surface,heightmap):
        terrain,_=self.height(heightmap);basis,yaw_quaternion=_root_yaw_basis(current_q)
        positions,rotations6d=self.fk(current_q[:,None]);positions=positions[:,0,1:7]
        rotations=_matrix_from_rotation6d(rotations6d[:,0,1:7])
        local_position=self._to_local(positions,current_q,basis)
        local_rotation=basis[:,None].transpose(-1,-2)@rotations
        local_anchor=self._to_local(current_anchor,current_q,basis)
        surface_index=torch.where(current_contact,current_surface.clamp(0,1),2)
        part=self.part_encoder(torch.cat((local_position,_rotation6d(local_rotation),
            local_anchor*current_contact[...,None],current_contact[...,None].float()),-1))
        part=part+self.part_identity.weight[None]+self.surface_embedding(surface_index)
        spatial,_=self.part_to_terrain(part,terrain,terrain,need_weights=False);part=self.norm(part+spatial)
        state=self.global_encoder(self._local_state(current_q,basis))[:,None]
        tokens=self.shared(torch.cat((state,part),1));body=self.norm(tokens[:,0]+tokens[:,1:].mean(1));part=self.norm(tokens[:,1:])
        role_logits,surface_logits=self.role_head(part),self.surface_head(part)
        contact=part+self.role_embedding(role_logits.softmax(-1).detach())
        contact=contact+self.predicted_surface_embedding(surface_logits.softmax(-1).detach())
        interaction=self.interaction_decoder(torch.cat((body[:,None],contact),1))
        shared=self.norm(interaction[:,0]+interaction[:,1:].mean(1))
        q=self._decode_pose(self.pose_head(shared),current_q,basis,yaw_quaternion)
        return NextInteraction(q,role_logits,surface_logits,F.softplus(self.duration_head(shared)[:,0]))


def objective(model,prediction,target,scene,**kwargs):
    from generator.next_interaction_surface import objective as newton_objective
    return newton_objective(model,prediction,target,scene,**kwargs)
