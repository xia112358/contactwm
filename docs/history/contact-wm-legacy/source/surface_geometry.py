"""Differentiable hand/plane placement proxy, never a contact detector.

Use the surface_v2 collision mesh without helper spheres. Newton's
contact point lies between the two surface witnesses; the plane midpoint is a
proxy for that convention. Multiple-contact averaging and solver timing still
require native validation. Do not use migration of this point as material slip.
"""
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch import nn
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.robot_assets import canonical_g1_urdf_path
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics


@lru_cache(maxsize=4)
def hand_collision_support(side, directions, collision_profile='surface_v2'):
    from somaforge_core.robot_assets import canonical_g1_asset_metadata
    from somaforge_core.collision_profiles import SURFACE
    if collision_profile != SURFACE:
        raise ValueError("Contact WM hand geometry requires surface_v2")
    asset=canonical_g1_asset_metadata(collision_profile=collision_profile)
    path=canonical_g1_urdf_path();root=ET.parse(path).getroot()
    collision=root.find(f"link[@name='{side}_sphere_hand_link']/collision")
    origin=collision.find('origin')
    if origin is not None and any(float(v) for key in ('xyz','rpy') for v in origin.get(key,'0 0 0').split()):
        raise ValueError('Unsupported hand collision origin')
    mesh=collision.find('geometry/mesh')
    vertices=np.unique(np.array([[float(v) for v in line.split()[1:4]]
        for line in (path.parent/mesh.get('filename')).read_text().splitlines() if line.startswith('v ')]),axis=0)
    vertices*=np.array([float(v) for v in mesh.get('scale','1 1 1').split()])
    # Uniform directional support samples avoid dependence on mesh tessellation
    # density and keep batched BPTT memory bounded. No recorded contacts are fit.
    i=np.arange(directions);z=1-2*(i+.5)/directions;angle=i*np.pi*(3-np.sqrt(5))
    radial=np.sqrt(1-z*z);normal=np.stack((radial*np.cos(angle),radial*np.sin(angle),z),-1)
    ids=np.unique(np.concatenate([(vertices@chunk.T).argmin(0) for chunk in np.array_split(normal,8)]))
    return vertices[ids].astype(np.float32),asset


class HandPlaneContact(nn.Module):
    """Smooth support point against a supplied locally planar target surface."""
    def __init__(self,joint_names,side='left',temperature=.0002,directions=512,*,collision_profile='surface_v2'):
        super().__init__()
        if side not in ('left','right') or temperature<=0 or directions<32:raise ValueError('Invalid hand support configuration')
        if len(joint_names)!=29 or set(joint_names)!=set(G1_29DOF_JOINT_ORDER):raise ValueError('Canonical joint order required')
        vertices,asset=hand_collision_support(side,directions,collision_profile)
        self.asset=asset;self.temperature=temperature
        self.links=(f'{side}_sphere_hand_link',)
        self.fk=BatchedG1ForwardKinematics()
        self.register_buffer('vertices',torch.tensor(vertices))
        self.register_buffer('order',torch.tensor([joint_names.index(n) for n in G1_29DOF_JOINT_ORDER]))

    def forward(self,state,normal,plane_point):
        # Inputs are flattened Bx71, Bx3, Bx3; normals are unit world normals.
        q=state[:,3:7]
        qpos=torch.cat((state[:,:3],q[:,[3,0,1,2]],state[:,13:42].index_select(-1,self.order)),-1)
        pos,rot=self.fk.link_poses(qpos,self.links)
        local_normal=(rot[:,0].transpose(-1,-2)@normal[...,None]).squeeze(-1)
        distances=local_normal@self.vertices.T
        weights=torch.softmax(-distances/self.temperature,-1)
        local=weights@self.vertices
        robot=pos[:,0]+(rot[:,0]@local[...,None]).squeeze(-1)
        gap=((robot-plane_point)*normal).sum(-1,keepdim=True)
        return robot-.5*gap*normal
