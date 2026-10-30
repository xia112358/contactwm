"""Differentiable fixed robot-side witness points on the canonical G1."""
import torch
from torch import nn
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics

LINKS=('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link',
       'right_wrist_yaw_link','left_knee_link','right_knee_link')


class WitnessGeometry(nn.Module):
    def __init__(self,joint_names,local_witness):
        super().__init__()
        if len(joint_names)!=29 or set(joint_names)!=set(G1_29DOF_JOINT_ORDER):
            raise ValueError('Canonical 29-joint names required')
        if local_witness.shape[-2:]!=(6,3):raise ValueError('Six robot-local witness points required')
        self.fk=BatchedG1ForwardKinematics()
        self.register_buffer('order',torch.tensor([joint_names.index(n) for n in G1_29DOF_JOINT_ORDER]))
        self.register_buffer('local_witness',local_witness)

    def forward(self,state):
        q=state[...,3:7]  # physical runtime xyzw -> canonical FK wxyz
        qpos=torch.cat((state[...,:3],q[...,[3,0,1,2]],state[...,13:42].index_select(-1,self.order)),-1)
        position,rotation=self.fk.link_poses(qpos,LINKS)
        return position+(rotation@self.local_witness[...,None]).squeeze(-1)
