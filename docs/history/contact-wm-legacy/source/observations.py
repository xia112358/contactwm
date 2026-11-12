"""Differentiable PHP input reconstruction without future reference motions."""
import torch
from torch import nn
from torch.nn import functional as F


def rotation_xyzw(q):
    q=F.normalize(q,dim=-1);x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def yaw_local(points,q):
    x,y,z,w=q.unbind(-1);angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    while angle.ndim<points.ndim-1:angle=angle.unsqueeze(-1)
    px,py,pz=points.unbind(-1)
    return torch.stack((angle.cos()*px+angle.sin()*py,-angle.sin()*px+angle.cos()*py,pz),-1)


def unpack_history(flat):
    """Manager stores each alphabetical term's five-frame history contiguously."""
    widths=(29,3,29,29,3)
    if flat.shape[-1]!=465:raise ValueError('PHP history must have 465 entries')
    return torch.cat([chunk.reshape(*flat.shape[:-1],5,w) for chunk,w in
                      zip(flat.split([5*w for w in widths],-1),widths)],-1)


def pack_history(history):
    if history.shape[-2:]!=(5,93):raise ValueError('Expected five steps of PHP proprioception')
    return torch.cat([x.flatten(-2) for x in history.split((29,3,29,29,3),-1)],-1)


class PHPObservation(nn.Module):
    """Keep published ordering/scales. Height reader must use available perception.

    height_reader(state) returns 63 relative heights, recomputed at the imagined
    pose using an observed static map. There is no default ground-truth shortcut.
    goal_hint is the explicit high-level 3-vector, not a reference-motion window.
    """
    def __init__(self,default_joint_position,height_reader,goal_hint,known_contact):
        super().__init__()
        self.register_buffer('default_joint_position',default_joint_position)
        self.height_reader=height_reader
        self.register_buffer('goal_hint',goal_hint)
        self.register_buffer('known_contact',known_contact)

    def forward(self,state,contact,history,previous_action,goal,step):
        R=rotation_xyzw(state[:,3:7]);Rt=R.transpose(-1,-2)
        angular=(Rt@state[:,10:13,None]).squeeze(-1)
        gravity=-R[:,2,:]
        proprio=torch.cat((previous_action,angular,state[:,13:42]-self.default_joint_position,
                           state[:,42:71],gravity),-1)
        history=torch.cat((history[:,:-1] if step==0 else history[:,1:],proprio[:,None]),1)
        active=goal.establish|goal.maintain
        target=torch.where(goal.maintain[...,None],goal.support_position_w,goal.position_w)
        local=yaw_local(target-state[:,:3,None].transpose(1,2),state[:,3:7])
        command=torch.cat((torch.where(active[...,None],local,0.).flatten(1),active.float()),-1)
        heights=self.height_reader(state)
        if heights.shape!=(len(state),63):raise ValueError('Expected measured 9x7 height reader')
        obs=torch.cat((contact,self.known_contact.float(),command,self.goal_hint,heights,pack_history(history)),-1)
        normal=yaw_local(goal.normal_w,state[:,3:7])
        # Relative arrival time, zero after the requested arrival; hold enforced by loss.
        deadline=state.new_full((len(state),1),max(0,goal.arrival_step-step)*.02)
        condition=torch.cat((normal.flatten(1),goal.maintain.float(),deadline),-1)
        return obs,condition,history
