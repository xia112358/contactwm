"""Normalized floating-base pose chart: world XYZ, WXYZ quaternion, joints.

Scale and reference are supplied by the caller; no robot or task is assumed.
The rotation logarithm uses the shortest branch and is local near pi.
"""
import math
import torch

def multiply(a,b):
    return torch.cat(((a[0]*b[0]-a[1:]@b[1:]).reshape(1),a[0]*b[1:]+b[0]*a[1:]+torch.cross(a[1:],b[1:],dim=0)))


def coordinates(q,reference,scale):
    a=q[3:7]/q[3:7].norm();b=reference[3:7]/reference[3:7].norm()
    rel=multiply(a,torch.cat((b[:1],-b[1:])))
    rel=torch.where(rel[0]<0,-rel,rel)
    length=rel[1:].norm()
    factor=torch.where(length<1e-7,2/rel[0].clamp_min(1e-12),2*torch.atan2(length,rel[0])/length.clamp_min(1e-12))
    return torch.cat((q[:3]-reference[:3],factor*rel[1:],q[7:]-reference[7:]))/scale


def pose(z,reference,scale):
    d=z*scale;rot=d[3:6];theta=rot.norm()
    a=torch.cat((torch.cos(theta/2).reshape(1),.5*torch.sinc(theta/(2*math.pi))*rot))
    quat=multiply(a,reference[3:7]);quat=quat/quat.norm()
    return torch.cat((reference[:3]+d[:3],quat,reference[7:]+d[6:]))


