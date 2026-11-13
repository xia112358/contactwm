"""Conservative oriented bounds of the actual foot collision mesh.

The eight bounds corners share the pose specified by the existing material
pairs. SAT detects box overlap even when all corners remain outside terrain.
Reference overlap is subtracted only at the exact original target pose.
"""
import torch
from .contact_terrain_distance import SOLID_TRAVEL_COST

class FootOccupancy:
    def __init__(self,vertices,local,target,metric):
        self.metric=metric
        lo,hi=vertices.amin(0),vertices.amax(0)
        self.center=(lo+hi)/2;self.half=(hi-lo)/2
        signs=vertices.new_tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        self.corners=self.center+signs*self.half
        a,b=local.mean(0),target.mean(0)
        u,_,vh=torch.linalg.svd((local-a).T@(target-b))
        fix=vertices.new_ones(3);fix[-1]=torch.linalg.det(vh.T@u.T)
        self.goal_rotation=(vh.T*fix[None])@u.T
        self.goal_position=b-self.goal_rotation@a
        goals=self.goal_position+self.corners@self.goal_rotation.T
        self.goals=goals;self.prepared=metric.prepare(goals)
        self.reference_overlap=self.overlap(self.goal_position[None],self.goal_rotation[None])[0]

    def overlap(self,position,rotation):
        center=position+torch.einsum('bij,j->bi',rotation,self.center)
        values=[(-(center[:,2]-(rotation[:,2].abs()*self.half).sum(-1))).clamp_min(0)]
        axes=rotation.transpose(-1,-2)
        for c,r,h in zip(self.metric.centers,self.metric.rotations,self.metric.half):
            box_axes=r.T.expand(len(position),-1,-1)
            cross=torch.cross(axes[:,:,None,:].expand(-1,3,3,-1),box_axes[:,None,:,:].expand(-1,3,3,-1),dim=-1).flatten(1,2)
            candidate=torch.cat((axes,box_axes,cross),1)
            length=candidate.norm(dim=-1);valid=length>1e-8
            unit=candidate/length.clamp_min(1e-8)[...,None]
            foot_radius=((unit@rotation).abs()*self.half).sum(-1)
            box_radius=((unit@r).abs()*h).sum(-1)
            distance=((center-c)[:,None]*unit).sum(-1).abs()
            penetration=(foot_radius+box_radius-distance).masked_fill(~valid,float('inf')).amin(-1).clamp_min(0)
            values.append(penetration)
        return torch.stack(values,-1)

    def cost(self,position,rotation):
        points=position[:,None]+torch.einsum('bij,kj->bki',rotation,self.corners)
        routed=self.metric.distance(points,self.prepared)
        return self.cost_from_distances(position,rotation,points,routed)

    def cost_from_distances(self,position,rotation,points,routed):
        straight=(points-self.goals).norm(dim=-1)
        detour=(routed-straight).clamp_min(0).amax(-1)
        # Target solver penetration / conservative bound overlap is not a new goal.
        intrusion=(self.overlap(position,rotation)-self.reference_overlap).clamp_min(0).amax(-1)
        return (detour.square()+(SOLID_TRAVEL_COST*intrusion).square())/.05**2
