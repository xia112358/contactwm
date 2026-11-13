"""Continuous weighted path distance through terrain, with costly solid travel.

This is an optimization distance, never a native contact detector. Interior travel approaches SOLID_TRAVEL_COST with depth; the boundary
has zero extra density and normal derivative. Penetrated queries have a
finite escape objective. Original targets, including solver penetration, stay
unchanged. This finite metric does not claim an impenetrable path guarantee.
"""
import itertools
import torch
from scipy.sparse.csgraph import floyd_warshall
from .rotations import quat_to_mat

SOLID_TRAVEL_COST = 10.
SURFACE_COST_SCALE = .05

class TerrainNavigationDistance:
    def __init__(self, boxes, device='cpu'):
        boxes=torch.as_tensor(boxes,dtype=torch.float32,device=device).reshape(-1,10)
        self.centers=boxes[:,:3];self.rotations=quat_to_mat(boxes[:,3:7]);self.half=boxes[:,7:]*.5
        if not len(boxes):
            self.nodes=boxes.new_empty((0,3));self.costs=boxes.new_empty((0,0));return
        coordinates=torch.tensor([x for x in itertools.product((-1.,0.,1.),repeat=3) if max(map(abs,x))==1],device=device)
        nodes=[]
        for c,r,h in zip(self.centers,self.rotations,self.half):
            world=(coordinates*h)@r.T+c;nodes.append(world)
            nodes.append(torch.cat((world[:,:2],torch.zeros_like(world[:,2:])),dim=-1))
        self.nodes=torch.cat(nodes).unique(dim=0)
        edges=self.segment_cost(self.nodes[:,None],self.nodes[None])
        self.costs=torch.as_tensor(floyd_warshall(edges.cpu().double().numpy(),directed=False),dtype=torch.float32,device=device)

    def segment_cost(self,start,end):
        """Midpoint integration of depth-weighted cost over solid intervals."""
        start,end=torch.broadcast_tensors(start,end);delta=end-start
        intervals=[]
        for c,r,h in zip(self.centers,self.rotations,self.half):
            p=(start-c)@r;v=delta@r;parallel=v.abs()<1e-10
            denominator=torch.where(parallel,torch.ones_like(v),v)
            a=(-h-p)/denominator;b=(h-p)/denominator
            lo=torch.where(parallel,torch.full_like(a,-torch.inf),torch.minimum(a,b)).amax(-1).clamp(0,1)
            hi=torch.where(parallel,torch.full_like(a,torch.inf),torch.maximum(a,b)).amin(-1).clamp(0,1)
            boundary_outside=p.abs()>=h
            # The bottom face of a grounded box is not an underpass.
            grounded=c[2]-(r[2].abs()*h).sum()<=0
            boundary_outside=torch.stack((boundary_outside[...,0],boundary_outside[...,1],boundary_outside[...,2]&~(grounded&(p[...,2]<0))),-1)
            outside=(parallel&boundary_outside).any(-1)
            hi=torch.where(outside,lo,torch.maximum(lo,hi));intervals.append((lo,hi))
        dz=delta[...,2];parallel=dz.abs()<1e-10
        crossing=(-start[...,2]/torch.where(parallel,torch.ones_like(dz),dz)).clamp(0,1)
        lo=torch.where(dz<0,crossing,torch.zeros_like(dz));hi=torch.where(dz<0,torch.ones_like(dz),crossing)
        lo=torch.where(parallel,torch.zeros_like(lo),lo)
        hi=torch.where(parallel,(start[...,2]<0).to(hi.dtype),hi);intervals.append((lo,hi))
        starts=torch.stack([x[0] for x in intervals],-1);stops=torch.stack([x[1] for x in intervals],-1)
        boundaries=torch.cat((starts,stops),-1).sort(-1).values
        mid=(boundaries[...,1:]+boundaries[...,:-1])*.5
        occupied=((mid[...,None]>=starts[...,None,:])&(mid[...,None]<stops[...,None,:])).any(-1)
        samples=start[...,None,:]+mid[...,None]*delta[...,None,:]
        depth=(-samples[...,2]).clamp_min(0)
        for c,r,h in zip(self.centers,self.rotations,self.half):
            local=(samples-c)@r
            faces=h-local.abs()
            grounded=c[2]-(r[2].abs()*h).sum()<=0
            # A grounded box has no free lower face along the floor.
            z=torch.where(grounded&(local[...,2]<0),h[2]-local[...,2],faces[...,2])
            interior=torch.minimum(faces[...,:2].amin(-1),z).clamp_min(0)
            depth=torch.maximum(depth,interior)
        density=(depth/(depth+SURFACE_COST_SCALE)).square()
        fraction=((boundaries[...,1:]-boundaries[...,:-1])*occupied*density).sum(-1)
        return delta.norm(dim=-1)*(1+(SOLID_TRAVEL_COST-1)*fraction)

    def exits(self,points):
        """All face projections, without selecting or detaching a nearest face."""
        result=[points,torch.cat((points[...,:2],torch.zeros_like(points[...,2:])),dim=-1)]
        for c,r,h in zip(self.centers,self.rotations,self.half):
            p=(points-c)@r
            for axis in range(3):
                for sign in (-1.,1.):
                    q=torch.stack([(sign*h[axis]).expand_as(p[...,axis]) if k==axis else p[...,k] for k in range(3)],-1)
                    result.append(q@r.T+c)
        return torch.stack(result,-2)

    def prepare(self,targets):
        targets=targets.to(device=self.centers.device,dtype=torch.float32)
        exits=self.exits(targets)
        ending=self.segment_cost(exits,targets[...,None,:])
        if not len(self.nodes):return targets,exits,ending,None
        edges=(self.segment_cost(self.nodes[:,None,:],exits[...,None,:,:])+ending[...,None,:]).amin(-1)
        tail=(self.costs+edges[...,None,:]).amin(-1)
        return targets,exits,ending,tail

    def distance(self,points,prepared):
        target,goal_exits,ending,tail=prepared
        exits=self.exits(points)
        prefix=self.segment_cost(points[...,None,:],exits)
        if len(self.nodes):
            destinations=torch.cat((goal_exits,self.nodes.expand(*goal_exits.shape[:-2],-1,3)),-2)
            routes=self.segment_cost(exits[..., :,None,:],destinations[...,None,:,:])
            count=goal_exits.shape[-2]
            direct=(routes[...,:count]+ending[...,None,:]).amin(-1)
            via=(routes[...,count:]+tail[...,None,:]).amin(-1)
            direct=torch.minimum(direct,via)
        else:
            direct=(self.segment_cost(exits[..., :,None,:],goal_exits[...,None,:,:])+ending[...,None,:]).amin(-1)
        return (prefix+direct).amin(-1)
