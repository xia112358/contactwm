"""Point navigation to unchanged contact goals; native contact semantics stay upstream."""
import itertools
import torch
from scipy.sparse.csgraph import floyd_warshall
from supermimic.contact_navigation import segment_visible
from supermimic.rotations import quat_to_mat
from supermimic.contact_material import world_points


class PointNavigation:
    """Cached visibility roadmap for box obstacles, with exact unobstructed distances.

    Routes are graph approximations, not exact geodesics. Original goals are
    immutable. For penetrating endpoints the metric includes distance to exit
    the solid; that extension does not count as contact evidence.
    """
    def __init__(self, boxes, goals):
        self.goals=goals.detach().clone()
        box=torch.as_tensor(boxes,device=goals.device,dtype=goals.dtype).reshape(-1,10)
        self.center=box[:,:3];self.rotation=quat_to_mat(box[:,3:7]);self.half=box[:,7:]/2
        nodes=[]
        for center,rotation,half in zip(self.center,self.rotation,self.half):
            grid=goals.new_tensor([v for v in itertools.product((-1.,0.,1.),repeat=3) if any(abs(x)==1 for x in v)])
            points=(grid*(half+1e-5))@rotation.T+center
            nodes.append(points[points[:,2]>=0])
        self.nodes=torch.cat(nodes) if nodes else goals.new_empty((0,3))
        if len(self.nodes):
            # Other boxes may cover some vertices; they are not route nodes.
            _,depth=self._exit(self.nodes)
            self.nodes=self.nodes[depth==0]
        n=len(self.nodes)
        if n:
            edges=(self.nodes[:,None]-self.nodes[None]).norm(dim=-1).masked_fill(~self.visible(self.nodes[:,None],self.nodes[None]),float('inf'))
            self.costs=torch.as_tensor(floyd_warshall(edges.cpu().double().numpy(),directed=False),device=goals.device,dtype=goals.dtype)
        else:self.costs=goals.new_empty((0,0))
        self.free_goal,self.goal_depth=self._exit(self.goals)
        if n:
            target=(self.free_goal[...,None,:]-self.nodes).norm(dim=-1).masked_fill(~self.visible(self.free_goal[...,None,:],self.nodes),float('inf'))
            self.tail=(self.costs+target[...,None,:]).amin(-1)

    def visible(self,start,end):
        result=torch.ones(torch.broadcast_shapes(start.shape[:-1],end.shape[:-1]),device=start.device,dtype=torch.bool)
        for c,R,h in zip(self.center,self.rotation,self.half):
            result &= segment_visible((start-c)@R,(end-c)@R,h)
        return result

    def _exit(self,points):
        free=points;depth=points.new_zeros(points.shape[:-1])
        for c,R,h in zip(self.center,self.rotation,self.half):
            local=(free-c)@R;gap=local.abs()-h;inside=gap.amax(-1)<0
            # Only exits above the physical ground are admissible.
            candidates=[];distances=[]
            for axis in range(3):
                for sign in (-1.,1.):
                    candidate=local.clone();candidate[...,axis]=sign*(h[axis]+1e-5)
                    world=candidate@R.T+c
                    candidates.append(world)
                    distances.append((world-free).norm(dim=-1).masked_fill(world[...,2]<0,float('inf')))
            distances=torch.stack(distances,-1);index=distances.argmin(-1)
            selected=torch.stack(candidates,-2).gather(-2,index[...,None,None].expand(*index.shape,1,3)).squeeze(-2)
            depth=depth+torch.where(inside,distances.amin(-1),0.)
            free=torch.where(inside[...,None],selected,free)
        return free,depth

    def distance(self,points,goals=None):
        free,depth=self._exit(points)
        free_goal,goal_depth=(self.free_goal,self.goal_depth) if goals is None else self._exit(goals)
        direct=(free-free_goal).norm(dim=-1).masked_fill(~self.visible(free,free_goal),float('inf'))
        if len(self.nodes):
            source=(free[...,None,:]-self.nodes).norm(dim=-1).masked_fill(~self.visible(free[...,None,:],self.nodes),float('inf'))
            tail=self.tail
            if goals is not None:
                target=(free_goal[...,None,:]-self.nodes).norm(dim=-1).masked_fill(~self.visible(free_goal[...,None,:],self.nodes),float('inf'))
                tail=(self.costs+target[...,None,:]).amin(-1)
            distance=torch.minimum(direct,(source+tail).amin(-1))
        else:distance=direct
        return distance+depth+goal_depth


class ContactNavigationObjective:
    def __init__(self,clock,surface,context):
        self.metrics={}
        for event in range(len(clock.event_ends)):
            end=clock.event_ends[event]-1
            boxes=context.tasks[int(clock.segment_motion[event])]['boxes']
            for part in (clock.target_masks[end]&~clock.keep_masks[end]).nonzero().flatten().tolist():
                target=(clock.data['point'][end,part:part+1] if part in (2,3) else
                        clock.paired_material_targets[event,part] if part>=4 and hasattr(clock,'paired_material_targets') else
                        clock.material_targets[event,part])
                self.metrics[event,part]=PointNavigation(boxes,target)
        self.clock=clock;self.surface=surface

    def cost(self,p,R,event,evaluate,stable,guidance,data,t,swap):
        clock=self.clock;cost=p.new_zeros(p.shape[:2])
        for (index,part),metric in self.metrics.items():
            slots=((event==index)&evaluate[:,part]).nonzero().flatten()
            if not len(slots):continue
            pos,rot=p[slots,part],R[slots,part];use=stable[slots,part]
            if part in (2,3):
                local=self.surface.atlas.points_local[part]
                points=pos[:,None]+torch.einsum('bij,kj->bki',rot,local)
                errors=metric.distance(points).square()/.05**2
                area=self.surface.atlas.areas[part];prior=area/area.sum()
                value=-self.surface.temperature*torch.logsumexp(prior.log()-errors/self.surface.temperature,-1)
                measured=data['contact_loss_point'][t,slots,part]
                anchor=(rot.detach().transpose(-1,-2)@(measured-pos.detach()).unsqueeze(-1)).squeeze(-1)
                witness=swap(pos+(rot@anchor.unsqueeze(-1)).squeeze(-1),measured)
                observed=metric.distance(witness[:,None]).square().squeeze(-1)/.05**2
                value=torch.where(use,observed,value)
            else:
                raw=(guidance[slots,part]&data['contact_raw_stable'][t,slots,part]) if part>=4 and 'contact_raw_stable' in data else torch.zeros_like(use)
                measured_p=data['contact_loss_position'][t,slots,part];measured_R=data['contact_loss_rotation'][t,slots,part]
                pos=torch.where(use[:,None],swap(pos,measured_p),pos);rot=torch.where(use[:,None,None],swap(rot,measured_R),rot)
                if part>=4 and hasattr(clock,'paired_material_local'):
                    pos=torch.where(raw[:,None],swap(pos,data['contact_raw_position'][t,slots,part]),pos)
                    rot=torch.where(raw[:,None,None],swap(rot,data['contact_raw_rotation'][t,slots,part]),rot)
                    local=clock.paired_material_local[index,part];valid=clock.paired_material_valid[index,part]
                    points=world_points(pos,rot,local)
                    errors=(metric.distance(points).square()/.05**2).masked_fill(~valid,float('inf'))
                    native=data['contact_loss_candidate'][t,slots,part].long()
                    # Preserve the paired raw-witness guidance objective, replacing
                    # both geometric distances with the same navigation metric.
                    correspondence=metric.distance(points,data['contact_raw_point'][t,slots,part,None]).square()/.05**2
                    guided=torch.where(raw[:,None],errors+correspondence,errors)
                    chosen=torch.where(use,native,guided.detach().argmin(-1))
                    value=guided.gather(-1,chosen[:,None]).squeeze(-1)
                else:
                    points=world_points(pos,rot,clock.material_local[index,part])
                    value=metric.distance(points).square().mean(-1)/.05**2
            cost[:,part]=cost[:,part].index_copy(0,slots,value)
        return cost


def contact_navigation_cost(clock,surface,context,p,R,event,evaluate,stable,guidance,data,t,swap):
    if not hasattr(surface,'contact_navigation_objective'):
        surface.contact_navigation_objective=ContactNavigationObjective(clock,surface,context)
    return surface.contact_navigation_objective.cost(p,R,event,evaluate,stable,guidance,data,t,swap)
