"""Box-top foot navigation on immutable event goals; native contacts stay upstream."""
import torch
from supermimic.contact_navigation import WholeFootBoxNavigation, segment_visible


class CachedFootNavigation(WholeFootBoxNavigation):
    """Exact static-goal specialization: cache the goal half of the roadmap query."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        nodes=self.road.nodes
        target=(self.goals[:,None]-nodes).norm(dim=-1).masked_fill(
            ~segment_visible(self.goals[:,None],nodes,self.nav_half[:,None]),float('inf'))
        self.tail=(self.road.costs+target[:,None]).min(-1).values

    def anchor_distance(self,points):
        position=(points-self.center)@self.R
        gap=position.abs()-self.nav_half[None]
        inside=gap.amax(-1)<0
        exit_gap=gap.clone()
        exit_gap[...,2]=torch.where(position[...,2]<0,-self.nav_half[None,:,2]+position[...,2],gap[...,2])
        axis=exit_gap.argmax(-1);face=torch.nn.functional.one_hot(axis,3).bool()
        sign=torch.where(position>=0,1.,-1.)
        sign=torch.where(face&(axis[...,None]==2),torch.ones_like(sign),sign)
        free=torch.where(inside[...,None],torch.where(face,sign*self.nav_half[None],position)+face*sign*1e-5,position)
        source=(free[:,:,None]-self.road.nodes[None]).norm(dim=-1).masked_fill(
            ~segment_visible(free[:,:,None],self.road.nodes[None],self.nav_half[None,:,None]),float('inf'))
        indirect=(source+self.tail[None]).min(-1).values
        direct=(free-self.goals[None]).norm(dim=-1).masked_fill(
            ~segment_visible(free,self.goals[None],self.nav_half[None]),float('inf'))
        depth=torch.where(inside,-exit_gap.amax(-1),0.)
        return torch.minimum(direct,indirect)+depth


class FootNavigationObjective:
    def __init__(self,clock,surface,context):
        self.metrics={}
        self.rows=[]
        for event in range(len(clock.event_ends)):
            end=clock.event_ends[event]-1
            required=clock.target_masks[end]&~clock.keep_masks[end]
            motion=int(clock.segment_motion[event]);boxes=context.tasks[motion]['boxes']
            box=torch.as_tensor(boxes,device=clock.material_targets.device)
            assert len(box)==1,'This 10-stage scene must have its original one-box terrain'
            top=box[0,2]+box[0,9]/2
            for part in range(2):
                if not bool(required[part]):continue
                target=clock.material_targets[event,part]
                # Select the demonstrated box-top objective by its source target,
                # not by student distance or contact state. Ground goals stay fixed.
                box_top=bool((target[:,2].mean()-top).abs()<target[:,2].mean().abs())
                if not box_top:continue
                vertices=getattr(surface,f'triangles_{part}').reshape(-1,3).unique(dim=0)
                metric=CachedFootNavigation(boxes,target,vertices,clock.material_local[event,part],target.device)
                self.metrics[event,part]=metric
                self.rows.append(dict(segment=event+1,case=int(clock.case_ids[event]),part=part,**metric.info))
        print('NAVIGATION_OBJECTIVE',self.rows,flush=True)

    def cost(self,points,position,rotation,event,required,original):
        cost=original
        for (index,part),metric in self.metrics.items():
            slots=((event==index)&required[:,part]).nonzero().flatten()
            if not len(slots):continue
            distance=metric.distance(points[slots,part],position[slots,part],rotation[slots,part])
            values=distance.square().mean(-1)/.05**2
            replacement=cost[:,part].index_copy(0,slots,values)
            cost=torch.stack([replacement if p==part else cost[:,p] for p in range(cost.shape[1])],-1)
        return cost


def foot_navigation_cost(clock,surface,context,points,position,rotation,event,required,original):
    if not hasattr(surface,'foot_navigation_objective'):
        surface.foot_navigation_objective=FootNavigationObjective(clock,surface,context)
    return surface.foot_navigation_objective.cost(points,position,rotation,event,required,original)
