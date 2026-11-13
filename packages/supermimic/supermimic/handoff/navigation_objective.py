"""Uniform obstacle-aware material-pair objective for every required body part."""
import torch
from supermimic.contact_terrain_distance import TerrainNavigationDistance


class MaterialNavigationObjective:
    def __init__(self,clock,context,surface,*,compile_queries=True):
        self.occupancy={};self.combined_prepared={};self.metrics={};self.prepared={};self.local={};self.rows=[];self.motion=clock.segment_motion.tolist()
        for event in range(len(clock.event_ends)):
            motion=int(clock.segment_motion[event])
            if motion not in self.metrics:
                self.metrics[motion]=TerrainNavigationDistance(context.tasks[motion]['boxes'],clock.material_targets.device)
            metric=self.metrics[motion];end=clock.event_ends[event]-1
            required=clock.target_masks[end]&~clock.keep_masks[end]
            for part in required.nonzero().flatten().tolist():
                if part in (2,3):
                    local=clock.material_local[event,part,:1]
                    target=clock.data['point'][end,part,None]
                    candidates=None
                elif part>=4 and hasattr(clock,'paired_material_local'):
                    valid=clock.paired_material_valid[event,part]
                    local=clock.paired_material_local[event,part,valid]
                    target=clock.paired_material_targets[event,part,valid]
                    candidates=valid.nonzero().flatten()
                else:
                    local=clock.material_local[event,part];target=clock.material_targets[event,part]
                    candidates=None
                if part<2:
                    from supermimic.contact_foot_occupancy import FootOccupancy
                    vertices=getattr(surface,f'triangles_{part}').reshape(-1,3).unique(dim=0)
                    self.occupancy[event,part]=FootOccupancy(vertices,local,target,metric)
                self.local[event,part]=(local,candidates)
                self.prepared[event,part]=metric.prepare(target)
                if part<2:
                    self.combined_prepared[event,part]=metric.prepare(torch.cat((target,self.occupancy[event,part].goals)))
                self.rows.append(dict(case=int(clock.case_ids[event]),part=part,materials=len(local),motion=motion))
        if compile_queries and clock.material_targets.is_cuda:
            self.compile_queries()
        print('UNIFIED_MATERIAL_NAVIGATION',self.rows,flush=True)

    def compile_queries(self):
        # Fuse the unchanged spatial objective, including its input gradients.
        # Static terrain/goal preparation stays outside the repeated query.
        for metric in self.metrics.values():
            # Keep route selection and tie reduction in eager PyTorch: compiling
            # the complete query produced nonfinite gradients on real replay.
            compiled=torch.compile(metric.segment_cost,dynamic=True,fullgraph=True,
                options={'triton.cudagraphs':False})
            def segment_cost(start,end,kernel=compiled):
                # Prefix and route queries have different broadcast ranks.
                # Own the input storage as well: Dynamo guards on view base shapes.
                # Compile one line batch, then restore the query shape.
                start,end=torch.broadcast_tensors(start,end)
                shape=start.shape[:-1]
                return kernel(start.reshape(-1,3).clone(memory_format=torch.contiguous_format),
                              end.reshape(-1,3).clone(memory_format=torch.contiguous_format)).reshape(shape)
            metric.segment_cost=segment_cost

    def cost(self,position,rotation,event,required,stable,candidate,hand_points):
        answer=position.new_zeros(required.shape)
        for (index,part),(local,candidates) in self.local.items():
            slots=((event==index)&required[:,part]).nonzero().flatten()
            if not len(slots):continue
            if part in (2,3):
                points=hand_points[slots,part,None]
            else:
                points=position[slots,part,None]+torch.einsum('bij,kj->bki',rotation[slots,part],local)
            metric=self.metrics[self.motion[index]]
            if part<2:
                foot=self.occupancy[index,part]
                corners=position[slots,part,None]+torch.einsum('bij,kj->bki',rotation[slots,part],foot.corners)
                routed=metric.distance(torch.cat((points,corners),1),self.combined_prepared[index,part])
                distances=routed[:,:len(local)]
                occupancy=foot.cost_from_distances(position[slots,part],rotation[slots,part],corners,routed[:,len(local):])
            else:
                distances=metric.distance(points,self.prepared[index,part])
            costs=distances.square()/.05**2
            if candidates is None:
                values=costs.mean(-1)
            else:
                # Stable rows keep their own native material index. Unconfirmed
                # guidance chooses a complete local-point/world-target pair.
                matched=candidate[slots,part,None]==candidates[None]
                qualified=costs.masked_fill(~matched,float('inf')).amin(-1)
                values=torch.where(stable[slots,part],qualified,costs.amin(-1))
            if part<2:
                values=values+occupancy
            column=answer[:,part].index_copy(0,slots,values)
            answer=torch.stack([column if k==part else answer[:,k] for k in range(answer.shape[1])],-1)
        return answer


def material_navigation_cost(clock,surface,context,position,rotation,event,required,stable,candidate,hand_points):
    if not hasattr(surface,'material_navigation_objective'):
        surface.material_navigation_objective=MaterialNavigationObjective(clock,context,surface)
    return surface.material_navigation_objective.cost(position,rotation,event,required,stable,candidate,hand_points)
