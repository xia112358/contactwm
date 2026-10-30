"""Coherent target-face queries for the current ground/box scene adapter.

Fixed FK material features are shared by interval bounds and query identity.
The interval scalar preserves the original minimum and tie derivatives.
This adapter supplies optimization geometry, never Newton contact labels.
Other scene types can supply the same SurfaceIntervalBounds protocol.
"""
import torch
from contact_solver.contact_surface_interval import SurfaceIntervalBounds,SurfaceIntervalQuery
from somaforge_core.motion_contracts import BODY_NAMES


def selected_point_bounds(gap, outside, margin, *, return_selection=False):
    distance=torch.linalg.vector_norm(torch.cat((gap.relu()[...,None],outside),-1),dim=-1)
    lower=(-gap).relu().square()
    upper=(distance-margin).relu().square()
    total=upper+lower
    # One common selected feature and mean tie derivative, as in cost.amin.
    selected=total==total.amin(-1,keepdim=True)
    weights=selected.to(total)/selected.sum(-1,keepdim=True)
    result=SurfaceIntervalBounds((weights*lower).sum(-1),(weights*upper).sum(-1),scalar=total.amin(-1))
    return (result,selected,total.argmin(-1)) if return_selection else result


def material_surface_interval(fk,geometry,q,surface,scene,margin):
    """Use exactly the source features and finite-face chart of control500."""
    if bool(((surface>=0)&(surface!=0)&(surface!=1)).any()):
        raise ValueError('Ground/box material adapter cannot represent this target face; provide its actual geometry adapter')
    position,rotation=fk.link_poses(q,BODY_NAMES[1:7])
    lower=q.new_zeros(len(q),6,4);upper=torch.zeros_like(lower)
    scalar=torch.zeros_like(lower)
    query_link=torch.full(lower.shape,-1,dtype=torch.long,device=q.device)
    query_point=q.new_zeros(lower.shape+(3,));query_normal=torch.zeros_like(query_point)
    query_gap=torch.zeros_like(lower);unique=torch.zeros_like(lower,dtype=torch.bool);interior=torch.zeros_like(unique)
    top_center=scene['box_center']+scene['box_rotation'][:,:,2]*scene['box_half_extents'][:,2:3]
    skin=geometry.material_skin
    skin_poses=None
    if skin is not None:
        if not skin.finalized:raise ValueError('Material bank must already be finalized')
        skin_poses=fk.link_poses(q,skin.link_names)
    for part in range(6):
        for region in range(len(geometry.names[part])):
            if skin is not None and bool(skin.known[part,region]):
                links=getattr(skin,f'link_{part}_{region}')
                local=getattr(skin,f'local_{part}_{region}').to(q)
                p,r=skin_poses
                points=p[:,links]+torch.einsum('bnij,nj->bni',r[:,links],local)
            else:
                cloud=getattr(geometry,f'cloud_{part}_{region}')
                points=position[:,part,None]+torch.einsum('bij,vj->bvi',rotation[:,part],cloud)
                local=cloud.to(q)
                links=None if skin is None else torch.full((len(cloud),),skin.link_names.index(BODY_NAMES[part+1]),device=q.device,dtype=torch.long)
            ground_gap=points[...,2]-scene['ground_height'][:,None]
            box_local=torch.einsum('bvi,bij->bvj',points-top_center[:,None],scene['box_rotation'])
            ground=surface[:,part,None]==0
            gap=torch.where(ground,ground_gap,box_local[...,2])
            outside=(box_local[...,:2].abs()-scene['box_half_extents'][:,None,:2]).relu()
            outside=torch.where(ground[...,None],0,outside)
            bound,selected,chosen=selected_point_bounds(gap,outside,margin[:,None],return_selection=True)
            lower[:,part,region]=bound.lower;upper[:,part,region]=bound.upper
            scalar[:,part,region]=bound.cost
            if links is not None:
                query_link[:,part,region]=links[chosen]
                query_point[:,part,region]=local[chosen]
                query_normal[:,part,region]=torch.where(ground,q.new_tensor([0.,0.,1.]),scene['box_rotation'][:,:,2])
                query_gap[:,part,region]=gap.gather(1,chosen[:,None]).squeeze(1)
                unique[:,part,region]=selected.sum(-1)==1
                interior[:,part,region]=outside.gather(1,chosen[:,None,None].expand(-1,1,2)).squeeze(1).eq(0).all(-1)
    query=None if skin is None else SurfaceIntervalQuery(tuple(skin.link_names),query_link,query_point,query_normal,query_gap,unique,interior)
    return SurfaceIntervalBounds(lower,upper,query=query,scalar=scalar)

