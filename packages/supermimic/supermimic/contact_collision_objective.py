"""Cheap contact semantics and continuous collision feedback.

Target region bounds are cached once from fixed targets. Native
activation is an input, never inferred from geometry by this objective.
"""
import torch

# Spatial clearance uses the same 5 cm normalization as contact objectives.
# Frozen native replay calibration: realtime-collision-calibration-20261108.
COLLISION_LOSS_WEIGHT = 1.0
COLLISION_CLEARANCE = .01
COLLISION_DISTANCE_SCALE = .05


def contact_regions(clock, padding):
    """Cached target envelopes; candidate knee contacts retain separate boxes."""
    target=clock.material_targets
    K=clock.paired_material_targets.shape[2]
    lower=target.amin(-2)[:,:,None,:].expand(-1,-1,K,-1).clone()
    upper=target.amax(-2)[:,:,None,:].expand(-1,-1,K,-1).clone()
    valid=torch.zeros(lower.shape[:-1],device=target.device,dtype=torch.bool)
    valid[:,:,0]=True
    hand=clock.data['point'][clock.event_ends-1,2:4]
    lower[:,2:4,0]=hand;upper[:,2:4,0]=hand
    lower[:,4:]=clock.paired_material_targets[:,4:]
    upper[:,4:]=clock.paired_material_targets[:,4:]
    valid[:,4:]=clock.paired_material_valid[:,4:]
    return lower-padding,upper+padding,valid


def classify_contacts(active, known, transferring, point, lower, upper, valid):
    measured=active & known & transferring
    inside=(((point[...,None,:]>=lower)&(point[...,None,:]<=upper)).all(-1)&valid).any(-1)
    return measured&inside,measured&~inside


def collision_loss(point, normal, signed_gap, measured_point,
                   active_transfer, in_target_region):
    """Spatial clearance for native collisions outside the target region.

    A 1 cm outward clearance provides feedback at first touching, including
    zero penetration. Retain the measured substep gap as the value, so a
    subsequent rebound cannot erase the collision. FK supplies the derivative.
    No velocity, force or timestep enters the objective.
    """
    displacement = ((point-measured_point.detach())*normal.detach()).sum(-1)
    gap = signed_gap.detach() + displacement - displacement.detach()
    clearance_error = (COLLISION_CLEARANCE-gap).clamp_min(0)/COLLISION_DISTANCE_SCALE
    enabled = active_transfer & ~in_target_region
    return torch.where(enabled, bounded_clearance_cost(clearance_error), 0.)


def collision_interruption(active_transfer, in_target_region):
    """Collector has already excluded entry support and non-transfer parts."""
    return (active_transfer & ~in_target_region).any(-1)


def bounded_clearance_cost(error):
    """Keep the existing square below one distance scale; cap its slope above it."""
    return torch.where(error <= 1., error.square(), 2.*error-1.)


@torch.no_grad()
def collision_prefix_witnesses(data, stop, required):
    """Nearest future native collision, within the same uninterrupted transfer.

    This only selects supervision evidence. It does not classify new contacts.
    An actual contact, a stage change, or a rollout stop ends backward carry.
    """
    T=len(data['done']); B=len(stop); device=stop.device
    result=torch.full((T,B,6),-1,device=device,dtype=torch.long)
    following=torch.full((B,6),-1,device=device,dtype=torch.long)
    for t in range(T-1,-1,-1):
        live=t<stop
        if t+1<T:
            same=data['goal_index'][t]==data['goal_index'][t+1]
            following=torch.where((live & ~data['done'][t] & same)[:,None],following,-1)
        actual=data['active'][t]&data['known'][t]
        following=torch.where(actual | ~required[t] | ~live[:,None],-1,following)
        collision=data['collision_mask'][t]&~data['collision_in_region'][t]&required[t]&live[:,None]
        following=torch.where(collision,torch.full_like(following,t),following)
        result[t]=following
    return result


def collision_prefix_loss(point, normal, gap, anchor, enabled):
    """Spatial correction only near the witnessed obstacle patch (one loss scale).

    The patch mask is frozen native replay evidence, not a contact test. Do not
    extend a local collision plane to distant portions of the trajectory.
    """
    near=(point.detach()-anchor).square().sum(-1) <= COLLISION_DISTANCE_SCALE**2
    enabled=enabled & near
    distance=gap.detach()+((point-anchor.detach())*normal.detach()).sum(-1)
    error=(COLLISION_CLEARANCE-distance).clamp_min(0)/COLLISION_DISTANCE_SCALE
    cost=torch.where(enabled,bounded_clearance_cost(error),0.)
    return cost,enabled


class ContinuousClearance:
    """All-frame spatial objective; never used as native contact evidence.

    Static task geometry is the same validated boxes/z=0 floor as the context.
    Surface quadrature is augmented with every realized collision mesh vertex.
    Target permissions apply at each nearest terrain point, not entire objects.
    """
    def __init__(self, surface, tasks, clock):
        from .rotations import quat_to_mat
        device=surface.atlas.points_local.device
        self.nodes=[]
        for part in range(6):
            vertices=getattr(surface,f'triangles_{part}').reshape(-1,3).unique(dim=0)
            self.nodes.append(torch.cat((surface.atlas.points_local[part],vertices)))
        self.terrain=[]
        for task in tasks:
            boxes=torch.tensor(task['boxes'],device=device,dtype=torch.float32).reshape(-1,10)
            self.terrain.append((boxes[:,:3],quat_to_mat(boxes[:,3:7]),boxes[:,7:10]*.5))
        # Same target contact region as the native task. No whole-object exemption.
        self.lower,self.upper,self.valid=contact_regions(clock,.1)
        self.version='all_frame_collision_surface_clearance_v1'

    def loss(self, positions, rotations, events, motions, transferring, entry_blocked):
        enabled=transferring&~entry_blocked
        result=positions.sum(-1)*0.
        for part in range(6):
            ids=enabled[:,part].nonzero().flatten()
            if not len(ids):continue
            points=positions[ids,part,None,:]+torch.einsum('bij,kj->bki',rotations[ids,part],self.nodes[part])
            event=events[ids]
            for motion in motions[ids].unique().tolist():
                use=(motions[ids]==motion).nonzero().flatten()
                p=points[use]
                center,R,half=self.terrain[motion]
                distance,nearest=terrain_distances(p,center,R,half)
                allowed=points_in_regions(nearest,self.lower[event[use],part],self.upper[event[use],part],self.valid[event[use],part])
                error=(COLLISION_CLEARANCE-distance).clamp_min(0)/COLLISION_DISTANCE_SCALE
                costs=torch.where(allowed,0.,bounded_clearance_cost(error))
                result[ids[use],part]=costs.flatten(1).amax(-1)
        return result


def points_in_regions(points, lower, upper, valid):
    """Pointwise target permission, with arbitrary point axes after batch."""
    shape=(len(points),)+ (1,)*(points.ndim-2)+lower.shape[1:]
    return (((points[...,None,:]>=lower.reshape(shape)) &
             (points[...,None,:]<=upper.reshape(shape))).all(-1)&
            valid.reshape(shape[:-1])).any(-1)


def terrain_distances(points, center, rotation, half):
    """Signed OBB/floor distances and closest surface positions, differentiable.

    Return [batch,node,obstacle] distances; last obstacle is the actual z=0 floor.
    Uses the shared exact analytic box distance field. Inside depth is
    provided by the nearest face; terrain parameters are cached once.
    """
    delta=points[:,:,None,:]-center[None,None]
    local=torch.einsum('bnki,kij->bnkj',delta,rotation)
    q=local.abs()-half
    from somaforge_core.box_distance import box_distance_local
    distance=box_distance_local(local,half)
    closest=local.clamp(-half,half)
    axis=q.argmax(-1)
    face=torch.nn.functional.one_hot(axis,3).bool()
    boundary=torch.where(local>=0,half,-half)
    inside_closest=torch.where(face,boundary,local)
    closest=torch.where((q.amax(-1)<0)[...,None],inside_closest,closest)
    world=center+torch.einsum('kij,bnkj->bnki',rotation,closest)
    floor=points.clone();floor[...,2]=0
    return torch.cat((distance,points[...,2,None]),-1),torch.cat((world,floor[:,:,None]),-2)
