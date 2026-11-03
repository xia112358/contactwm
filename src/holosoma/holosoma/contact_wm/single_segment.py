"""Main-task single-segment objective adapted to the historical policy/state layout."""
import torch
from torch import nn
from holosoma.contact_wm.geometry import LINKS


class SurfacePoses(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry=geometry

    def poses(self, state):
        qpos=torch.cat((state[...,:3],state[...,[6,3,4,5]],
                       state[...,13:42].index_select(-1,self.geometry.order)),-1)
        return self.geometry.fk.link_poses(qpos,LINKS)


def objective(learner,x,states,actions,priors,valid,native):
    H,B=states.shape[:2];new=learner.establish.expand(B,-1)
    live=native['live'];stop=live.sum(0)
    required=new|learner.maintain.expand(B,-1)
    target=torch.where(learner.maintain[...,None],x['support'],x['target'])
    position,rotation=learner.segment_surface.geometry.poses(states.flatten(0,1))
    position=position.reshape(H,B,6,3);rotation=rotation.reshape(H,B,6,3,3)
    point=position+(rotation@native['stable_contact_local'].detach()[...,None]).squeeze(-1)
    score=native['stable_contact_mask']&native['active']&native['known']&live[...,None]&required[None]
    # Unstable transition frames contribute neither value nor gradient.
    error=((point-target[None])/.05).square().sum(-1)
    count=score.sum(0)
    part_loss=torch.where(score,error,0.).sum(0)/count.clamp_min(1)
    missing=required&(count==0)
    if missing.any():
        end_state=states[stop-1,torch.arange(B,device=states.device)]
        region=learner.segment_surface.region_loss(end_state,target,
            required.any(0).nonzero().flatten().tolist(),normals=learner.normal_w.expand(B,-1,-1))
        guidance=torch.where(missing,region['loss'],0.)
    else:guidance=torch.zeros_like(part_loss)
    reduce=lambda value:(torch.where(required,value,0.).sum(-1)/required.sum(-1).clamp_min(1)).mean()
    stable_contact=reduce(part_loss)
    terminal_guidance=reduce(guidance)
    contact=stable_contact+terminal_guidance
    terminal=live&(torch.arange(H,device=states.device)[:,None]==stop[None]-1)
    ti,bi=terminal.nonzero(as_tuple=True);end=states[ti,bi];ref=learner.segment_reference
    root_position=((ref[:3]-end[:,:3])/.1).square().mean(-1)
    root_velocity=.1*((end[:,7:10]-ref[7:10]).square().mean(-1)+(end[:,10:13]-ref[10:13]).square().mean(-1))
    body=(root_position+root_velocity).mean()
    previous=torch.cat((x['previous'][None],actions[:-1]),0)
    smooth=.01*(actions-previous).square().mean(-1)
    magnitude=.001*actions.square().mean(-1)
    prior=.01*(actions-priors).square().sum(-1)
    average=lambda v:(torch.where(live,v,0.).sum(0)/stop).mean()
    regular=average(smooth+magnitude+prior)
    zero=contact.new_zeros(())
    # Preserve legacy metric names; explicit component metrics are also exported.
    values=torch.stack((contact+body+regular,contact,zero,zero,zero,regular,average(prior),body,contact,zero))
    learner.segment_metrics=dict(root_position=float(root_position.detach().mean()),
        root_orientation=0.,root_velocity=float(root_velocity.detach().mean()),
        action_smoothness=float(average(smooth).detach()),action_magnitude=float(average(magnitude).detach()),
        dagger_prior=float(average(prior).detach()),segment_mean_steps=float(stop.float().mean()),
        stable_contact_loss=float(stable_contact.detach()),terminal_guidance_loss=float(terminal_guidance.detach()),
        stable_contact_frames=int(score.sum()),missing_stable_parts=int(missing.sum()))
    return values,(valid|~live).all()&torch.isfinite(values).all()
