"""Minimal WM-BPTT building blocks, with explicit runtime adapters.

No motion frames enter the policy. The runtime must supply differentiable
observation/terrain reconstruction and body contact-point geometry. This module
never silently substitutes reference states or geometric proximity for contacts.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

PARTS=('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')


class ContactPolicy(nn.Module):
    """Direct PHP MLP, extended with 18 normals, 6 support flags and a deadline.

    The existing 567 inputs retain target XYZ/masks and deployable observations.
    Additional inputs are appended, not passed through an imposed latent encoder.
    New columns are zero-initialized, preserving the pretrained mean action.
    All original layers remain trainable. Contact names follow the PHP schema.
    """
    def __init__(self, prior_mlp, extra_dim=25, action_clip=10.):
        super().__init__()
        self.net=deepcopy(prior_mlp)
        old=self.net[0]
        if not isinstance(old,nn.Linear) or old.in_features!=567:
            raise ValueError('Expected published PHP direct 567-input MLP')
        new=nn.Linear(old.in_features+extra_dim,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:567].copy_(old.weight);new.bias.copy_(old.bias)
        self.net[0]=new
        self.extra_dim=extra_dim
        self.action_clip=action_clip

    def forward(self, observation, condition):
        if observation.shape[-1]!=567 or condition.shape[-1]!=self.extra_dim:
            raise ValueError('Observation/condition contract mismatch')
        return self.net(torch.cat((observation,condition),-1)).clamp(-self.action_clip,self.action_clip)


class ContactWorldModel(nn.Module):
    """Residual learned dynamics with independent native contact/force heads.

    State: [p(3), quaternion xyzw(4), world v(3), world w(3), q(29), qdot(29)].
    Context contains environment observations, never target contacts or future
    reference states. Previous contact probabilities allow recursive rollout.
    Forces are world-frame N; normalization is explicit. Context reconstruction
    is the caller's responsibility, including relative frame changes.
    """
    def __init__(self,context_dim,hidden=256,force_scale=300.):
        super().__init__()
        if context_dim<1 or force_scale<=0:raise ValueError('Invalid WM dimensions/scales')
        self.config=dict(context_dim=context_dim,hidden=hidden,force_scale=force_scale)
        self.net=nn.Sequential(nn.Linear(71+29+6+context_dim,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,71+6+18))
        self.register_buffer('state_scale',torch.cat((torch.ones(13),torch.ones(29),torch.full((29,),10.))))
        self.register_buffer('delta_scale',torch.cat((torch.full((3,),.02),torch.full((4,),.02),
            torch.full((6,),.2),torch.full((29,),.05),torch.ones(29))))

    def forward(self,state,action,contact,context):
        if state.shape[-1]!=71 or action.shape[-1]!=29 or contact.shape[-1]!=6 or context.shape[-1]!=self.config['context_dim']:
            raise ValueError('WM transition dimensions mismatch')
        y=self.net(torch.cat((state/self.state_scale,action,contact,context),-1))
        value=state+y[...,:71]*self.delta_scale
        quaternion=F.normalize(value[...,3:7],dim=-1,eps=1e-6)
        value=torch.cat((value[...,:3],quaternion,value[...,7:]),-1)
        logits=y[...,71:77]
        return dict(state=value,contact_logits=logits,contact=logits.sigmoid(),
                    force_w=y[...,77:].reshape(*y.shape[:-1],6,3)*self.config['force_scale'])


def fit_loss(prediction,next_state,active,force_w,known,transition_valid,force_scale=300.,switch_weight=1.):
    """Native solve labels and state targets, masked across resets/unknowns.

    State and contact labels retain their separately recorded sampling times.
    switch_weight may upweight real activation changes without altering labels.
    """
    valid=transition_valid.bool()
    if not valid.any():return prediction['state'].sum()*0
    state=prediction['state'][valid];target=next_state[valid]
    # Quaternion double-cover must not manufacture a large rotational error.
    sign=torch.where((state[...,3:7]*target[...,3:7]).sum(-1,keepdim=True)>=0,1.,-1.)
    error=torch.cat((state[...,:3]-target[...,:3],state[...,3:7]-sign*target[...,3:7],state[...,7:]-target[...,7:]),-1)
    scale=error.new_tensor([.02]*3+[.02]*4+[.2]*6+[.05]*29+[1.]*29)
    state_loss=(error/scale).square().mean()
    mask=known[valid].bool();logits=prediction['contact_logits'][valid]
    bce=F.binary_cross_entropy_with_logits(logits,active[valid].float(),reduction='none')
    safe_force=torch.where(mask[...,None],force_w[valid],0.)
    ferror=((prediction['force_w'][valid]-safe_force)/force_scale).square().mean(-1)
    # where avoids unknown NaN force targets poisoning a valid batch.
    contact_loss=torch.where(mask,bce+ferror,0.).sum()/mask.sum().clamp_min(1)
    return state_loss+switch_weight*contact_loss


def load_planned_contact_points(path, *, device):
    """Absolute world-frame points in the canonical six-contact order.

    Planning supplies positions; task contact roles and timing are unchanged.
    No achieved position or reference-relative offset enters this interface.
    """
    import json
    from pathlib import Path
    from holosoma.contact_wm.geometry import LINKS
    data=json.loads(Path(path).read_text())
    if data.get('frame')!='world' or data.get('links')!=list(LINKS):
        raise ValueError('Planned contacts require world frame and canonical contact links')
    points=torch.tensor(data['position_w'],dtype=torch.float32,device=device)
    if points.shape!=(6,3) or not torch.isfinite(points).all():
        raise ValueError('Expected six finite absolute planned contact points')
    return points[None]


@dataclass
class ContactGoal:
    position_w: torch.Tensor       # [B,6,3], desired interface point
    normal_w: torch.Tensor         # [B,6,3], unit outward surface normal
    establish: torch.Tensor        # [B,6] contacts to establish
    maintain: torch.Tensor         # [B,6] supports required throughout
    support_position_w: torch.Tensor  # [B,6,3], initial support anchors
    min_normal_force: torch.Tensor # [B,6], N; physical success separately validated
    arrival_step: int
    hold_steps: int

    def validate(self,batch,horizon):
        for field in (self.position_w,self.normal_w,self.support_position_w):
            if field.shape!=(batch,6,3) or not torch.isfinite(field).all():raise ValueError('Invalid contact geometry')
        for field in (self.establish,self.maintain,self.min_normal_force):
            if field.shape!=(batch,6):raise ValueError('Invalid contact mask/force shape')
        if self.establish.dtype!=torch.bool or self.maintain.dtype!=torch.bool:raise ValueError('Boolean contact masks required')
        if (self.establish & self.maintain).any():raise ValueError('New contacts and maintained supports must be disjoint')
        if not torch.isfinite(self.min_normal_force).all() or (self.min_normal_force<0).any():raise ValueError('Invalid support force')
        relevant=self.establish|self.maintain
        norms=self.normal_w.norm(dim=-1)
        if ((norms-1).abs()[relevant]>1e-4).any():raise ValueError('Contact normals must be unit vectors')
        if self.arrival_step<1 or self.hold_steps<1 or self.arrival_step+self.hold_steps-1>horizon:
            raise ValueError('Unroll must include arrival and a sustained hold interval')


@contextmanager
def frozen_parameters(module):
    flags=[p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(module.parameters(),flags):p.requires_grad_(flag)


def _masked_mean(value,mask):
    return torch.where(mask,value,0.).sum()/mask.sum().clamp_min(1)


def contact_geometry(normal,tangent,new_mask,support_mask,target_tangent_scale=.05):
    """Equal weight per sample and per role, independent of support duration."""
    def average(value,mask):
        mask=mask.expand_as(value)
        return (torch.where(mask,value,0.).sum((0,2))/mask.sum((0,2)).clamp_min(1)).mean()
    target=average((normal/.02).square()+tangent.square().sum(-1)/target_tangent_scale**2,new_mask)
    support=average((normal/.02).square()+tangent.square().sum(-1)/.05**2,support_mask)
    return target,support


def contact_objective(predictions,points_w,initial_points,actions,initial_action,goal,dt=.02,prior_actions=None,target_tangent_scale=.05):
    """Differentiable surrogate: establish, load, hold, no-slip and support retention.

    points_w are consistent robot-side witness points from differentiable FK,
    NOT learned freely movable contact-point heads. Whole-body stability is
    additionally supplied by the task-specific stability callback in unroll().
    """
    horizon,batch=points_w.shape[:2];goal.validate(batch,horizon)
    if dt<=0:raise ValueError('Positive dt required')
    t=torch.arange(1,horizon+1,device=points_w.device)[:,None,None]
    hold=(t>=goal.arrival_step)&(t<goal.arrival_step+goal.hold_steps)
    required=goal.maintain[None].expand(horizon,-1,-1)|(hold&goal.establish[None])
    target=torch.where(goal.maintain[...,None],goal.support_position_w,goal.position_w)
    error=points_w-target[None]
    normal=(error*goal.normal_w[None]).sum(-1)
    tangent=error-normal[...,None]*goal.normal_w[None]
    target_geometry,support_geometry=contact_geometry(normal,tangent,hold&goal.establish[None],
        goal.maintain[None].expand_as(required),target_tangent_scale)
    geometry=target_geometry+support_geometry
    logits=torch.stack([p['contact_logits'] for p in predictions])
    activation=_masked_mean(F.softplus(-logits),required)
    forces=torch.stack([p['force_w'] for p in predictions])
    fn=(forces*goal.normal_w[None]).sum(-1)
    load=_masked_mean((F.relu(goal.min_normal_force[None]-fn)/100.).square(),required)
    velocity=(points_w-torch.cat((initial_points[None],points_w[:-1]),0))/dt
    vn=(velocity*goal.normal_w[None]).sum(-1)
    vt=velocity-vn[...,None]*goal.normal_w[None]
    slip=_masked_mean(vt.square().sum(-1)+.1*vn.square(),required)
    action_prev=torch.cat((initial_action[None],actions[:-1]),0)
    regularization=(actions-action_prev).square().mean()+.01*actions.square().mean()
    motion=actions.sum()*0 if prior_actions is None else (actions-prior_actions.detach()).square().mean()
    total=geometry+activation+load+.1*slip+.01*regularization+.1*motion
    return dict(loss=total,geometry=geometry,target_geometry=target_geometry,support_geometry=support_geometry,activation=activation,load=load,slip=slip,
                regularization=regularization,motion=motion)


def unroll(policy,world_model,state,contact,history,initial_action,goal,*,horizon,
           observation,geometry,context,stability,prior=None,dt=.02,target_tangent_scale=.05):
    """BPTT with no detach between imagined steps and no optimizer side effects.

    observation(state, contact, history, previous_action, goal, step) returns
    (567-vector, 25-vector, updated_history). context(state, contact, previous_prediction) must reconstruct
    environment coordinates. geometry(state) returns six fixed witness points.
    stability(state, prediction, goal) returns a scalar soft penalty, e.g. base
    tilt/height plus supported-CoM, never a hard geometric contact classifier.
    All adapters are explicit and may not consume future reference states.
    """
    goal.validate(len(state),horizon)
    initial_points=geometry(state);previous=initial_action;previous_prediction=None
    predictions=[];points=[];actions=[];priors=[];stable=[]
    with frozen_parameters(world_model):
        for step in range(horizon):
            obs,extra,history=observation(state,contact,history,previous,goal,step)
            action=policy(obs,extra)
            action.retain_grad() if action.requires_grad else None
            if prior is not None:
                with torch.no_grad():priors.append(prior(obs))
            prediction=world_model(state,action,contact,context(state,contact,previous_prediction))
            state,contact=prediction['state'],prediction['contact']
            previous_prediction=prediction
            predictions.append(prediction);actions.append(action);points.append(geometry(state))
            stable.append(stability(state,prediction,goal));previous=action
    losses=contact_objective(predictions,torch.stack(points),initial_points,torch.stack(actions),initial_action,
                             goal,dt,torch.stack(priors) if priors else None,target_tangent_scale)
    losses['stability']=torch.stack(stable).mean()
    losses['loss']=losses['loss']+losses['stability']
    return dict(losses=losses,states=[p['state'] for p in predictions],actions=actions,predictions=predictions)


def gradient_alignment(autograd_gradient,directions,plus_losses,minus_losses,epsilon):
    """Compare WM gradients with externally measured paired simulator probes.

    Caller must use matched initialization/randomness and real simulator losses.
    No result certifies gradients outside these tested action perturbations.
    """
    if epsilon<=0:raise ValueError('Positive finite-difference step required')
    predicted=(directions*autograd_gradient).flatten(1).sum(-1)
    measured=(plus_losses-minus_losses)/(2*epsilon)
    denominator=predicted.norm()*measured.norm()
    cosine=torch.where(denominator>1e-12,(predicted*measured).sum()/denominator.clamp_min(1e-12),torch.nan)
    return dict(predicted=predicted,measured=measured,cosine=cosine,
                sign_agreement=((predicted*measured)>0).float().mean())


def load_php_policy(checkpoint):
    """Initialize the direct conditional policy from the explicit published prior."""
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    validate_g1_asset_metadata(saved['robot_asset'],context='Contact WM policy prior')
    if saved['student_observation_layout']!={'actor_obs':102,'actor_proprio_history':465}:
        raise ValueError('Unexpected PHP observation layout')
    if saved.get('actor_obs_normalizer_state_dict') is not None:
        raise ValueError('This adapter requires explicit normalization support for this checkpoint')
    cfg=saved['experiment_config']['algo']['config'];actor=cfg['module_dict']['actor']
    layer=actor['layer_config']
    if actor['type']!='MLP' or layer['activation']!='ELU' or layer['use_layer_norm'] or layer['dropout_prob']:
        raise ValueError('Unsupported PHP prior architecture')
    widths=[567,*layer['hidden_dims'],29];layers=[]
    for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2:layers.append(nn.ELU())
    prior=nn.Sequential(*layers)
    prefix='actor_module.module.'
    prior.load_state_dict({k[len(prefix):]:v for k,v in saved['actor_model_state_dict'].items() if k.startswith(prefix)},strict=True)
    policy=ContactPolicy(prior,action_clip=cfg.get('action_clip',10.))
    return policy,prior,dict(source=str(checkpoint),iteration=saved['iter'],parts=PARTS,
                             robot_asset=saved['robot_asset'],action_clip=policy.action_clip)
