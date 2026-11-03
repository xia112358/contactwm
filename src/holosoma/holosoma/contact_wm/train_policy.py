"""Train a direct goal-contact policy through the calibrated WM, then export it.

Single-transition experiment. Native evaluation is mandatory; imagined success
is explicitly not physical success. No future reference frames enter the policy.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from holosoma.contact_wm.core import load_php_policy,ContactGoal,unroll
from holosoma.contact_wm.supermimic import SuperMimicForceWM
from holosoma.contact_wm.observations import PHPObservation,unpack_history,rotation_xyzw
from holosoma.contact_wm.height_context import HeightRaster
from holosoma.contact_wm.geometry import WitnessGeometry,LINKS
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics


@dataclass(frozen=True)
class Options:
    recording:Path
    atlas:Path
    wm_checkpoint:Path
    output_dir:Path
    prior_checkpoint:Path|None=None
    start:int=266
    arrival:int=286
    hold:int=10
    steps:int=100
    learning_rate:float=1e-5
    device:str='cpu'
    seed:int=19
    resume:Path|None=None
    goal_offset:tuple[float,float,float]=(0.,0.,0.)
    planned_contact_points:Path|None=None
    motion_penalty:float=.01
    target_tangent_scale:float=.05
    save_every:int=25
    initial_recordings:tuple[Path,...]=()
    multi_target:bool=False
    target_span:float=.04
    arrival_delay:int=0
    target_learning_rate_scale:float=1.
    source_wm:bool=False
    panorama:bool=False
    rotations:Path|None=None


class CalibratedDynamics(nn.Module):
    """Empirically adapted Newton model, without asserting solver equivalence."""
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,state,action,contact,context):
        return self.model.source_step(state,action,context['physics'],context['height'],context['force'])


def build_training(o):
    import math
    if not math.isfinite(o.target_tangent_scale) or o.target_tangent_scale<=0:
        raise ValueError("Positive finite target tangent scale required")
    """Build the shared task and learner; online training supplies native starts."""
    torch.manual_seed(o.seed);torch.set_num_threads(1)
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((o.recording/'report.json').read_text());z=np.load(o.recording/'transitions.npz')
    if o.prior_checkpoint is not None:meta['checkpoint']=str(o.prior_checkpoint.resolve())
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    validate_g1_asset_metadata(meta.get('robot_asset'),context='Contact WM native recording')
    if 'reference_origin_w' not in meta:
        origin_directory=o.atlas.parent/f"motion_{meta['motion_index']:03d}"
        origin_meta=json.loads((origin_directory/'report.json').read_text())
        origin_state=np.load(origin_directory/'transitions.npz')['state'][0,:3]
        if not np.allclose(origin_state,z['state'][0,:3],atol=1e-5):raise ValueError('Atlas origin does not match recording')
        meta['reference_origin_w']=origin_meta['reference_origin_w']
    event=next(e for e in meta['transitions'] if e['arrival']==o.arrival)
    if o.start<1 or o.start>=o.arrival or o.arrival+o.hold>len(z['state']):raise ValueError('Invalid transition window')
    tensor=lambda a:torch.as_tensor(a,device=o.device,dtype=torch.float32)
    state=tensor(z['state'][o.start:o.start+1]);history=unpack_history(tensor(z['observation'][o.start:o.start+1,102:]))
    action=tensor(z['action'][o.start-1:o.start]);contact=tensor(z['observation'][o.start:o.start+1,:6])
    known=torch.tensor(z['previous_known'][o.start:o.start+1],device=o.device)
    if not known.all():raise ValueError('Unknown starting contact evidence')
    initial_force=tensor(z['previous_force_w'][o.start:o.start+1])
    goal=ContactGoal(tensor(z['target_position_w'][o.start:o.start+1]),tensor([[[0,0,1]]*6]),
        torch.tensor([event['establish']],device=o.device,dtype=torch.bool),
        torch.tensor([event['maintain']],device=o.device,dtype=torch.bool),
        tensor(z['solver_point_w'][o.start-1:o.start]),tensor([[1.]*6]),o.arrival-o.start+1,o.hold)
    goal.min_normal_force=torch.where(goal.maintain,10.,goal.min_normal_force)
    if o.planned_contact_points is not None:
        if any(o.goal_offset):raise ValueError('Explicit planned points cannot be combined with goal offsets')
        from holosoma.contact_wm.core import load_planned_contact_points
        planned=load_planned_contact_points(o.planned_contact_points,device=o.device)
        goal.position_w=torch.where(goal.establish[...,None],planned,goal.position_w)
    else:
        goal.position_w=goal.position_w+goal.establish[...,None]*tensor(o.goal_offset)
    target_span=0. if o.planned_contact_points is not None else o.target_span
    goal.arrival_step+=o.arrival_delay
    horizon=goal.arrival_step+goal.hold_steps-1
    # Robot-side witnesses from native labels; never replace desired targets by achieved points.
    order=[meta['joint_names'].index(n) for n in G1_29DOF_JOINT_ORDER]
    poses=tensor(z['next_state'][[o.start-1,o.arrival]])
    qpos=torch.cat((poses[:,:3],poses[:,[6,3,4,5]],poses[:,13:42][:,order]),-1)
    p,R=BatchedG1ForwardKinematics().link_poses(qpos,LINKS)
    point=tensor(z['solver_point_w'][[o.start-1,o.arrival]])
    local=(R.transpose(-1,-2)@(point-p)[...,None]).squeeze(-1)
    local=torch.where(goal.establish[0,:,None],local[1],local[0])
    geometry=WitnessGeometry(meta['joint_names'],local).to(o.device)
    if o.multi_target and o.initial_recordings:raise ValueError('Matched-target experiment requires one identical initial state')
    records=[z]*9 if o.multi_target else [z]+[np.load(p/'transitions.npz') for p in o.initial_recordings]
    batch=len(records)
    if batch>1:
        state=torch.cat([tensor(r['state'][o.start:o.start+1]) for r in records])
        history=unpack_history(torch.cat([tensor(r['observation'][o.start:o.start+1,102:]) for r in records]))
        action=torch.cat([tensor(r['action'][o.start-1:o.start]) for r in records])
        contact=torch.cat([tensor(r['observation'][o.start:o.start+1,:6]) for r in records])
        known=torch.cat([torch.tensor(r['previous_known'][o.start:o.start+1],device=o.device) for r in records])
        initial_force=torch.cat([tensor(r['previous_force_w'][o.start:o.start+1]) for r in records])
        for k,v in vars(goal).items():
            if isinstance(v,torch.Tensor):setattr(goal,k,v.expand(batch,*v.shape[1:]).clone())
        goal.support_position_w=torch.cat([tensor(r['solver_point_w'][o.start-1:o.start]) for r in records])
        if not known.all() or not (contact.bool()|~goal.maintain).all():raise ValueError('Missing maintained support in initial-state replay')
    a=np.load(o.atlas);height=HeightRaster(tensor(a['height'][:1]).expand(batch,-1,-1),tensor(a['minimum'][:1]).expand(batch,-1),float(a['resolution'])).to(o.device)
    php_height=HeightRaster(tensor(a['height'][:1]).expand(batch,-1,-1),tensor(a['minimum'][:1]).expand(batch,-1),float(a['resolution']),php=True).to(o.device)
    hint=torch.cat([tensor(r['observation'][o.start:o.start+1,36:39]) for r in records])
    observation=PHPObservation(tensor(meta['default_joint_pos']),php_height,hint,known).to(o.device)
    policy,prior,provenance=load_php_policy(meta['checkpoint']);policy.to(o.device);prior.to(o.device).requires_grad_(False)
    from somaforge_core.collision_profiles import require_profile
    profile=meta['robot_asset']['collision_profile']
    require_profile(meta['robot_asset'],'surface_v2',context='Contact WM recording')
    require_profile(provenance['robot_asset'],profile,context='Contact WM policy prior versus recording')
    if o.panorama:
        from holosoma.contact_wm.panorama_policy import PanoramaContactPolicy
        policy=PanoramaContactPolicy(policy,tensor(z['observation'][o.start,39:102])).to(o.device)
    if o.target_learning_rate_scale<=0:raise ValueError('Positive goal learning-rate scale required')
    wm=SuperMimicForceWM(o.rotations).to(o.device)
    if o.source_wm:wm.load_source(o.wm_checkpoint)
    else:
        saved=torch.load(o.wm_checkpoint,map_location=o.device,weights_only=False);wm.load_state_dict(saved['wm'],strict=True)
        validate_g1_asset_metadata(saved.get('robot_asset'),context='Contact WM checkpoint')
        require_profile(saved.get('robot_asset'),profile,context='Contact WM versus recording')
    if o.source_wm:
        require_profile(wm.transfer_report['robot_asset'],profile,context='Source dynamics versus recording')
    dynamics=CalibratedDynamics(wm);physics=state.new_zeros(batch,33);physics[:,0]=(1.-.75)/.45
    context=lambda s,c,p:dict(physics=physics,height=height(s),force=initial_force if p is None else p['force_w'])
    initial_up=rotation_xyzw(state[:,3:7])[:,2,2].detach()
    from holosoma.contact_wm.target_response import target_grid,response_metrics
    base_target=goal.position_w.clone()
    offsets=target_grid(target_span,o.device) if o.multi_target else state.new_zeros(batch,3)
    if o.multi_target:
        if goal.establish[0].sum()!=1:raise ValueError('Response experiment requires one new contact')
        part=int(goal.establish[0].nonzero()[0])
        # Terrain must actually exist at every perturbed goal. Preserve the
        # reference contact-to-surface offset rather than inventing new heights.
        positions=base_target[:,part]+offsets
        atlas_min=tensor(a['minimum'][0]);indices=((positions[:,:2]-atlas_min)/float(a['resolution'])).round().long()
        raster=tensor(a['height'][0])
        if (indices<0).any() or (indices[:,0]>=raster.shape[1]).any() or (indices[:,1]>=raster.shape[0]).any():raise ValueError('Goal outside terrain')
        levels=raster[indices[:,1],indices[:,0]]
        if (levels-levels[4]).abs().max()>.005:raise ValueError('Target grid crosses surfaces; choose a valid same-platform grid')
    def stability(s,p,g):
        upright=rotation_xyzw(s[:,3:7])[:,2,2]
        return 100*torch.relu(torch.maximum(initial_up-.05,initial_up.new_tensor(.3))-upright).square().mean()+10*torch.relu(state[:,2]-.12-s[:,2]).square().mean()
    def rollout(target_offsets=None):
        if o.planned_contact_points is not None and target_offsets is not None:
            raise ValueError('Explicit planned points must not be perturbed during evaluation')
        goal.position_w=base_target+goal.establish[...,None]*(offsets if target_offsets is None else target_offsets)[:,None,:]
        result=unroll(policy,dynamics,state,contact,history,action,goal,horizon=horizon,
            observation=observation,geometry=geometry,context=context,stability=stability,prior=prior,target_tangent_scale=o.target_tangent_scale)
        result['losses']['loss']=result['losses']['loss']+(o.motion_penalty-.1)*result['losses']['motion']
        if o.multi_target:
            endpoints=torch.stack([geometry(s)[:,part] for s in result['states'][-o.hold:]]).mean(0)
            result['response']=response_metrics(endpoints,goal.position_w[:,part],exclude_center=target_offsets is not None)
        return result
    optimizer=torch.optim.Adam(policy.parameters(),lr=o.learning_rate)
    iteration=0
    if o.resume is not None:
        previous=torch.load(o.resume,map_location=o.device,weights_only=False)
        if previous['schema']!='contact_wm_direct_policy_v1':raise ValueError('Invalid policy resume schema')
        policy.load_state_dict(previous['policy'],strict=True);optimizer.load_state_dict(previous['optimizer'])
        for group in optimizer.param_groups:group['lr']=o.learning_rate
        iteration=int(previous['iteration']);torch.set_rng_state(previous['torch_rng_state'].cpu())
    def export(iteration,result):
        if 'response' in result:(out/f'response_{iteration:04d}.json').write_text(json.dumps(result['response'],indent=2))
        torch.save(dict(schema='contact_wm_direct_policy_v1',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
            iteration=iteration,prior=provenance,wm_checkpoint=str(o.wm_checkpoint.resolve()),
            goal={k:(base_target[:1] if k=='position_w' else v[:1]).detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in vars(goal).items()},
            goal_hint=hint[:1].cpu(),local_witness=local.cpu(),start_step=o.start,motion_index=meta['motion_index'],
            horizon=horizon,reference_origin_w=meta['reference_origin_w'],torch_rng_state=torch.get_rng_state(),
            options={k:str(v) if isinstance(v,Path) else v for k,v in asdict(o).items()},
            native_success_verified=False),out/f'policy_{iteration:04d}.pt')
        np.savez_compressed(out/f'imagined_{iteration:04d}.npz',
            states=torch.stack(result['states']).detach().cpu().numpy(),actions=torch.stack(result['actions']).detach().cpu().numpy())
    def set_start(sample,age):
        nonlocal state,history,action,contact,known,initial_force,initial_up,horizon
        expand=lambda x:x.expand(batch,*x.shape[1:]).clone()
        state=expand(sample['state']);history=unpack_history(expand(sample['observation'][:,102:]))
        action=history[:,-1,:29];contact=expand(sample['observation'][:,:6])
        known=expand(sample['previous_known']);initial_force=expand(sample['previous_force_w'])
        observation.known_contact=known
        initial_up=rotation_xyzw(state[:,3:7])[:,2,2].detach()
        goal.support_position_w=expand(sample['support'])
        goal.arrival_step=o.arrival-o.start+1+o.arrival_delay-(age-5)
        horizon=goal.arrival_step+goal.hold_steps-1
    return dict(policy=policy,prior=prior,wm=wm,optimizer=optimizer,rollout=rollout,export=export,
                set_start=set_start,goal=goal,meta=meta,z=z,geometry=geometry,local=local,hint=hint,
                base_target=base_target,target_span=target_span,target_tangent_scale=o.target_tangent_scale,provenance=provenance,out=out,iteration=iteration,batch=batch,horizon=horizon)


def main():
    import tyro
    from holosoma.contact_wm.target_response import target_grid
    o=tyro.cli(Options)
    setup=build_training(o)
    policy=setup['policy']
    optimizer=setup['optimizer']
    rollout=setup['rollout']
    export=setup['export']
    out=setup['out']
    iteration=setup['iteration']
    batch=setup['batch']
    meta=setup['meta']
    horizon=setup['horizon']
    first=rollout();export(iteration,first);before={k:float(v.detach()) for k,v in first['losses'].items()}
    response_before=first.get('response');del first
    heldout_before=None
    if o.multi_target and o.planned_contact_points is None:
        with torch.no_grad():heldout_before=rollout(target_grid(o.target_span/2,o.device))['response']
    with (out/'metrics.jsonl').open('x') as log:
        for step in range(o.steps):
            optimizer.zero_grad(set_to_none=True);r=rollout();loss=r['losses']['loss']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite policy loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            before_goal_weights=policy.net[0].weight[:,12:36].detach().clone()
            optimizer.step()
            # Scale the actual Adam update, not its gradient (Adam would mostly
            # cancel a constant gradient scaling through its second moment).
            with torch.no_grad():
                w=policy.net[0].weight[:,12:36]
                w.add_((o.target_learning_rate_scale-1)*(w-before_goal_weights))
            row=dict(step=iteration+step+1,gradient_norm=float(norm),**{k:float(v.detach()) for k,v in r['losses'].items()})
            if 'response' in r:
                row['target_response_gain_xy']=r['response']['target_to_endpoint_jacobian_xy']
            log.write(json.dumps(row)+'\n');log.flush()
            if step%10==0:print(json.dumps(row),flush=True)
            del r
            if o.save_every>0 and (step+1)%o.save_every==0 and step+1<o.steps:
                with torch.no_grad():snapshot=rollout();export(iteration+step+1,snapshot)
    final=rollout();export(iteration+o.steps,final)
    heldout_after=None
    if o.multi_target and o.planned_contact_points is None:
        with torch.no_grad():heldout_after=rollout(target_grid(o.target_span/2,o.device))['response']
    report=dict(before=before,after={k:float(v.detach()) for k,v in final['losses'].items()},steps=o.steps,
                motion=meta['motion_index'],start=o.start,arrival=o.arrival,horizon=horizon,
                initial_state_replay=batch,motion_penalty=o.motion_penalty,
                multi_target=o.multi_target,response_before=response_before,response_after=final.get('response'),
                heldout_response_before=heldout_before,heldout_response_after=heldout_after,
                policy_updated=True,native_success_verified=False,oracle_terrain=True,
                limitation='single-transition BPTT experiment; empirical WM gradients unverified at switching')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
