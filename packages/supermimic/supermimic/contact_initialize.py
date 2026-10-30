"""Fresh native DAgger initialization; offline BC is explicit opt-in only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task_manifest',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--slices',type=Path,help='Audited cases_validated.pt; preserve explicit starts, stops and original goals')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collision_profile',choices=('legacy_v1','surface_v2'),default='surface_v2')
    p.add_argument('--contact_response',choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_soft_v1','manifold_balanced_v1'),default='legacy')
    p.add_argument('--num_envs',type=int,default=4096)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bc_steps',type=int,default=0)
    p.add_argument('--bc_batch',type=int,default=4096)
    p.add_argument('--bc_learning_rate',type=float,default=3e-4)
    p.add_argument('--dagger_rounds',type=int,default=40)
    p.add_argument('--dagger_min_rounds',type=int,default=20)
    p.add_argument('--dagger_updates_per_round',type=int,default=100)
    p.add_argument('--dagger_rollouts_per_segment',type=int,default=1)
    p.add_argument('--dagger_action_noise',type=float,default=.03)
    p.add_argument('--dagger_resume',type=Path)
    p.add_argument('--bc_patience',type=int,default=8)
    p.add_argument('--bc_min_improvement',type=float,default=.001)
    p.add_argument('--bc_relative_improvement',type=float,default=.005)
    p.add_argument('--headless',action='store_true')
    add_launcher_args(p);a=p.parse_args()
    if a.headless:a.visualizer=None
    import torch
    from supermimic.contact_io import load_checkpoint_cpu
    from supermimic.lowstep import install,checkpoint_metadata,validate_checkpoint
    spec=install(a.task_manifest,contact_model='force6',collision_profile=a.collision_profile,contact_response=a.contact_response)
    from supermimic.contact_control import ContactActor,ContactClock,ContactGeometry,observation
    from supermimic.contact_dagger import run_dagger,BalancedReplay
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.networks import Policy
    from supermimic.env import TrackingSim,make_sim
    from supermimic.motion import MotionLib
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP,OBS_DIM,unflatten_state
    from supermimic.robot import default_joint_pos_list
    import supermimic.networks as networks
    torch.manual_seed(a.seed);torch.set_num_threads(4)
    device='cuda:0';metadata=checkpoint_metadata(spec)
    source=load_checkpoint_cpu(a.checkpoint)
    recording=torch.load(a.recording,map_location=device,weights_only=False)
    validate_checkpoint(source,metadata);validate_checkpoint(recording['metadata'],metadata)
    if a.slices:
        import hashlib
        if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=recording['metadata']['source_checkpoint_sha256']:
            raise ValueError('Selected slices require their exact validated tracking teacher')
    demo=recording['data']
    if demo['term'].any():raise ValueError('DAgger requires a successful fresh native demonstration')
    networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
    context=HeightForceContext(spec['tasks']);actor=ContactActor().to(device)
    from supermimic.contact_library import build_contact_clock,load_slice_clock,segment_ranges,history_indices,case_world_counts
    geometry=ContactGeometry(demo,device)
    clock=load_slice_clock(recording,a.slices) if a.slices else build_contact_clock(recording,hold=1,role_aware=True)
    starts,stops=segment_ranges(clock)
    default=torch.tensor(default_joint_pos_list(),device=device)
    teacher_hist=source['args'].get('hist',10);goal_frames=source['args'].get('goal_frames',10)
    teacher=Policy(OBS_DIM,73*goal_frames,hist=teacher_hist).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(source['pi'])
    kin=G1Kinematics(device);motion=MotionLib.from_dir(spec['motion_dir'],kin,device);mdp=MDP(kin,motion,default)
    a.output.mkdir(parents=True,exist_ok=False)
    config=dict(vars(a),wm_implementation='tracking_source_frozen_dagger',
        contact_bounds=clock.bounds,actor_initialization=('random; optional BC then native DAgger' if a.bc_steps else 'random; directly native DAgger; no offline action fitting'),
        source_native_transitions=len(demo['s']),sampling='fresh executed student transitions; repeated simulator runs, no copied data expansion')
    if a.slices:
        import hashlib
        config.update(slices_sha256=hashlib.sha256(a.slices.read_bytes()).hexdigest(),
            case_ids=clock.case_ids.tolist(),case_starts=starts.tolist(),case_stops=stops.tolist(),
            case_goal_indices=(clock.event_ends-1).tolist(),case_source_motions=clock.segment_motion.tolist())
    counts=case_world_counts(clock,len(spec['tasks']),a.num_envs) if hasattr(clock,'segment_motion') else None
    config['task_world_counts']=counts
    (a.output/'config.json').write_text(json.dumps(config,default=str,indent=2)+'\n')
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    started=time.time()
    def status(**fields):
        path=a.output/'status.tmp'
        path.write_text(json.dumps(dict(updated_at=time.time(),elapsed_s=time.time()-started,**fields),indent=2))
        path.replace(a.output/'status.json')
    obs=observation(unflatten_state(demo['s']),demo['a_prev'],demo['xi'],default,context)
    selected_frames=torch.cat([torch.arange(begin,end,device=device) for begin,end in zip(starts.tolist(),stops.tolist())])
    actor.norm.update(obs[selected_frames])
    optimizer=torch.optim.Adam(actor.parameters(),lr=a.bc_learning_rate,betas=(.9,.95))
    if a.bc_steps:
        groups={}
        with torch.no_grad():
            for i,(begin,end) in enumerate(zip(starts.tolist(),stops.tolist())):
                frame=torch.arange(begin,end,device=device)
                indices=history_indices(clock,frame,actor.hist,include_current=True)
                groups[i]=dict(history=obs[indices].detach(),
                    condition=clock.condition(unflatten_state(demo['s'][frame]),frame,ends=clock.event_ends[i].expand_as(frame)).detach(),action=demo['a'][frame].detach())
        bc_replay=BalancedReplay(groups)
        status(phase='bc',updates=0)
        for step in range(a.bc_steps):
            if stopped:raise RuntimeError('Initialization interrupted during BC')
            batch=bc_replay.sample(a.bc_batch)
            loss=10*(actor(batch['history'],batch['condition'])-batch['action']).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step%100==0:status(phase='bc',updates=step+1,loss=float(loss.detach()))
        del groups,bc_replay,batch
    del obs
    status(phase='dagger_sim_initialization',offline_bc_updates=a.bc_steps)
    cfg=make_sim(device,a)
    with launch_simulation(cfg,a):
        env=TrackingSim(cfg,a.num_envs,motion,mdp,eval_mode=True,**({'task_world_counts':counts} if counts else {}))
        chosen=(torch.isin(env.task_ids,clock.segment_motion.unique()).nonzero().flatten()
                if hasattr(clock,'segment_motion') else (env.task_ids==recording['metadata']['motion']).nonzero().flatten())
        if len(chosen)<len(clock.event_ends):raise ValueError('Insufficient native worlds for all contact segments')
        dagger=run_dagger(args=a,env=env,actor=actor,optimizer=optimizer,tracker=teacher,motion=motion,mdp=mdp,
            clock=clock,demo=demo,context=context,default_q=default,chosen=chosen,teacher_hist=teacher_hist,
            goal_frames=goal_frames,status=status,stopped=lambda:stopped)
    if stopped:raise RuntimeError('Initialization interrupted during DAgger')
    saved=dict(schema='supermimic_contact_demo_v1',iteration=0,actor=actor.state_dict(),wm=source['wm'],
        actor_optimizer=optimizer.state_dict(),wm_optimizer={},policy_updates=0,wm_updates=0,
        geometry=geometry.state_dict(),config=config,source_metadata=metadata,dagger=dagger,
        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state())
    torch.save(saved,a.output/'checkpoint_dagger.pt')
    status(phase='complete',dagger=dagger)


if __name__=='__main__':main()
