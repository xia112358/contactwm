"""Audit segment-one cold start using recorded warm states and matched histories."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
pass


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_manifest',required=True)
    parser.add_argument('--tracker',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish_dir',type=Path)
    parser.add_argument('--num_envs',type=int,default=4096)
    parser.add_argument('--contact_criteria',type=Path,default=ROOT/'experiments/contact_completion.json')
    parser.add_argument('--grace_steps',type=int,default=10)
    parser.add_argument('--root_tolerance',type=float,default=.25)
    parser.add_argument('--reset_history_on_handoff',action='store_true',
                        help='Diagnostic ablation: repeat current observation at handoff; never reset physical state')
    parser.add_argument('--full_budget',action='store_true',help='Give warm starts the full 92-step segment budget')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--headless',action='store_true')
    add_launcher_args(parser);args=parser.parse_args()
    if args.headless:args.visualizer=None
    import torch
    from supermimic.lowstep import install,checkpoint_metadata,validate_checkpoint
    spec=install(args.task_manifest,contact_model='force6')
    from supermimic.env import TrackingSim,make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.motion import MotionLib
    from supermimic.mdp import MDP,OBS_DIM,unflatten_state
    from supermimic.networks import Policy
    from supermimic import networks
    from supermimic.robot import default_joint_pos_list
    from supermimic.lowstep_observations import HeightForceContext
    from supermimic.contact_control import ContactActor,ContactClock,load_contact_tolerances,observation
    from supermimic.contact_native import NativeContacts
    from supermimic.contact_task import ContactTask
    from supermimic.rotations import rot_angle_sq
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda:0';metadata=checkpoint_metadata(spec)
    source=torch.load(args.tracker,map_location=device,weights_only=False);validate_checkpoint(source,metadata)
    networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
    rec=torch.load(args.recording,map_location=device,weights_only=False);validate_checkpoint(rec['metadata'],metadata)
    demo=rec['data'];clock=ContactClock(demo,hold=1,role_aware=True)
    radii=load_contact_tolerances(clock,rec['metadata'],args.contact_criteria)
    default=torch.tensor(default_joint_pos_list(),device=device)
    kin=G1Kinematics(device);motion=MotionLib.from_dir(spec['motion_dir'],kin,device);mdp=MDP(kin,motion,default)
    hist=source['args'].get('hist',10);goals=source['args'].get('goal_frames',10)
    teacher=Policy(OBS_DIM,73*goals,hist=hist).to(device).eval().requires_grad_(False)
    teacher.load_state_dict(source['pi']);del source
    actor=ContactActor().to(device).eval().requires_grad_(False);context=HeightForceContext(spec['tasks'])
    counts=[1]*len(spec['tasks']);counts[rec['metadata']['motion']]=args.num_envs-len(counts)+1
    assert min(counts)>0
    args.output.mkdir(parents=True,exist_ok=False)
    if args.publish_dir:args.publish_dir.mkdir(parents=True,exist_ok=True)
    config=dict(vars(args),contact_bounds=clock.bounds,contact_tolerances_m=radii,
                execution='Segment one only; grouped recorded-state starts at frames 0/10/20/40/60/70, repeated versus recorded sensor history, original tracking cold-start control.',
                training=False,wm_used_for_execution=False,root_tolerance_m=args.root_tolerance,
                max_steps=92)
    (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
    started=time.perf_counter();results=[]

    def status(phase,**extra):
        payload=dict(phase=phase,elapsed_s=time.perf_counter()-started,results=results,**extra)
        for directory in [args.output]+([args.publish_dir] if args.publish_dir else []):
            tmp=directory/'status.tmp';tmp.write_text(json.dumps(payload,indent=2));tmp.replace(directory/'status.json')

    status('initializing')
    cfg=make_sim(device,args)
    with launch_simulation(cfg,args),torch.no_grad():
        env=TrackingSim(cfg,args.num_envs,motion,mdp,eval_mode=True,task_world_counts=counts)
        reader=NativeContacts(env,direct=True)
        variants=[('cold',0,False),('frame10_repeat',10,False),('frame10_history',10,True),
                  ('frame20_repeat',20,False),('frame20_history',20,True),
                  ('frame40_repeat',40,False),('frame40_history',40,True),
                  ('frame60_repeat',60,False),('frame60_history',60,True),
                  ('frame70_repeat',70,False),('frame70_history',70,True),('teacher_cold',0,False)]
        for path in args.checkpoints:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            actor.load_state_dict(checkpoint['actor']);del checkpoint
            task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default,reader,hist,goals,
                             args.grace_steps,0,args.root_tolerance,'accumulated_target',radii,sequential=True)
            group=torch.arange(task.N,device=device)%len(variants)
            offsets=torch.tensor([v[1] for v in variants],device=device)[group]
            task.starts.copy_(offsets);task.reset(torch.arange(task.N,device=device))
            if not args.full_budget:task.budgets-=offsets
            task.starts.zero_() # Identical segment-one command, only reset state changes.
            full_obs=observation(unflatten_state(demo['s']),demo['a_prev'],demo['xi'],default,context)
            history_index=(offsets[:,None]+torch.arange(-actor.hist+1,1,device=device)).clamp_min(0)
            seeded=full_obs[history_index]
            seed_mask=torch.tensor([v[2] for v in variants],device=device)[group]
            teacher_mask=group==len(variants)-1
            teacher_history=None
            first=True
            def forward(history,condition):
                nonlocal first,teacher_history
                if first:
                    history[seed_mask]=seeded[seed_mask];first=False
                action=actor(history,condition)
                ids=task.ids[teacher_mask]
                state=tuple(v[ids] for v in env.state())
                obs=mdp.obs(state,env.a_prev[ids],env.xi[ids],env.f[ids])
                teacher_history=(obs[:,None].expand(-1,hist,-1).clone() if teacher_history is None
                                 else torch.cat((teacher_history[:,1:],obs[:,None]),1))
                action[teacher_mask]=teacher(teacher_history,motion.goal(env.f[ids],goals))
                return action
            class RoutedActor:
                hist=actor.hist
                def __call__(self,history,condition):return forward(history,condition)
            task.actor=RoutedActor()
            def finish(done,success,transition):
                task.sequence_progress+=success.long()
                task.sequence_completed|=success
                task.sequence_live&=~done
            task.finish_sequence=finish
            records=[]
            for step in range(92):
                row=task.step()
                records.append({k:row[k].clone() for k in ('s','s1','a','goal_index','stage_step','stage_done','stage_success','sequence_live','point','active','known','contact_goal','root_error','angle_error')})
                if not task.sequence_live.any():break
            data={k:torch.stack([r[k] for r in records]) for k in records[0]}
            stop=data['stage_done'].long().argmax(0);batch=torch.arange(task.N,device=device)
            new=clock.target_masks[81]&~clock.keep_masks[81]
            dist=(data['point']-demo['point'][81]).norm(dim=-1)
            observed=data['active']&data['known']&data['sequence_live'][...,None]
            best=dist.masked_fill(~observed,float('inf')).amin(0)
            rows=[]
            for g,(name,start,seeded_history) in enumerate(variants):
                mask=group==g;ix=batch[mask];last=stop[mask]
                rows.append(dict(variant=name,start_frame=start,recorded_history=seeded_history,n=int(mask.sum()),
                    passed=int(task.sequence_completed[mask].sum()),
                    mean_steps=float((last+1).float().mean()),
                    root_cm=float(data['root_error'][last,ix].mean()*100),
                    angle_rad=float(data['angle_error'][last,ix].mean()),
                    best_contact_cm=[float(v) for v in best[mask][:,new].median(0).values*100]))
            result=dict(checkpoint=str(path),variants=rows)
            results.append(result)
            out=args.output/path.parent.name;out.mkdir(exist_ok=True)
            torch.save(dict(data={k:v.cpu() for k,v in data.items()},group=group.cpu(),offsets=offsets.cpu()),out/'rollout.pt')
            status('checkpoint_complete',checkpoint=str(path))
            print(json.dumps(result),flush=True)
        (args.output/'report.json').write_text(json.dumps(dict(config=config,results=results),default=str,indent=2))
        status('complete')

if __name__=='__main__':main()
