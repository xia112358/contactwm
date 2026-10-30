"""Evaluate and continue segment-one DAgger, recording actual-state teacher error."""
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
    parser.add_argument('--online_rounds',type=int,default=12)
    parser.add_argument('--online_updates',type=int,default=100)
    parser.add_argument('--online_lr',type=float,default=1e-4)
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
    assert int(clock.event_ends[0])==82 and args.grace_steps==10, 'This diagnostic uses the original 82-frame first segment and 10-frame grace'
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
                execution='Cold-start segment one; fixed-checkpoint controls then targeted pure DAgger with teacher queried on actual student histories.',
                training=args.online_rounds>0,wm_used_for_execution=False,root_tolerance_m=args.root_tolerance,
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
        from supermimic.mdp import flatten_state
        replay=[];training_metrics=[];optimizer=None
        for case in range(len(args.checkpoints)+args.online_rounds):
            online=case>=len(args.checkpoints)
            if not online:
                path=args.checkpoints[case]
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                actor.load_state_dict(checkpoint['actor']);base_config=checkpoint['config'];del checkpoint
                name=path.parent.name
            else:
                name=f'online-{case-len(args.checkpoints)+1:02d}'
                if optimizer is None:
                    actor.requires_grad_(True)
                    optimizer=torch.optim.Adam(actor.parameters(),lr=args.online_lr)
                history=torch.cat([v['history'] for v in replay]);conditions=torch.cat([v['condition'] for v in replay]);labels=torch.cat([v['action'] for v in replay])
                with torch.enable_grad():
                    for _ in range(args.online_updates):
                        ix=torch.randint(len(labels),(256,),device=device)
                        loss=10*(actor(history[ix],conditions[ix])-labels[ix]).square().mean()
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                with torch.no_grad():
                    error=[]
                    for start in range(0,len(labels),1024):
                        sl=slice(start,start+1024);error.append((actor(history[sl],conditions[sl])-labels[sl]).square().mean(-1))
                    train_mse=float(torch.cat(error).mean())
                training_metrics.append(dict(round=case-len(args.checkpoints)+1,train_mse=train_mse,samples=len(labels)))
            actor.eval()
            task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default,reader,hist,goals,
                             args.grace_steps,0,args.root_tolerance,'accumulated_target',radii,sequential=True)
            teacher_history=None;new_rows=[];teacher_actions=None;teacher_valid=None
            sample=torch.linspace(0,task.N-1,16,device=device).long()
            current_step=0
            def forward(history,condition):
                nonlocal teacher_history,teacher_actions,teacher_valid
                ids=task.ids;state=tuple(x[ids] for x in env.state())
                obs=mdp.obs(state,env.a_prev[ids],env.xi[ids],env.f[ids])
                teacher_history=(obs[:,None].expand(-1,hist,-1).clone() if teacher_history is None else torch.cat((teacher_history[:,1:],obs[:,None]),1))
                teacher_actions=teacher(teacher_history,motion.goal(env.f[ids],goals))
                teacher_valid=~mdp.termination(state,env.f[ids])
                selected=sample[task.sequence_live[sample]&teacher_valid[sample]]
                if current_step<int(clock.event_ends[0]) and len(selected):
                    new_rows.append(dict(history=history[selected].clone(),condition=condition[selected].clone(),action=teacher_actions[selected].clone()))
                return actor(history,condition)
            class RoutedActor:
                hist=actor.hist
                def __call__(self,history,condition):return forward(history,condition)
            task.actor=RoutedActor()
            def finish(done,success,transition):
                task.sequence_progress+=success.long();task.sequence_completed|=success;task.sequence_live&=~done
            task.finish_sequence=finish
            trace=[];all_errors=[];all_valid=[];all_live=[]
            best=torch.full((task.N,6),float('inf'),device=device)
            final_root=torch.zeros(task.N,device=device);final_angle=final_root.clone()
            new=clock.target_masks[81]&~clock.keep_masks[81]
            for current_step in range(92):
                row=task.step();live=row['sequence_live'];done=row['stage_done']
                error=(row['a']-teacher_actions).square().mean(-1)
                valid=teacher_valid&live&(current_step<82)
                all_errors.append(error.clone());all_valid.append(valid.clone());all_live.append(live.clone())
                observed=row['active']&row['known']&live[:,None]
                distance=(row['point']-demo['point'][81]).norm(dim=-1)
                best=torch.minimum(best,distance.masked_fill(~observed,float('inf')))
                final_root[done]=row['root_error'][done];final_angle[done]=row['angle_error'][done]
                saved={k:row[k][sample].clone() for k in ('s','s1','a','stage_step','stage_done','stage_success','sequence_live','point','active','known','contact_goal','root_error','angle_error')}
                saved.update(teacher_action=teacher_actions[sample].clone(),teacher_valid=teacher_valid[sample].clone(),action_mse=error[sample].clone())
                trace.append(saved)
                if not task.sequence_live.any():break
            errors=torch.stack(all_errors);valid=torch.stack(all_valid);live=torch.stack(all_live)
            # Each fixed fit is evaluated independently. Online continuation starts
            # from the final (aggregate-data) fit and its freshly collected states.
            if online or case==len(args.checkpoints)-1:
                replay.append({k:torch.cat([v[k] for v in new_rows]) for k in new_rows[0]})
            result=dict(case=name,online=online,attempts=task.N,passed=int(task.sequence_completed.sum()),
                on_policy_valid_action_mse=float(errors[valid].mean()),teacher_valid_fraction=float(valid[:82].sum()/live[:82].sum()),
                per_step_action_mse=[float(e[v].mean()) if v.any() else None for e,v in zip(errors,valid)],
                root_cm=float(final_root.mean()*100),angle_rad=float(final_angle.mean()),
                best_contact_cm=[float(x) for x in best[:,new].median(0).values*100],
                training=training_metrics[-1] if online else None)
            results.append(result);output=args.output/name;output.mkdir(exist_ok=True)
            torch.save(dict(data={k:torch.stack([v[k] for v in trace]).cpu() for k in trace[0]},world_ids=sample.cpu()),output/'rollout.pt')
            if online:
                torch.save(dict(actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},config=base_config,policy_updates=0,diagnostic=result),output/'checkpoint.pt')
            (args.output/'report.json').write_text(json.dumps(dict(config=config,results=results),default=str,indent=2))
            status('executing',current_case=name);print(json.dumps(result),flush=True)
        status('complete')

if __name__=='__main__':main()
