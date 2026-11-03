"""Compare all-segment DAgger replay strategies from the accepted joint initialization."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_manifest',required=True)
    parser.add_argument('--tracker',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--dagger_run',type=Path,required=True)
    parser.add_argument('--teacher_control',action='store_true',help='Read-only tracking teacher execution from identical segment resets; requires zero training rounds')
    parser.add_argument('--branches',nargs='+',choices=['uniform_history','half_latest'],default=['uniform_history','half_latest'])
    parser.add_argument('--resume_probe',type=Path,help='Load best-mean actor/optimizer and saved aggregate from a completed probe branch')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--publish_dir',type=Path)
    parser.add_argument('--num_envs',type=int,default=4096)
    parser.add_argument('--contact_criteria',type=Path,default=ROOT/'experiments/contact_completion.json')
    parser.add_argument('--grace_steps',type=int,default=10)
    parser.add_argument('--root_tolerance',type=float,default=.25)
    parser.add_argument('--online_rounds',type=int,default=24)
    parser.add_argument('--online_updates',type=int,default=200)
    parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--worlds_saved_per_segment',type=int,default=4)
    parser.add_argument('--online_lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--headless',action='store_true')
    add_launcher_args(parser);args=parser.parse_args()
    if args.headless:args.visualizer=None
    if args.teacher_control and args.online_rounds:raise ValueError('Teacher control is read-only; set online_rounds=0')
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
                execution='All 19 segments simultaneously from recorded initial states; student-only actions, teacher labels on actual states; equal segment sampling, all-replay versus half-latest replay.',
                training=args.online_rounds>0,wm_used_for_execution=False,root_tolerance_m=args.root_tolerance,
                max_steps=max(b-a for a,b in zip(clock.bounds,clock.bounds[1:]))+args.grace_steps)
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
        import copy
        keys=('history','condition','action')
        base=torch.load(args.dagger_run/'checkpoint_dagger.pt',map_location='cpu',weights_only=False)
        original_best=torch.load(args.dagger_run/'dagger/best.pt',map_location='cpu',weights_only=False)
        old_config=json.loads((args.dagger_run/'config.json').read_text())
        paths={}
        for directory in [Path(old_config['dagger_resume']).parent,args.dagger_run/'dagger']:
            for path in sorted(directory.glob('rollout_*.pt')):paths[int(path.stem.split('_')[-1])]=path
        initial={i:[] for i in range(len(clock.event_ends))}
        for round_id,path in sorted(paths.items()):
            if round_id>=original_best['round']:continue
            data=torch.load(path,map_location=device,weights_only=False)['data']
            for i,row in data.items():initial[i].append({k:row[k] for k in keys})
        initial={i:{k:torch.cat([v[k] for v in rows]) for k in keys} for i,rows in initial.items()}
        resumed=None
        if args.resume_probe:
            resumed=torch.load(args.resume_probe/'best_mean.pt',map_location='cpu',weights_only=False)
            initial=torch.load(args.resume_probe/'aggregate.pt',map_location=device,weights_only=False)
        def pack(groups):
            lengths=torch.tensor([len(groups[i]['action']) for i in range(len(initial))],device=device)
            offsets=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)[:-1]))
            return {k:torch.cat([groups[i][k] for i in range(len(initial))]) for k in keys},lengths,offsets
        def draw(packed,segments):
            data,lengths,offsets=packed
            ids=offsets[segments]+(torch.rand(len(segments),device=device)*lengths[segments]).long()
            return {k:v[ids] for k,v in data.items()}
        for branch in args.branches:
            fresh_fraction={'uniform_history':0.,'half_latest':.5}[branch]
            torch.manual_seed(args.seed)
            actor.load_state_dict((resumed or base)['actor']);actor.requires_grad_(True);actor.eval()
            optimizer=torch.optim.Adam(actor.parameters(),lr=args.online_lr)
            optimizer.load_state_dict(copy.deepcopy((resumed or original_best)['optimizer']))
            for param_group in optimizer.param_groups:param_group['lr']=args.online_lr
            groups={i:{k:v.clone() for k,v in row.items()} for i,row in initial.items()}
            branch_dir=args.output/branch;branch_dir.mkdir()
            best_mean=best_max=float('inf');latest=None;branch_results=[]
            for round_id in range(args.online_rounds+1):
                started_round=time.perf_counter();train_mse=None
                if round_id:
                    packed=pack(groups);recent=pack(latest);losses=[]
                    with torch.enable_grad():
                        for update in range(args.online_updates):
                            segments=torch.randint(len(initial),(args.batch,),device=device)
                            n=int(args.batch*fresh_fraction)
                            first=draw(recent,segments[:n]);second=draw(packed,segments[n:])
                            batch={k:torch.cat((first[k],second[k])) for k in keys}
                            mse=(actor(batch['history'],batch['condition'])-batch['action']).square().mean()
                            optimizer.zero_grad(set_to_none=True);(10*mse).backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                            losses.append(mse.detach())
                    train_mse=float(torch.stack(losses[-50:]).mean())
                train_seconds=time.perf_counter()-started_round;collection_started=time.perf_counter()
                task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default,reader,hist,goals,
                                 args.grace_steps,0,args.root_tolerance,'accumulated_target',radii,sequential=True)
                task.segment=torch.arange(task.N,device=device)%len(initial)
                bounds=torch.tensor(clock.bounds,device=device)
                task.starts=bounds[task.segment];task.ends=clock.event_ends[task.segment]
                durations=task.ends-task.starts;task.budgets=durations+args.grace_steps
                task.contact_tolerance=task.contact_tolerances_by_segment[task.segment,None]
                reader.set_targets(demo['point'][task.ends-1],task.ids)
                task.reset(torch.arange(task.N,device=device))
                sample=torch.cat([(task.segment==g).nonzero().flatten()[:args.worlds_saved_per_segment] for g in range(len(initial))])
                sample_groups=task.segment[sample]
                teacher_history=None;labels=None;valid=None
                label_live=torch.ones(task.N,device=device,dtype=torch.bool)
                passed=torch.zeros_like(label_live);mse_sum=torch.zeros(task.N,device=device);label_steps=mse_sum.clone()
                phase_sum=torch.zeros((len(initial),4),device=device);phase_count=phase_sum.clone()
                fresh_rows=[];traces=[];step=0
                final_root=mse_sum.clone();final_angle=mse_sum.clone();min_contact=torch.full_like(mse_sum,float('inf'))
                new=clock.target_masks[task.ends-1]&~clock.keep_masks[task.ends-1]
                def forward(history,condition):
                    nonlocal teacher_history,labels,valid,label_live
                    ids=task.ids;state=tuple(v[ids] for v in env.state())
                    obs=mdp.obs(state,env.a_prev[ids],env.xi[ids],env.f[ids])
                    teacher_history=(obs[:,None].expand(-1,hist,-1).clone() if teacher_history is None else torch.cat((teacher_history[:,1:],obs[:,None]),1))
                    labels=teacher(teacher_history,motion.goal(env.f[ids],goals))
                    label_live&=~mdp.termination(state,env.f[ids])
                    valid=label_live&task.sequence_live&(step<durations)&torch.isfinite(labels).all(-1)
                    chosen=sample[valid[sample]]
                    if len(chosen):fresh_rows.append(dict(history=history[chosen].clone(),condition=condition[chosen].clone(),action=labels[chosen].clone(),segment=task.segment[chosen].clone(),local_step=torch.full_like(chosen,step)))
                    return labels if args.teacher_control else actor(history,condition)
                class RoutedActor:
                    hist=actor.hist
                    def __call__(self,history,condition):return forward(history,condition)
                task.actor=RoutedActor()
                def finish(done,success,transition):
                    passed.logical_or_(success)
                    # Collect original imitation duration even after contact success.
                    # Success is latched, but the teacher never drives task worlds.
                    physical=(~torch.isfinite(transition['s1']).all(-1))|(transition['s1'][:,2]<.15)
                    terminal=task.sequence_live&((task.stage_step>=task.budgets)|physical)
                    transition['stage_done']=terminal;transition['done']=terminal
                    transition['stage_success']=terminal&passed
                    task.sequence_live&=~terminal
                task.finish_sequence=finish
                for step in range(config['max_steps']):
                    row=task.step();live=row['sequence_live'];done=row['stage_done']
                    err=(row['a']-labels).square().mean(-1)
                    mse_sum+=torch.where(valid,err,0.);label_steps+=valid
                    phase=torch.minimum(torch.full_like(durations,step)*4//durations,torch.full_like(durations,3))
                    flat=task.segment*4+phase
                    phase_sum.view(-1).index_add_(0,flat,torch.where(valid,err,0.))
                    phase_count.view(-1).index_add_(0,flat,valid.float())
                    final_root[done]=row['root_error'][done];final_angle[done]=row['angle_error'][done]
                    distances=(row['point']-demo['point'][task.ends-1]).norm(dim=-1)
                    distances=distances.masked_fill(~(row['active']&row['known']&live[:,None]),float('inf'))
                    per_world=distances.masked_fill(~new,0.).amax(-1)
                    min_contact=torch.minimum(min_contact,per_world)
                    saved={k:row[k][sample].clone() for k in ('s','s1','a','stage_step','stage_done','stage_success','sequence_live','point','active','known','contact_goal','root_error','angle_error')}
                    saved.update(action_mse=err[sample].clone(),teacher_valid=valid[sample].clone())
                    traces.append(saved)
                    if not task.sequence_live.any():break
                assert not task.sequence_live.any()
                data={k:torch.cat([r[k] for r in fresh_rows]) for k in fresh_rows[0]}
                latest={i:{k:data[k][data['segment']==i] for k in keys} for i in range(len(initial))}
                per_segment=[]
                for g in range(len(initial)):
                    mask=task.segment==g
                    assert len(latest[g]['action'])>0
                    per_segment.append(dict(segment=g+1,n=int(mask.sum()),passed=int(passed[mask].sum()),
                        action_mse=float((mse_sum[mask]/label_steps[mask].clamp_min(1)).mean()),
                        valid_fraction=float((label_steps[mask]/durations[mask]).mean()),
                        phase_mse=(phase_sum[g]/phase_count[g].clamp_min(1)).tolist(),
                        root_cm=float(final_root[mask].mean()*100),best_contact_cm=float(min_contact[mask].median()*100)))
                mean=sum(v['action_mse'] for v in per_segment)/len(initial);worst=max(v['action_mse'] for v in per_segment)
                result=dict(branch=branch,round=round_id,teacher_control=args.teacher_control,macro_action_mse=mean,worst_segment_mse=worst,
                    passed=int(passed.sum()),attempts=task.N,train_batch_mse=train_mse,segments=per_segment,
                    train_seconds=train_seconds,collection_seconds=time.perf_counter()-collection_started,
                    fresh_fraction=fresh_fraction,learning_rate=optimizer.param_groups[0]['lr'],updates=round_id*args.online_updates)
                branch_results.append(result);results.append(result)
                saved=dict(actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},optimizer=optimizer.state_dict(),round=round_id,report=result,config=base['config'])
                if not args.teacher_control:torch.save(saved,branch_dir/'latest.pt')
                if mean<best_mean:
                    best_mean=mean
                    if not args.teacher_control:torch.save(saved,branch_dir/'best_mean.pt')
                if worst<best_max:
                    best_max=worst
                    if not args.teacher_control:torch.save(saved,branch_dir/'best_worst.pt')
                torch.save(dict(data={k:v.cpu() for k,v in data.items()},report=result),branch_dir/f'rollout_{round_id:04d}.pt')
                torch.save(dict(data={k:torch.stack([r[k] for r in traces]).cpu() for k in traces[0]},segments=sample_groups.cpu(),world_ids=sample.cpu()),branch_dir/'latest_trace.pt')
                for g in groups:
                    groups[g]={k:torch.cat((groups[g][k],latest[g][k])) for k in keys}
                (args.output/'report.json').write_text(json.dumps(dict(config=config,results=results),default=str,indent=2))
                status('training',branch=branch,round=round_id,total_rounds=args.online_rounds)
                with (branch_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                print(json.dumps({k:v for k,v in result.items() if k!='segments'}),flush=True)
            if args.teacher_control:continue
            # Preserve a resumable aggregate and full compatible actor/WM checkpoint.
            torch.save({g:{k:v.cpu() for k,v in row.items()} for g,row in groups.items()},branch_dir/'aggregate.pt')
            best=torch.load(branch_dir/'best_mean.pt',map_location='cpu',weights_only=False)
            full=dict(base);full.update(actor=best['actor'],actor_optimizer=best['optimizer'],dagger_joint_probe=best['report'])
            torch.save(full,branch_dir/'checkpoint_dagger.pt')
        status('complete')

if __name__=='__main__':main()
