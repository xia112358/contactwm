"""Execute trained contact policies continuously, without resets between goals."""
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
    parser.add_argument('--contact_response', choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_v3','manifold_soft_v1','manifold_balanced_v1'), default=None)
    parser.add_argument('--collision_profile',choices=('legacy_v1','surface_v2'),default=None,
                        help='Defaults to the recording physical revision')
    parser.add_argument('--allow_collision_transfer',action='store_true',help='Allow a legacy tracker as initialization only')
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
    parser.add_argument('--contact_only',action='store_true',help='Diagnostic: requested body contact alone passes; ignore placement/root/orientation')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--headless',action='store_true')
    add_launcher_args(parser);args=parser.parse_args()
    if args.headless:args.visualizer=None
    import torch
    from supermimic.lowstep import install,checkpoint_metadata,validate_checkpoint
    recording_header=torch.load(args.recording,map_location='cpu',weights_only=False)['metadata']
    profile=args.collision_profile or recording_header.get('collision_profile','legacy_v1')
    spec=install(args.task_manifest,contact_model='force6',collision_profile=profile,contact_response=args.contact_response or json.loads(recording_header.get('contact_response_contract','{"profile":"legacy"}'))['profile'])
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
    source=torch.load(args.tracker,map_location=device,weights_only=False)
    validate_checkpoint(source,metadata,allow_collision_transfer=args.allow_collision_transfer)
    networks.AMP_DTYPE=torch.bfloat16 if source['args'].get('amp')=='bf16' else None
    rec=torch.load(args.recording,map_location=device,weights_only=False);validate_checkpoint(rec['metadata'],metadata)
    if rec['metadata'].get('recording_spans'):
        raise ValueError('A cross-motion transfer library uses independent episodes; continuous replay requires one source motion')
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
                execution='One reset at sequence start; switch goals on original accumulated-contact/root criterion; retain physical state, velocity and previous action. '+
                          ('Diagnostic: repeat current observation to rebuild actor history at handoff.' if args.reset_history_on_handoff else 'Retain full sensor history.'),
                training=False,wm_used_for_execution=False,root_tolerance_m=args.root_tolerance,
                max_steps=sum(b-a+args.grace_steps for a,b in zip(clock.bounds,clock.bounds[1:])))
    if args.contact_only:config['execution']='Requested parts accumulate any actual contact; no placement, root or orientation gate. No physical/history resets at handoff. Physical failure and per-stage deadlines remain.'
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
        for path in args.checkpoints:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            validate_checkpoint(checkpoint.get('source_metadata',{}),metadata)
            assert Path(checkpoint['config']['recording']).resolve()==args.recording.resolve()
            actor.load_state_dict(checkpoint['actor']);saved_actor={k:v.clone() for k,v in actor.state_dict().items()}
            updates=int(checkpoint['policy_updates']);del checkpoint
            output=args.output/f'update-{updates}';output.mkdir()
            task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default,reader,hist,goals,
                             args.grace_steps,None,args.root_tolerance,'accumulated_target',radii,sequential=True,contact_only=args.contact_only)
            student=torch.zeros(env.N,dtype=torch.bool,device=device);student[task.ids]=True
            original_reset=env.reset;reset_calls=0

            def checked_reset(ids,*a,**kw):
                nonlocal reset_calls
                if bool(student[ids].any()):
                    reset_calls+=1
                    raise AssertionError('Student reset during continuous execution')
                return original_reset(ids,*a,**kw)

            env.reset=checked_reset
            records=[];entry_records=[];previous=None;max_continuity=0.
            fields=('s','s1','a','a_prev','xi','next_xi','ep_step','goal_index','stage_step','stage_done','stage_success',
                    'sequence_live','done','term','point','active','known','contact_goal','root_error','angle_error')
            try:
                for step in range(config['max_steps']):
                    row=task.step();live=row['sequence_live']
                    if args.reset_history_on_handoff:
                        handoff=row['stage_success']&~row['done']
                        if handoff.any():
                            current=tuple(v[task.ids] for v in env.state())
                            obs=observation(current,env.a_prev[task.ids],env.xi[task.ids],default,context)
                            task.history[handoff]=obs[handoff,None].expand(-1,actor.hist,-1)
                    assert bool((row['ep_step'][live]==step).all()),'Global step reset at handoff'
                    if previous is not None:
                        mismatch=float((row['s'][live]-previous[live]).abs().max()) if live.any() else 0.
                        max_continuity=max(max_continuity,mismatch)
                        assert mismatch<1e-4,('State discontinuity at handoff',step,mismatch)
                    previous=row['s1']
                    records.append({k:row[k].clone() for k in fields})
                    entries=live&(row['stage_step']==0)
                    if entries.any():
                        ids=entries.nonzero().flatten();segments=row['goal_index'][ids]
                        starts=torch.as_tensor(clock.bounds,device=device)[segments]
                        actual=unflatten_state(row['s'][ids]);reference=unflatten_state(demo['s'][starts])
                        entry_records.append(dict(world=ids,segment=segments,
                            root_cm=(actual[0]-reference[0]).norm(dim=-1)*100,
                            angle_rad=rot_angle_sq(actual[1],reference[1]).sqrt(),
                            velocity_m_s=(actual[2]-reference[2]).norm(dim=-1),
                            joint_rms_rad=(actual[4]-reference[4]).square().mean(-1).sqrt()))
                    if step%25==0 or not bool(task.sequence_live.any()):
                        status('executing',policy_updates=updates,step=step+1,active=int(task.sequence_live.sum()),
                               completed=int(task.sequence_completed.sum()),attempts=task.N,
                               progress_histogram=torch.bincount(task.sequence_progress,minlength=task.count+1).tolist())
                    if not bool(task.sequence_live.any()):break
                assert not bool(task.sequence_live.any()),'Sequence exceeded sum of per-stage budgets'
            finally:
                env.reset=original_reset
            data={k:torch.stack([row[k] for row in records]) for k in fields};del records
            entries={k:torch.cat([row[k] for row in entry_records]) for k in entry_records[0]}
            stop=data['done'].long().argmax(0);batch=torch.arange(task.N,device=device)
            assert data['done'].any(0).all()
            final_goal=data['goal_index'][stop,batch]
            segments=[]
            for g in range(task.count):
                ended=data['stage_done']&(data['goal_index']==g)
                passed=ended&data['stage_success'];failed=ended&~data['stage_success']
                entry=entries['segment']==g
                segments.append(dict(segment=g+1,entered=int(entry.sum()),passed=int(passed.sum()),failed=int(failed.sum()),
                    mean_entry_root_cm=float(entries['root_cm'][entry].mean()) if entry.any() else None,
                    mean_entry_velocity_error_m_s=float(entries['velocity_m_s'][entry].mean()) if entry.any() else None,
                    mean_entry_angle_rad=float(entries['angle_rad'][entry].mean()) if entry.any() else None,
                    mean_entry_joint_rms_rad=float(entries['joint_rms_rad'][entry].mean()) if entry.any() else None,
                    mean_failed_root_cm=float(data['root_error'][failed].mean()*100) if failed.any() else None,
                    mean_failed_angle_rad=float(data['angle_error'][failed].mean()) if failed.any() else None))
            progress=task.sequence_progress
            median=int(progress.argsort()[len(progress)//2]);furthest=int(progress.argmax())
            examples=[median]+([furthest] if int(progress[furthest])>int(progress[median]) else [])
            example_reports=[]
            labels=['左脚','右脚','左手','右手','左膝','右膝']
            for i in examples:
                last=int(stop[i]);g=int(final_goal[i]);end=int(clock.event_ends[g]);new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
                current=(data['goal_index'][:last+1,i]==g)&data['sequence_live'][:last+1,i]
                observed=data['active'][:last+1,i]&data['known'][:last+1,i]&current[:,None]
                distances=(data['point'][:last+1,i]-demo['point'][end-1]).norm(dim=-1)
                best=distances.masked_fill(~observed,float('inf')).amin(0)
                parts=[dict(part=labels[j],best_contact_cm=float(best[j]*100) if torch.isfinite(best[j]) else None,
                            achieved=bool((data['contact_goal'][:last+1,i,j]&current).any())) for j in new.nonzero().flatten().tolist()]
                z=float(data['s1'][last,i,2]);physical=not bool(torch.isfinite(data['s1'][last,i]).all()) or z<.15
                example_reports.append(dict(world=i,completed_segments=int(progress[i]),completed=bool(task.sequence_completed[i]),
                    stopped_segment=g+1,global_steps=last+1,stage_steps=int(data['stage_step'][last,i])+1,
                    reason='完成全部目标' if task.sequence_completed[i] else '物理失败' if physical else '阶段超时',
                    root_cm=float(data['root_error'][last,i])*100,angle_rad=float(data['angle_error'][last,i]),parts=parts))
            unchanged=all(torch.equal(v,saved_actor[k]) for k,v in actor.state_dict().items());assert unchanged and reset_calls==0
            result=dict(checkpoint=str(path.resolve()),policy_updates=updates,attempts=task.N,
                        contact_only=args.contact_only,reset_history_on_handoff=args.reset_history_on_handoff,
                        completed=int(task.sequence_completed.sum()),success_rate=float(task.sequence_completed.float().mean()),
                        mean_completed_segments=float(progress.float().mean()),max_completed_segments=int(progress.max()),
                        progress_histogram=torch.bincount(progress,minlength=task.count+1).tolist(),segments=segments,examples=example_reports,
                        state_continuity_max_error=max_continuity,student_resets_during_sequence=reset_calls,actor_unchanged=unchanged,
                        root_tolerance_m=args.root_tolerance,contact_tolerances_m=radii,grace_steps=args.grace_steps,
                        steps=step+1,output=str(output.resolve()))
            torch.save(dict(data={k:v.cpu() for k,v in data.items()},entries={k:v.cpu() for k,v in entries.items()},
                            metadata=rec['metadata'],report=result),output/'rollout.pt')
            (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            results.append(result);status('variant_complete',policy_updates=updates)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            del data,task,previous,entry_records,entries,row,saved_actor
        status('complete')
    (args.output/'report.json').write_text(json.dumps(dict(config=config,results=results),default=str,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
