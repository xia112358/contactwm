"""Test student-generated standing history before executing segment one."""
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
    from supermimic.rotations import rot_angle_sq,yaw_mat,mat_to_6d
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
                execution='Same student stands for 0/5/10/20 frames then receives segment-one goal; retain versus repeat history, no physical reset at switch; full 92 target steps.',
                training=False,wm_used_for_execution=False,root_tolerance_m=args.root_tolerance,
                max_steps=112)
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
        variants=[('direct',0,False),('stand5_keep',5,False),('stand5_repeat',5,True),
                  ('stand10_keep',10,False),('stand10_repeat',10,True),
                  ('stand20_keep',20,False),('stand20_repeat',20,True)]
        for path in args.checkpoints:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            actor.load_state_dict(checkpoint['actor']);del checkpoint
            saved_weights={k:v.clone() for k,v in actor.state_dict().items()}
            task=ContactTask(env,actor,teacher,motion,mdp,clock,demo,context,default,reader,hist,goals,
                             args.grace_steps,0,args.root_tolerance,'accumulated_target',radii,sequential=True)
            group=torch.arange(task.N,device=device)%len(variants)
            warm=torch.tensor([v[1] for v in variants],device=device)[group]
            repeat=torch.tensor([v[2] for v in variants],device=device)[group]
            task.stage_step=-warm.clone()
            initial=tuple(x[task.ids].clone() for x in env.state())
            target_points=torch.zeros((task.N,6,3),device=device)
            stand_point_frames=[]
            for part in range(2):
                valid=(demo['active'][:10,part]&demo['known'][:10,part]).nonzero().flatten()
                assert len(valid), 'Standing foot needs an observed contact target'
                frame=int(valid[0]);stand_point_frames.append(frame)
                target_points[:,part]=demo['point'][frame,part]
            stand_mask=torch.zeros((task.N,6),device=device);stand_mask[:,:2]=1
            switch_state=torch.zeros_like(demo['s'][0]).expand(task.N,-1).clone()
            switch_history_rms=torch.zeros(task.N,device=device)
            switch_action_rms=torch.zeros(task.N,device=device)
            step=0
            def forward(history,condition):
                now=step==warm
                if now.any():
                    current=tuple(v[task.ids] for v in env.state())
                    from supermimic.mdp import flatten_state
                    switch_state[now]=flatten_state(current)[now]
                    repeated=history[:,-1:, :].expand_as(history)
                    switch_history_rms[now]=(history[now]-repeated[now]).square().mean((1,2)).sqrt()
                    switch_action_rms[now]=(actor(history[now],condition[now])-actor(repeated[now],condition[now])).square().mean(1).sqrt()
                    change=now&repeat
                    history[change]=repeated[change].clone()
                    task.valid_contacts[now]=False
                    task.recontact[now]=False
                    task.onset_point[now]=0
                    task.onset_face[now]=-1
                standing=step<warm
                if standing.any():
                    state=tuple(v[task.ids] for v in env.state())
                    Ht=yaw_mat(state[1]).transpose(-1,-2)
                    relative=(Ht[:,None]@(target_points-state[0][:,None])[...,None]).squeeze(-1)
                    root=(Ht@(initial[0]-state[0])[...,None]).squeeze(-1)
                    rotation=mat_to_6d(state[1].transpose(-1,-2)@initial[1])
                    stand=torch.cat(((relative*stand_mask[...,None]).flatten(1),stand_mask,root,rotation,stand_mask),-1)
                    condition=condition.clone();condition[standing]=stand[standing]
                return actor(history,condition)
            class RoutedActor:
                hist=actor.hist
                def __call__(self,history,condition):return forward(history,condition)
            task.actor=RoutedActor()
            def finish(done,success,transition):
                enabled=step>=warm
                done&=enabled;success&=enabled
                transition['stage_done']&=enabled
                transition['stage_success']&=enabled
                transition['done']=done.clone()
                task.sequence_progress+=success.long()
                task.sequence_completed|=success
                task.sequence_live&=~done
            task.finish_sequence=finish
            records=[]
            for step in range(112):
                row=task.step()
                row['target_phase']=(step>=warm)&row['sequence_live']
                records.append({k:row[k].clone() for k in ('s','s1','a','stage_step','stage_done','stage_success','sequence_live','target_phase','point','active','known','contact_goal','root_error','angle_error')})
                if not task.sequence_live.any():break
            assert not task.sequence_live.any()
            assert all(torch.equal(v,saved_weights[k]) for k,v in actor.state_dict().items())
            data={k:torch.stack([r[k] for r in records]) for k in records[0]}
            stop=data['stage_done'].long().argmax(0);batch=torch.arange(task.N,device=device)
            end=int(clock.event_ends[0])-1
            new=clock.target_masks[end]&~clock.keep_masks[end]
            dist=(data['point']-demo['point'][end]).norm(dim=-1)
            observed=data['active']&data['known']&data['target_phase'][...,None]
            best=dist.masked_fill(~observed,float('inf')).amin(0)
            rows=[]
            for g,(name,nwarm,repeated) in enumerate(variants):
                mask=group==g;ix=batch[mask];last=stop[mask]
                rows.append(dict(variant=name,warm_frames=nwarm,repeat_history_at_switch=repeated,n=int(mask.sum()),
                    passed=int(task.sequence_completed[mask].sum()),
                    mean_target_steps=float((last+1-warm[mask]).float().mean()),
                    root_cm=float(data['root_error'][last,ix].mean()*100),
                    angle_rad=float(data['angle_error'][last,ix].mean()),
                    best_contact_cm=[float(v) for v in best[mask][:,new].median(0).values*100],
                    switch_root_displacement_cm=float((switch_state[mask,:3]-demo['s'][0,:3]).norm(dim=-1).mean()*100),
                    switch_root_height_m=float(switch_state[mask,2].mean()),
                    switch_root_speed_m_s=float(switch_state[mask,7:10].norm(dim=-1).mean()),
                    switch_history_rms=float(switch_history_rms[mask].mean()),
                    same_state_history_action_difference_rms=float(switch_action_rms[mask].mean())))
            result=dict(checkpoint=str(path),actor_unchanged=True,stand_point_frames=stand_point_frames,variants=rows)
            results.append(result)
            out=args.output/path.parent.name;out.mkdir(exist_ok=True)
            torch.save(dict(data={k:v.cpu() for k,v in data.items()},group=group.cpu(),warm=warm.cpu(),switch_state=switch_state.cpu()),out/'rollout.pt')
            status('checkpoint_complete',checkpoint=str(path))
            print(json.dumps(result),flush=True)
        (args.output/'report.json').write_text(json.dumps(dict(config=config,results=results),default=str,indent=2))
        status('complete')

if __name__=='__main__':main()
