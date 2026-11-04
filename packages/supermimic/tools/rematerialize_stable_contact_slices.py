"""Collect native evidence or diagnose tracking restarts.

Dataset admission and topology: rebuild_contact_topology_library.py.
Tracking restart outcomes never change source dataset membership.
"""
import argparse,hashlib,json,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('capture','validate'),required=True)
    p.add_argument('--source_library',type=Path,required=True)
    p.add_argument('--task_manifest',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--headless',action='store_true')
    add_launcher_args(p);a=p.parse_args()
    if a.headless:a.visualizer=None
    import torch
    torch.set_num_threads(4)
    from supermimic.contact_stability import CONTRACT,NativeContactStability
    from supermimic.contact_phase import VERSION,reference_release_starts,LandingPhaseTracker
    source=torch.load(a.source_library/'rollout.pt',map_location='cpu',weights_only=False)
    from supermimic.lowstep import install,checkpoint_metadata,validate_checkpoint
    spec=install(a.task_manifest,contact_model='force6',collision_profile='surface_v2',contact_response='newton_contacts_v2')
    from supermimic.env import TrackingSim,make_sim
    from supermimic.kinematics import G1Kinematics
    from supermimic.motion import MotionLib,CONTROL_DT
    from supermimic.mdp import MDP,OBS_DIM,unflatten_state,flatten_state
    from supermimic.robot import default_joint_pos_list
    from supermimic.rotations import mat_to_quat,rot_angle_sq
    from supermimic.contact_native import NativeContacts
    from supermimic.contact_library import history_indices
    import supermimic.networks as networks
    torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True
    device='cuda:0';ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    validate_checkpoint(ck,checkpoint_metadata(spec));validate_checkpoint(source['metadata'],checkpoint_metadata(spec))
    if sha(a.checkpoint)!=source['metadata']['source_checkpoint_sha256']:raise ValueError('Use the original demonstration tracking checkpoint')
    networks.AMP_DTYPE=torch.bfloat16 if ck['args'].get('amp')=='bf16' else None
    hist,lookahead=ck['args']['hist'],ck['args']['goal_frames'];policy=networks.Policy(OBS_DIM,73*lookahead,hist=hist).to(device).eval().requires_grad_(False);policy.load_state_dict(ck['pi'])
    kin=G1Kinematics(device);motion=MotionLib.from_dir(spec['motion_dir'],kin,device);default=torch.tensor(default_joint_pos_list(),device=device);mdp=MDP(kin,motion,default)
    a.num_envs=len(spec['tasks']) if a.mode=='capture' else None
    if a.mode=='validate':
        library=a.output/'library';recording=torch.load(library/'rollout.pt',map_location=device,weights_only=False);slices=torch.load(library/'cases.pt',map_location=device,weights_only=False)
        counts=[max(1,int((slices['motion_ids']==i).sum())) for i in range(len(spec['tasks']))];a.num_envs=sum(counts)
    cfg=make_sim(device,a)
    with launch_simulation(cfg,a),torch.no_grad():
        env=TrackingSim(cfg,a.num_envs,motion,mdp,eval_mode=True,**({} if a.mode=='capture' else dict(task_world_counts=counts)))
        all_ids=torch.arange(env.N,device=device);env.reset(all_ids,motion.clip_start[env.task_ids]);env.refresh()
        reader=NativeContacts(env,direct=True,primary_faces=True)
        if a.mode=='capture':
            dest=a.output/f'capture-seed{a.seed}';dest.mkdir(parents=True,exist_ok=False);reader.export_collision_model(dest/'solver-collision.mjb')
            (dest/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
            support=NativeContactStability(CONTROL_DT);history=None;pending=set(range(len(spec['tasks'])));rows={i:[] for i in pending};raw=[];final={}
            worlds={i:int((env.task_ids==i).nonzero().flatten()[0]) for i in pending}
            from supermimic.contact_substep_capture import SubstepCapture
            with SubstepCapture(env,dest/'substeps') as substeps:
                for t in range(int(motion.clip_len.max())-1):
                    state=env.state();obs=mdp.obs(state,env.a_prev,env.xi,env.f);history=obs[:,None].expand(-1,hist,-1).clone() if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                    out=env.step(policy(history,motion.goal(env.f,lookahead)));substeps.flush(t,[worlds[i] for i in pending]);evidence=reader.read_device();evidence.update(substeps.monitor.summary());evidence.update(support.update(reader.last_view,reader.last_part_poses,env.ep_step))
                    raw.append(dict(control_frame=t,evidence=reader.last_evidence.export(),eligible=reader.last_primary_selection['eligible'].cpu(),groups=reader.last_view.groups.cpu(),view_rows=reader.last_view.rows.cpu(),part_position=reader.last_part_poses[0].cpu(),part_rotation=reader.last_part_poses[1].cpu()))
                    for i in list(pending):
                        w=worlds[i];row={k:out[k][w].cpu().clone() for k in ('s','s1','a','a_prev','xi','f','term','done','ep_step','r')};row.update({k:v[w].cpu().clone() for k,v in evidence.items()});rows[i].append(row)
                        if bool(row['done']) or len(rows[i])>=int(motion.clip_len[i])-1:final[i]=env.xi[w].cpu().clone();pending.remove(i)
                    if t%100==0:print(json.dumps(dict(stage='capture',seed=a.seed,frame=t,pending=len(pending))),flush=True)
                    if not pending:break
                    env.f=torch.minimum(env.f,motion.clip_end[env.task_ids]-1)
            reports=[]
            for i,trace in rows.items():
                data={k:torch.stack([x[k] for x in trace]) for k in trace[0]};data['next_xi']=torch.cat((data['xi'][1:],final[i][None]),0)
                report=dict(motion=i,name=spec['tasks'][i]['name'],frames=len(trace),successful_rollout=len(trace)==int(motion.clip_len[i])-1 and not bool(data['term'].any()))
                torch.save(dict(data=data,report=report),dest/f'motion-{i:03d}.pt');reports.append(report)
            torch.save(raw,dest/'native-rows.pt');(dest/'report.json').write_text(json.dumps(dict(motions=reports,substep_support_contract=substeps.monitor.contract,stable_support_contract=CONTRACT,phase_version=VERSION,checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),task_manifest_sha256=sha(a.task_manifest),policy_updated=False),indent=2));return
        validate(a,env,reader,policy,motion,mdp,recording,slices,hist,lookahead,NativeContactStability,CONTROL_DT,reference_release_starts,LandingPhaseTracker,unflatten_state,flatten_state,mat_to_quat,rot_angle_sq)


def validate(a,env,reader,policy,motion,mdp,recording,slices,hist,lookahead,Support,dt,releases,Tracker,unflatten,flatten,quat,angle_fn):
    import torch
    from types import SimpleNamespace
    from supermimic.contact_library import history_indices
    data=recording['data'];N=len(slices['starts']);ids=torch.empty(N,dtype=torch.long,device=env.device)
    for i in slices['motion_ids'].unique().tolist():
        ci=(slices['motion_ids']==i).nonzero().flatten();ids[ci]=(env.task_ids==i).nonzero().flatten()[:len(ci)]
    starts,goals=slices['starts'],slices['goals'];env.reset(ids,data['f'][starts].long())
    pos,R,v,w,q,qd=unflatten(data['s'][starts]);qr=quat(R);env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos+env.scene.env_origins[ids],qr[:,1:],qr[:,:1]),-1),env_ids=ids);env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v,w),-1),env_ids=ids);env.robot.write_joint_position_to_sim_index(position=q,env_ids=ids);env.robot.write_joint_velocity_to_sim_index(velocity=qd,env_ids=ids);env.robot.set_joint_position_target_index(target=q,env_ids=ids)
    env.a_prev[ids]=data['a_prev'][starts];env.xi[ids]=data['xi'][starts];env.ep_step[ids]=0;env.refresh()
    restored=flatten(env.state())[ids];reset_error=(restored[:,7:]-data['s'][starts,7:]).abs().max();reset_root=(restored[:,:3]-data['s'][starts,:3]).norm(dim=-1).max()
    if reset_error>1e-3 or reset_root>1e-4:raise ValueError('Native restart state restoration failed')
    clock=SimpleNamespace(segment_starts=starts,segment_stops=slices['stops'],event_ends=goals+1,data=data,target_masks=torch.zeros_like(data['active']),keep_masks=torch.zeros_like(data['active']))
    clock.target_masks[goals]=slices['target_masks'];clock.keep_masks[goals]=slices['keep_masks'];new=slices['target_masks']&~slices['keep_masks'];tracker=Tracker(releases(clock,2),new,2);support=Support(dt)
    obs=mdp.obs(env.state(),env.a_prev,env.xi,env.f);h=obs[:,None].expand(-1,hist,-1).clone();hi=slices['history_indices'];h[ids]=mdp.obs(unflatten(data['s'][hi].flatten(0,1)),data['a_prev'][hi].flatten(0,1),data['xi'][hi].flatten(0,1),data['f'][hi].flatten()).reshape(N,hist,-1)
    reader.set_targets(data['point'][goals],ids);budget=slices['stops']-starts+10;finished=torch.zeros(N,dtype=torch.bool,device=env.device);passed=finished.clone();latched=torch.zeros(N,6,3,device=env.device);report=[None]*N;reference=unflatten(data['s1'][goals]);trace=[]
    for t in range(int(budget.max())):
        state=env.state()
        if t==0:env.xi[ids]=data['xi'][starts]
        obs=mdp.obs(state,env.a_prev,env.xi,env.f);h=torch.cat((h[:,1:],obs[:,None]),1);out=env.step(policy(h,motion.goal(env.f,lookahead)));evidence=reader.read_device();features=support.update(reader.last_view,reader.last_part_poses,env.ep_step);e={k:v[ids] for k,v in dict(evidence,**features).items()};hit=tracker.update(e['active'],e['known'],e['contact_stable']);latched=torch.where(hit[...,None],e['point'],latched)
        refready=((~new)|tracker.confirmed).all(-1);distance=(latched-data['point'][goals]).norm(dim=-1);position=((~new)|(tracker.confirmed&(distance<=.05))).all(-1);s=unflatten(out['s1'][ids]);root=(s[0]-reference[0]).norm(dim=-1);angle=angle_fn(s[1],reference[1]).sqrt();failed=(~torch.isfinite(out['s1'][ids]).all(-1))|(s[0][:,2]<.15)
        kept=((~slices['keep_masks'])|(e['active']&e['known'])).all(-1);ok=refready&position&(root<.25)&(angle<.4)&~failed;ok=torch.where(new.any(-1),ok,kept&(root<.25)&(angle<.4)&~failed)
        done=~finished&((new.any(-1)&refready)|ok|failed|(t+1>=budget));passed|=done&ok
        for i in done.nonzero().flatten().tolist():report[i]=dict(case=i+1,passed=bool(ok[i]),stop=t+1,confirmed=tracker.confirmed[i].cpu().tolist(),root_error_cm=float(root[i]*100),angle_error_rad=float(angle[i]),errors_cm=[float(distance[i,p]*100) if tracker.confirmed[i,p] else None for p in range(6)],failed=bool(failed[i]))
        trace.append(dict(active=e['active'].cpu(),known=e['known'].cpu(),point=e['point'].cpu(),contact_first=hit.cpu(),contact_stable=e['contact_stable'].cpu(),contact_material_speed=e['contact_material_speed'].cpu(),contact_load=e['contact_load'].cpu(),contact_row_stable=e['contact_row_stable'].cpu(),contact_load_stable=e['contact_load_stable'].cpu(),s1=out['s1'][ids].cpu(),done=done.cpu()))
        finished|=done
        if finished.all():break
        env.f=torch.minimum(env.f,motion.clip_end[env.task_ids]-1)
    root=a.output/'tracking-restart-diagnostic';root.mkdir(exist_ok=False);torch.save(trace,root/'replay.pt');result=dict(total=N,passed=int(passed.sum()),failed_restart=int((~passed).sum()),diagnostic_only=True,admission_changed=False,max_reset_state_error=float(reset_error),max_reset_root_error=float(reset_root),policy_updated=False,cases=report);(root/'report.json').write_text(json.dumps(result,indent=2))
    diagnostic=dict(case_ids=slices['case_ids'].cpu(),restart_passed=passed.cpu(),diagnostic_only=True,admission_changed=False)
    torch.save(diagnostic,root/'cases_restart_diagnostic.pt')
    (root/'status.json').write_text(json.dumps(dict(stage='diagnostic_complete',training_started=False,admission_changed=False,cases=N,restart_passed=int(passed.sum()),restart_failed=int((~passed).sum())),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)



if __name__=='__main__':main()
