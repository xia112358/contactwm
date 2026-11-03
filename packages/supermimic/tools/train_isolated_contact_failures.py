"""Independent DMO refinement of recorded no-contact failure entries."""
from supermimic.paths import runtime_path
import sys,json,runpy,copy,signal,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch
from supermimic.contact_task import ContactTask
native_step=ContactTask.step
base=runtime_path('tmp/contact-tracker-20261027-v1');source=base/'dmo-continuous-aligned-band-100-resume-v1';out=base/'isolated-region-normal4-v1';web=runtime_path('results/mixed_comparison_web/isolated-contact-region')
stopping=False
def request_stop(*a):
 global stopping
 stopping=True

def run(task):
    from supermimic.contact_control import ContactGeometry,observation
    from supermimic.contact_dmo import replay_loss
    from supermimic.contact_surface import SurfaceRegionObjective
    from supermimic.networks import WorldModel
    from supermimic.mdp import unflatten_state,normalize_xi
    from supermimic.rotations import mat_to_quat,quat_to_mat
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    cfg=json.loads((source/'config.json').read_text());audit=json.loads((source/'terminal-contact-source-audit.json').read_text());old=json.loads((runtime_path('results/mixed_comparison_web/aligned-failure-videos-v1/report.json')).read_text());selected=[x for x in old['examples'] if any(y['source']=='FK_material_point' for a in audit['examples'] if a['segment']==x['segment'] and a['entry']==x['entry'] for y in a['parts'])]
    env=task.env;env.reset=type(env).reset.__get__(env,type(env));dev=env.device;K=len(selected);N=task.N;group=torch.arange(N,device=dev)%K;slots=[(group==k).nonzero().flatten() for k in range(K)]
    models=torch.nn.ModuleList([copy.deepcopy(task.actor).requires_grad_(True) for _ in selected]);optim=[torch.optim.Adam(m.parameters(),lr=3e-6,betas=(.9,.95)) for m in models]
    class Router(torch.nn.Module):
        hist=task.actor.hist
        def forward(self,h,c):
            a=h.new_zeros(len(h),29)
            for k,ids in enumerate(slots):a[ids]=models[k](h[ids],c[ids])
            return a
    task.actor=Router();ck=torch.load(source/'checkpoint_000140.pt',map_location=dev,weights_only=False);wm=WorldModel().to(dev);wm.load_state_dict(ck['wm']);wm.requires_grad_(False);geom=ContactGeometry(task.demo,dev);geom.load_state_dict(ck['geometry']);surface=SurfaceRegionObjective(base/'initialization-fast-v3-solver.mjb',geom,dev);prior=copy.deepcopy(models[0]).requires_grad_(False);prior.load_state_dict(torch.load(cfg['prior_checkpoint'],map_location=dev,weights_only=False)['actor']);del ck
    if (source/'isolated-entry-pack.pt').exists():
        packed=torch.load(source/'isolated-entry-pack.pt',map_location='cpu',weights_only=False)
        lookup={(x['segment'],x['entry'],x['world']):i for i,x in enumerate(packed['examples'])}
        choice=[selected[i%K] for i in range(N)];g=torch.tensor([x['segment']-1 for x in choice],device=dev);caps=torch.tensor([x['cap'] for x in choice],device=dev);begins=torch.tensor(task.clock.bounds,device=dev)[g]
        pick=torch.tensor([lookup[x['segment'],x['entry'],x['world']] for x in choice])
        initial={key:value[pick].to(dev) for key,value in packed['initial'].items()}
        def history_from(key):
            h=packed[key];hs=h['s'][pick].to(dev);ha=h['a_prev'][pick].to(dev);hx=h['xi'][pick].to(dev)
            return observation(unflatten_state(hs.flatten(0,1)),ha.flatten(0,1),hx.flatten(0,1),task.default_q,task.context).reshape(N,models[0].hist,-1)
        entry_history=history_from('history');previous_history=history_from('previous_history')
    else:
        buf=torch.load(source/'replay.pt',map_location='cpu',weights_only=False,mmap=True);bd=buf['data'];ix=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T'];ss=(bd['ep_step'][ix]==0).all(-1).nonzero().flatten().tolist();start,finish=[(a,z) for a,z in zip(ss,ss[1:]+[len(ix)]) if bd['done'][ix[a:z]].any(0).all()][-1];rows=ix[start:finish]
        choice=[selected[i%K] for i in range(N)];w=torch.tensor([x['world'] for x in choice]);entry=torch.tensor([x['entry_frame'] for x in choice]);g=torch.tensor([x['segment']-1 for x in choice],device=dev);caps=torch.tensor([x['cap'] for x in choice],device=dev);begins=torch.tensor(task.clock.bounds,device=dev)[g]
        initial={key:bd[key][rows[entry],w].to(dev) for key in ['s','xi','a_prev','ep_step']}
        hi=(entry[:,None]+torch.arange(-models[0].hist+1,1)).clamp_min(0);hr=rows[hi];hw=w[:,None]
        hs=bd['s'][hr,hw].to(dev);ha=bd['a_prev'][hr,hw].to(dev);hx=bd['xi'][hr,hw].to(dev)
        entry_history=observation(unflatten_state(hs.flatten(0,1)),ha.flatten(0,1),hx.flatten(0,1),task.default_q,task.context).reshape(N,models[0].hist,-1)
        # Previous history ends one frame before entry, because task.step appends entry.
        hp=(entry[:,None]+torch.arange(-models[0].hist,0)).clamp_min(0);pr=rows[hp]
        ps=bd['s'][pr,hw].to(dev);pa=bd['a_prev'][pr,hw].to(dev);px=bd['xi'][pr,hw].to(dev)
        previous_history=observation(unflatten_state(ps.flatten(0,1)),pa.flatten(0,1),px.flatten(0,1),task.default_q,task.context).reshape(N,models[0].hist,-1)
    # Nearest bounded static box face (plus floor); calculated once, not during collection.
    spec=json.loads((runtime_path(cfg['task_manifest'])).read_text());boxes=spec['tasks'][int(task.demo['xi'][0,33])]['boxes'];normal_rows=[];normal_dist=[]
    for x in selected:
        target=task.demo['point'][task.clock.event_ends[x['segment']-1]-1];ns=[];ds=[]
        for target_part in target:
            candidates=[(abs(float(target_part[2])),target_part.new_tensor([0,0,1]))]
            for box in boxes:
                center=target_part.new_tensor(box[:3]);R=quat_to_mat(target_part.new_tensor(box[3:7]));half=target_part.new_tensor(box[7:10])/2;local=R.T@(target_part-center)
                for axis in range(3):
                    for sign in [-1,1]:
                        q=local.clamp(-half,half);q[axis]=sign*half[axis];distance=float((local-q).norm());normal=R[:,axis]*sign;candidates.append((distance,normal))
            distance,normal=min(candidates,key=lambda y:y[0]);ns.append(normal);ds.append(distance)
        normal_rows.append(torch.stack(ns));normal_dist.append(ds)
    normals=torch.stack(normal_rows);required=task.clock.target_masks[task.clock.event_ends[g]-1]&~task.clock.keep_masks[task.clock.event_ends[g]-1]
    for k,x in enumerate(selected):
        for j in required[slots[k][0]].nonzero().flatten().tolist():assert normal_dist[k][j]<.03,('target far from surface',x['segment'],normal_dist[k][j])
    @torch.no_grad()
    def reset():
        ids=task.ids;env.reset(ids,task.demo['f'][begins].long());S=unflatten_state(initial['s']);quat=mat_to_quat(S[1]);env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((S[0]+env.scene.env_origins[ids],quat[:,1:],quat[:,:1]),-1),env_ids=ids);env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2],S[3]),-1),env_ids=ids);env.robot.write_joint_position_to_sim_index(position=S[4],env_ids=ids);env.robot.write_joint_velocity_to_sim_index(velocity=S[5],env_ids=ids);env.robot.set_joint_position_target_index(target=task.default_q+initial['xi'][:,1:30]+env.alpha*initial['a_prev'],env_ids=ids);env.a_prev[ids]=initial['a_prev'];env.xi[ids]=initial['xi'];env.ep_step[ids]=initial['ep_step'].to(env.ep_step.dtype);env.refresh()
        task.segment=g.clone();task.starts=begins.clone();task.ends=task.clock.event_ends[g];task.budgets=caps.clone();task.sequence_last=g.clone();task.stage_step.zero_();task.sequence_live.fill_(True);task.sequence_completed.zero_();task.sequence_progress.zero_();task.valid_contacts.zero_();task.recontact.zero_();task.onset_point.zero_();task.onset_face.fill_(-1);task.contact_tolerance=task.contact_tolerances_by_segment[g,None];task.history=previous_history.clone();task.previous_contact.zero_();task.reader.set_targets(task.demo['point'][task.ends-1],ids)
    out.mkdir(exist_ok=False,parents=True);web.mkdir(exist_ok=True,parents=True);meta=dict(source=str(source),source_policy=140,iterations=50,worlds=env.N,student_worlds=N,learning_rate=3e-6,normal_weight=4,tangent_weight=1,wm_frozen=True,independent_models=True,contact_position_loss='surface_region',surface_objective_contract=surface.contract,collision_catalog=surface.catalog,examples=selected,normals=normals.cpu().tolist(),normal_surface_distances=normal_dist);(out/'config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    all_metrics=[];started=time.perf_counter()
    def status(phase,iteration,**extra):
        d=dict(phase=phase,iteration=iteration,iterations=50,elapsed_s=time.perf_counter()-started,metrics=all_metrics,**extra)
        for folder in [out,web]:
            t=folder/'status.tmp';t.write_text(json.dumps(d,ensure_ascii=False));t.replace(folder/'status.json')
    for iteration in range(51):
        if stopping:break
        if iteration%10==0:
            torch.save(dict(policy_updates=iteration,actors=[m.state_dict() for m in models],optimizers=[v.state_dict() for v in optim],config=meta),out/f'checkpoint_{iteration:03d}.pt')
        t0=time.perf_counter();reset();records=[]
        with torch.no_grad():
            for t in range(int(caps.max())):
                row=native_step(task)
                if t==0:actual_entry_history=task.history.clone()
                records.append({key:row[key].clone() for key in ['s','s1','xi','next_xi','a','a_prev','ep_step','goal_index','done','term','point','active','known','contact_goal','stage_success']})
                if not task.sequence_live.any():break
        data={key:torch.stack([r[key] for r in records]) for key in records[0]};collect_s=time.perf_counter()-t0;reports=[]
        for k,x in enumerate(selected):
            ids=slots[k];batch={key:v[:,ids] for key,v in data.items()};mask=required[ids];stop=batch['done'].long().argmax(0);ar=torch.arange(len(ids),device=dev);success=~batch['term'][stop,ar]
            with torch.enable_grad():
                loss,parts=replay_loss(models[k],wm,task.clock,geom,task.context,normalize_xi,task.default_q,batch,prior_actor=prior,prior_weight=.01,completion_mode='accumulated_target',initial_history=actual_entry_history[ids],goal_normals=normals[k:k+1].expand(len(ids),-1,-1),normal_weight=4,tangent_weight=1,contact_surface=surface)
                if iteration<50:
                    optim[k].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(models[k].parameters(),1.);optim[k].step()
            live=torch.arange(len(batch['done']),device=dev)[:,None]<=stop[None]
            achieved=((~mask)|(batch['contact_goal']&live[...,None]).any(0)).all(-1);observed=batch['known']&batch['active']&live[...,None];distance=(batch['point']-task.demo['point'][task.ends[ids]-1][None]).norm(dim=-1);best=distance.masked_fill(~observed,float('inf')).amin(0).masked_fill(~mask,0).max(-1).values;finite=torch.isfinite(best)
            reports.append(dict(segment=x['segment'],entry=x['entry'],attempts=len(ids),success_rate=float(success.float().mean()),target_contact_rate=float(achieved.float().mean()),loss=float(loss.detach()),normal_loss=parts['contact_normal'],tangent_loss=parts['contact_tangent'],contact_loss=parts['terminal_contact'],root_loss=parts['terminal_body'],best_contact_cm=float(best[finite].mean()*100) if finite.any() else None,budget=x['cap'],action_replay_error=parts['action_replay_max_error']))
        all_metrics.append(dict(iteration=iteration,collect_seconds=collect_s,total_seconds=time.perf_counter()-t0,cases=reports));status('training',iteration);print(json.dumps(all_metrics[-1]),flush=True)
        if iteration%10==0 or iteration==50:
            # One real trajectory for each independently optimized case.
            torch.save(dict(iteration=iteration,data={key:v[:,torch.tensor([int(ids[0]) for ids in slots],device=dev)].cpu() for key,v in data.items()},config=meta),out/f'rollout_{iteration:03d}.pt')
    status('interrupted' if stopping else 'complete',iteration)
    raise SystemExit(0)
ContactTask.step=lambda task:run(task)
runpy.run_path(str(ROOT/'tools/check_contact_sequence.py'),run_name='__main__')
