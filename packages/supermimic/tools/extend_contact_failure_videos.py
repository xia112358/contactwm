"""Restore saved failure endpoints and histories; extend frozen latest policy ten frames."""
from supermimic.paths import runtime_path, runtime_root
import sys,os,json,runpy,subprocess,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch,numpy as np
from supermimic.contact_task import ContactTask
original_step=ContactTask.step
source=runtime_path('tmp/contact-tracker-20261027-v1/dmo-continuous-aligned-band-100-resume-v1')
oldweb=runtime_path('results/mixed_comparison_web/aligned-failure-videos-v1')
web=runtime_path('results/mixed_comparison_web/aligned-failure-extension-v1')

def extend(task):
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import mat_to_quat,quat_to_mat,rot_angle_sq
    from supermimic.contact_control import observation
    cfg=json.loads((source/'config.json').read_text());report=json.loads((oldweb/'report.json').read_text());examples=report['examples'];env=task.env;device=env.device
    buf=torch.load(source/'replay.pt',map_location='cpu',weights_only=False,mmap=True);ix=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T'];bd=buf['data'];ss=(bd['ep_step'][ix]==0).all(-1).nonzero().flatten().tolist();a,z=[(a,z) for a,z in zip(ss,ss[1:]+[len(ix)]) if bd['done'][ix[a:z]].any(0).all()][-1];rows=ix[a:z]
    # Fill unused student worlds with repeated examples; export each example once.
    n=len(examples);choice=[examples[i%n] for i in range(task.N)];ws=torch.tensor([x['world'] for x in choice]);ts=torch.tensor([x['global_steps']-1 for x in choice]);g=torch.tensor([x['segment']-1 for x in choice],device=device);caps=torch.tensor([x['cap'] for x in choice],device=device)
    state=bd['s1'][rows[ts],ws].to(device);xi=bd['next_xi'][rows[ts],ws].to(device);prev=bd['a'][rows[ts],ws].to(device)
    ids=task.ids;S=unflatten_state(state);quat=mat_to_quat(S[1]);env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((S[0]+env.scene.env_origins[ids],quat[:,1:],quat[:,:1]),-1),env_ids=ids);env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((S[2],S[3]),-1),env_ids=ids);env.robot.write_joint_position_to_sim_index(position=S[4],env_ids=ids);env.robot.write_joint_velocity_to_sim_index(velocity=S[5],env_ids=ids);env.robot.set_joint_position_target_index(target=task.default_q+env.alpha*prev+xi[:,1:30],env_ids=ids)
    env.a_prev[ids]=prev;env.xi[ids]=xi;env.ep_step[ids]=bd['ep_step'][rows[ts],ws].to(device=device,dtype=env.ep_step.dtype)+1;env.refresh()
    assert float((torch.cat((env.state()[0][ids],),-1)-S[0]).abs().max())<1e-5
    hi=(ts[:,None]+torch.arange(-task.actor.hist+1,1)).clamp_min(0);hr=rows[hi];hw=ws[:,None]
    hs=bd['s'][hr,hw].to(device);ha=bd['a_prev'][hr,hw].to(device);hx=bd['xi'][hr,hw].to(device)
    task.history=observation(unflatten_state(hs.flatten(0,1)),ha.flatten(0,1),hx.flatten(0,1),task.default_q,task.context).reshape(task.N,task.actor.hist,-1)
    task.segment=g;task.starts=torch.as_tensor(task.clock.bounds,device=device)[g];task.ends=task.clock.event_ends[g];task.budgets=caps+10;task.sequence_last=g.clone();task.stage_step=caps.clone();task.sequence_live.fill_(True);task.sequence_completed.zero_();task.sequence_progress.zero_();task.contact_tolerance=task.contact_tolerances_by_segment[g,None]
    task.valid_contacts.zero_()
    for i,x in enumerate(choice):task.valid_contacts[i]=bd['contact_goal'][rows[x['entry_frame']:x['global_steps']],x['world']].any(0).to(device)
    task.previous_contact=(bd['active'][rows[ts],ws]&bd['known'][rows[ts],ws]).to(device);task.reader.set_targets(task.demo['point'][task.ends-1],ids)
    records=[]
    with torch.no_grad():
        for k in range(10):
            row=original_step(task);records.append({key:row[key][:n].cpu().clone() for key in ['s1','point','active','known','stage_success','root_error','angle_error','done']})
    web.mkdir(parents=True,exist_ok=True);renders=[];cards=[];results=[]
    from concurrent.futures import ThreadPoolExecutor
    pool=ThreadPoolExecutor(max_workers=3)
    for i,x in enumerate(examples):
        stop=next((k+1 for k,v in enumerate(records) if v['done'][i]),10);achieved=next((k+1 for k,v in enumerate(records) if v['stage_success'][i]),None);stem=Path(x['video']).stem
        old=np.load(oldweb/(stem+'.npz'));arr={k:old[k] for k in old.files};s1=torch.stack([v['s1'][i] for v in records[:stop]])
        def append(key,tail):arr[key]=np.concatenate((arr[key],tail),axis=0)
        append('root_p',s1[:,:3].numpy()[:,None]);append('root_quat',s1[:,3:7].numpy()[:,None]);append('q',s1[:,13:42].numpy()[:,None]);append('alive',np.ones((stop,1),bool));oldlen=len(arr['f']);append('f',np.arange(oldlen,oldlen+stop)[:,None])
        for key in ['ref_root_p','ref_root_R','ref_q','goal_points_by_frame','goal_root_by_frame']:append(key,np.repeat(arr[key][-1:],stop,axis=0))
        append('measured_contact_points',torch.stack([v['point'][i] for v in records[:stop]]).numpy());end=int(task.clock.event_ends[x['segment']-1])-1;new=(task.clock.target_masks[end]&~task.clock.keep_masks[end]).cpu();append('measured_contact_active',(torch.stack([v['active'][i]&v['known'][i] for v in records[:stop]])&new).numpy())
        notes=arr['frame_annotations'].tolist();notes[-1]+=' | ORIGINAL DEADLINE'
        for k,v in enumerate(records[:stop]):notes.append(f'AFTER DEADLINE +{k+1} | stage {x["segment"]} | '+('TARGET ACHIEVED' if v['stage_success'][i] else 'not achieved')+f'\nRoot {float(v["root_error"][i])*100:.2f}cm | angle {float(v["angle_error"][i]):.3f}rad')
        arr['frame_annotations']=np.array(notes);arr['clip_len']=np.array([len(arr['f'])]);npz=web/(stem+'.npz');np.savez_compressed(npz,**arr)
        cmd=[sys.executable,str(ROOT/'tools/render.py'),str(npz),str(web/(stem+'.mp4')),'--task_manifest',cfg['task_manifest'],'--seconds',str(len(arr['f'])*.02+1),'--hide_reference','--fps','50','--width','1280','--height','720','--camera_distance','3.0','--title',f'Segment {x["segment"]} | extended +10 | continuation policy 140'];renders.append(pool.submit(subprocess.run,cmd,cwd=runtime_root(),check=True))
        result=dict(x,extra_to_success=achieved,extra_recorded=stop,original_policy=139,continuation_policy=140,restored_physical_state=True,restored_action_history=True,solver_cache_restored=False);results.append(result)
        label='示范入口' if x['entry']=='direct' else '衔接入口';text=f'晚 {achieved} 帧（{achieved*.02:.2f}秒）达成' if achieved else '额外10帧内仍未达成';cut=x['global_steps']*.02
        cards.append(f'<section><h2>第{x["segment"]}段 · {label}：{text}</h2><video controls playsinline preload="metadata" src="{stem}.mp4"></video><button onclick="jump(this,{max(0,cut-.15)})">原截止前几帧</button><button onclick="jump(this,{cut})">原截止帧</button><p>原段上限 {x["cap"]} 帧；{text}。原失败原因：{html.escape(x["reason"])}</p></section>')
    (web/'report.json').write_text(json.dumps(dict(scope='Restore recorded failure state/velocity/action history/accumulated contacts. Continue with latest policy140; original recorded prefix uses policy139. Solver cache is not saved, so restored continuations are diagnostic, not bitwise-exact counterfactuals.',examples=results),ensure_ascii=False,indent=2))
    for future in renders:future.result()
    pool.shutdown()
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>失败动作额外10帧</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1200px;margin:auto;padding:20px}section{background:#203047;padding:18px;margin:20px 0;border-radius:12px}video{width:100%}button,select{font:inherit;padding:5px;margin:6px}a{color:#9cf}</style><h1>失败动作：原截止后继续执行10帧</h1><p>保留原失败判定。原录像来自策略139；延长部分恢复原截止的物理状态、速度、历史与累计接触，使用最终策略140。不恢复求解器缓存，因此这是恢复状态后的诊断，不是逐位完全复现。视频标明 ORIGINAL DEADLINE 和 AFTER DEADLINE +帧数。</p><p><select onchange="document.querySelectorAll(\'video\').forEach(v=>v.playbackRate=+this.value)"><option>.25</option><option>.5</option><option>1</option></select> <a href="report.json">逐条结果与来源</a></p>'+''.join(cards)+'<script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.25);function jump(b,t){const v=b.parentNode.querySelector("video");v.pause();v.currentTime=Math.min(t,v.duration-.02)}</script></html>'
    (web/'index.html').write_text(page);print('EXTENSION COMPLETE',json.dumps([(x['segment'],x['entry'],x['extra_to_success']) for x in results]),flush=True)
    raise SystemExit(0)

def intercept(task):return extend(task)
ContactTask.step=intercept
runpy.run_path(str(ROOT/'tools/check_contact_sequence.py'),run_name='__main__')
