"""Frozen comparison of original versus stationary-prefix-trimmed segment starts."""
from supermimic.paths import runtime_path, runtime_root
import sys,json,runpy,subprocess,os
from pathlib import Path
import argparse
_inputs_parser=argparse.ArgumentParser(description=__doc__)
_inputs_parser.add_argument('--source-root',type=Path,required=True,help='Existing experiment directory containing the historical run inputs')
_inputs,_forwarded=_inputs_parser.parse_known_args()
sys.argv=[sys.argv[0],*_forwarded]
ROOT=Path(__file__).resolve().parents[1]
import torch,numpy as np
from supermimic.contact_task import ContactTask
native_step=ContactTask.step
base=_inputs.source_root;tracking_mode=bool(os.environ.get('TRIM_TRACKING_TEACHER'));out=base/('stationary-trim-tracking-v1' if tracking_mode else 'stationary-trim-v1');web=runtime_path('results/mixed_comparison_web')/('contact-stationary-trim-tracking' if tracking_mode else 'contact-stationary-trim')
def probe(task):
    from supermimic.contact_control import observation
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import quat_to_mat
    from concurrent.futures import ThreadPoolExecutor
    dev=task.env.device;demo=task.demo;clock=task.clock;N=task.N
    S=unflatten_state(demo['s']);v=S[2].norm(dim=-1);w=S[3].norm(dim=-1);qd=S[5].square().mean(-1).sqrt();quiet=(v<.05)&(w<.15)&(qd<.2);spans=[]
    for g,(a,b) in enumerate(zip(clock.bounds,clock.bounds[1:])):
        flags=quiet[a:b].cpu().tolist();runs=[];begin=None
        for t,flag in enumerate(flags+[False]):
            if flag and begin is None:begin=t
            if not flag and begin is not None:runs.append((begin,t));begin=None
        # Prefix settling may precede a genuine standing wait; require a long stable run.
        early=[(x,y) for x,y in runs if y-x>=10 and y<=(b-a)*.6 and x<=20 and torch.equal(demo['active'][a+x:a+y],demo['active'][a+x:a+x+1].expand(y-x,-1))]
        lead=max(0,max((y for x,y in early),default=0)-2)
        tail=next((b-a-x for x,y in runs if y==b-a and y-x>=10),0);end=b-max(0,tail-2)
        if end-(a+lead)<20:lead=0;end=b
        spans.append(dict(segment=g+1,original_start=a,original_end=b,trimmed_start=a+lead,trimmed_end=end,removed_head=lead,removed_tail=b-end))
    oldcfg=json.loads((base/'dmo-duration-history-fresh-v1/config.json').read_text());rows=[json.loads(x) for x in (base/'dmo-duration-history-fresh-v1/metrics.jsonl').read_text().splitlines()];match=next(x for x in rows if x.get('collected_policy_updates')==40);centers=[x['center'] for x in match['duration_filter']['segments']]
    g=torch.arange(N,device=dev)%task.count;variant=(torch.arange(N,device=dev)//task.count)%2;original=torch.tensor(clock.bounds[:-1],device=dev)[g];head=torch.tensor([x['removed_head'] for x in spans],device=dev)[g]*variant;starts=original+head;caps=torch.tensor(centers,device=dev)[g]+2-head
    task.segment=g.clone();task.starts=starts;task.ends=clock.event_ends[g];task.budgets=caps;task.sequence_last=g.clone();task.stage_step.zero_();task.sequence_live.fill_(True);task.sequence_completed.zero_();task.sequence_progress.zero_();task.contact_tolerance=task.contact_tolerances_by_segment[g,None];task.reader.set_targets(demo['point'][task.ends-1],task.ids)
    task.env.reset=type(task.env).reset.__get__(task.env,type(task.env));task.reset(torch.arange(N,device=dev))
    # Warm starts inherit the observed history preceding their new reset frame.
    hi=(starts[:,None]+torch.arange(-task.actor.hist,0,device=dev)).clamp_min(0);H=unflatten_state(demo['s'][hi].flatten(0,1));hist=observation(H,demo['a_prev'][hi].flatten(0,1),demo['xi'][hi].flatten(0,1),task.default_q,task.context).reshape(N,task.actor.hist,-1);task.history=hist;warm=head>0;task.env.ep_step[task.ids[warm]]=starts[warm].to(task.env.ep_step.dtype)
    if tracking_mode:
        thi=(starts[:,None]+torch.arange(-task.teacher_hist,0,device=dev)).clamp_min(0)
        teacher_history=task.mdp.obs(unflatten_state(demo['s'][thi].flatten(0,1)),demo['a_prev'][thi].flatten(0,1),demo['xi'][thi].flatten(0,1),demo['f'][thi].flatten()).reshape(N,task.teacher_hist,-1)
        contact_hist=task.actor.hist
        class TrackingAdapter(torch.nn.Module):
            hist=contact_hist
            def forward(self,history,condition):
                nonlocal teacher_history
                current=tuple(v[task.ids] for v in task.env.state())
                obs=task.mdp.obs(current,task.env.a_prev[task.ids],task.env.xi[task.ids],task.env.f[task.ids])
                teacher_history=torch.cat((teacher_history[:,1:],obs[:,None]),1)
                return task.teacher(teacher_history,task.motion.goal(task.env.f[task.ids],task.goal_frames))
        task.actor=TrackingAdapter()
    records=[]
    with torch.no_grad():
        for t in range(int(caps.max())):
            r=native_step(task);records.append({k:r[k].clone() for k in ['s','s1','a','point','active','known','stage_success','done','stage_step','root_error','angle_error']})
            if not task.sequence_live.any():break
    data={k:torch.stack([r[k] for r in records]) for k in records[0]};stop=data['done'].long().argmax(0);ar=torch.arange(N,device=dev);success=data['stage_success'][stop,ar];result=[]
    for segment in range(task.count):
        for kind in [0,1]:
            m=(g==segment)&(variant==kind);result.append(dict(segment=segment+1,trimmed=bool(kind),attempts=int(m.sum()),success_rate=float(success[m].float().mean()),mean_frames=float((stop[m]+1).float().mean()),budget=int(caps[m][0]),removed_head=int(head[m][0])))
    out.mkdir(exist_ok=True,parents=True);web.mkdir(exist_ok=True,parents=True);report=dict(policy='tracking_6000' if tracking_mode else 40,scope='Frozen comparison; original contact goals retained. Only first-stage initialization/head budget changes. Last-stage demo tail crop is visualization/data proposal only: its learned deadline is already 17 frames.',thresholds=dict(root_speed=.05,angular_speed=.15,joint_velocity_rms=.2,minimum_quiet_frames=10,context_frames=2),spans=spans,results=result)
    (out/'trim-report.json').write_text(json.dumps(report,indent=2));(web/'report.json').write_text(json.dumps(report,indent=2))
    metadata=torch.load(base/'demo0-roles-v1/rollout.pt',map_location='cpu',weights_only=False)['metadata']
    videos=[]
    # Full demo first/last segment versus proposed removal of standing waits.
    for segment in [1,19]:
        span=spans[segment-1]
        for trimmed in [False,True]:
            a=span['trimmed_start'] if trimmed else span['original_start'];b=span['trimmed_end'] if trimmed else span['original_end'];trace=torch.cat((demo['s'][a:a+1],demo['s1'][a:b]));videos.append(dict(stem=f'demo-{segment}-{int(trimmed)}',label=f'示范第{segment}段 · '+('裁剪后' if trimmed else '原切分'),trace=trace.cpu(),points=torch.cat((torch.zeros_like(demo['point'][a:a+1]),demo['point'][a:b])).cpu(),active=torch.cat((torch.zeros_like(demo['active'][a:a+1]),demo['active'][a:b]&demo['known'][a:b])).cpu(),segment=segment,text=f'Recorded demo frames {a}..{b}; {len(trace)-1} transitions'))
    for kind in [0,1]:
        slots=((g==0)&(variant==kind)).nonzero().flatten();chosen=int(slots[len(slots)//2]);end=int(stop[chosen])+1;trace=torch.cat((data['s'][:1,chosen],data['s1'][:end,chosen])).cpu();videos.append(dict(stem=f'policy-first-{kind}',label='第40轮策略 · '+('新起点，保留真实历史' if kind else '原起点'),trace=trace,points=torch.cat((torch.zeros(1,6,3,device=dev),data['point'][:end,chosen])).cpu(),active=torch.cat((torch.zeros(1,6,dtype=torch.bool,device=dev),data['active'][:end,chosen]&data['known'][:end,chosen])).cpu(),segment=1,text=f'Frozen policy40 | steps {end}/{int(caps[chosen])} | success {bool(success[chosen])}'))
    pool=ThreadPoolExecutor(max_workers=3);futures=[];cards=[]
    for x in videos:
        end=int(clock.event_ends[x['segment']-1])-1;target=demo['point'][end].cpu();new=(clock.target_masks[end]&~clock.keep_masks[end]).cpu();target=target.clone();target[~new]=torch.tensor([0.,0.,-1000.]);ref=demo['s1'][end].cpu();T=len(x['trace']);p=web/(x['stem']+'.npz');trace=x['trace'];np.savez_compressed(p,names=np.array(['lowclimb_0000']),root_p=trace[:,:3].numpy()[:,None],root_quat=trace[:,3:7].numpy()[:,None],q=trace[:,13:42].numpy()[:,None],alive=np.ones((T,1),bool),f=np.arange(T)[:,None],ref_root_p=ref[:3].numpy(),ref_root_R=quat_to_mat(ref[3:7]).numpy(),ref_q=ref[13:42].numpy(),clip_start=np.array([0]),clip_len=np.array([T]),goal_points=target.numpy(),goal_root=ref[:3].numpy(),measured_contact_points=x['points'].numpy(),measured_contact_active=(x['active']&new).numpy(),frame_annotations=np.array([x['text']]*T),**{k:np.asarray(v) for k,v in metadata.items() if isinstance(v,(str,int,float,bool))})
        cmd=[sys.executable,str(ROOT/'tools/render.py'),str(p),str(web/(x['stem']+'.mp4')),'--task_manifest','data/lowclimb_boxes/task.json','--seconds',str(T*.02),'--hide_reference','--fps','50','--width','960','--height','540','--camera_distance','3.0','--title',x['stem']];futures.append(pool.submit(subprocess.run,cmd,cwd=runtime_root(),check=True));cards.append(f'<section><h2>{x["label"]}</h2><p>{x["text"]}</p><video controls playsinline src="{x["stem"]}.mp4"></video></section>')
    for f in futures:f.result()
    pool.shutdown();table=''.join(f'<tr><td>{x["segment"]}</td><td>{x["original_end"]-x["original_start"]}</td><td>{x["trimmed_end"]-x["trimmed_start"]}</td><td>{x["removed_head"]}</td><td>{x["removed_tail"]}</td></tr>' for x in spans)
    rates=[x for x in result if x['segment']==1];ptext='；'.join(('新起点' if x['trimmed'] else '原起点')+f'：{x["success_rate"]*100:.1f}%成功，平均{x["mean_frames"]:.1f}帧，上限{x["budget"]}帧' for x in rates)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>切分首尾站立裁剪</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1100px;margin:auto;padding:24px}section{background:#203047;padding:16px;margin:18px 0}video{width:100%}td,th{padding:6px 15px;border-bottom:1px solid #456}a{color:#9cf}</style><h1>首尾站立裁剪：示范与冻结策略对照</h1><p>只裁剪长时间连续低速等待，保留2帧边界上下文；原基线与目标接触点未修改。末段的目标仍是原终点，裁掉的静止尾巴不用于重新定义目标。</p><p>'+ptext+'</p><table><tr><th>段</th><th>原帧数</th><th>裁剪后</th><th>删开头</th><th>删结尾</th></tr>'+table+'</table>'+''.join(cards)+'<p><a href="report.json">完整切分与冻结策略结果</a></p><script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.5)</script></html>';(web/'index.html').write_text(page);print('TRIM COMPLETE',json.dumps(spans),json.dumps(rates),flush=True);raise SystemExit(0)
ContactTask.step=lambda task:probe(task)
runpy.run_path(str(ROOT/'tools/check_contact_sequence.py'),run_name='__main__')
