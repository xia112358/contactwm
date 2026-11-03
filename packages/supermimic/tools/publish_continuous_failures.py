"""Render direct and inherited-entry failures from existing continuous training replay."""
from supermimic.paths import runtime_root
import argparse,json,sys,subprocess,html
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
import torch,numpy as np
torch.set_num_threads(4)
from supermimic.contact_control import ContactClock
from supermimic.rotations import quat_to_mat
cfg=json.loads((args.run/'config.json').read_text());rec=torch.load(cfg['recording'],map_location='cpu',weights_only=False);demo=rec['data'];clock=ContactClock(demo,hold=1,role_aware=True)
buf=torch.load(args.run/'replay.pt',map_location='cpu',weights_only=False,mmap=True);ix=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T'];bd=buf['data']
starts=(bd['ep_step'][ix]==0).all(-1).nonzero().flatten().tolist();complete=[(a,b) for a,b in zip(starts,starts[1:]+[len(ix)]) if bd['done'][ix[a:b]].any(0).all()];assert complete
begin,finish=complete[-1];rows=ix[begin:finish];goal=bd['goal_index'][rows];ended=bd['stage_done'][rows];passed=bd['stage_success'][rows]
metrics=[json.loads(x) for x in (args.run/'metrics.jsonl').read_text().splitlines()];matches=[]
for m in metrics:
 os=m['native_batch'].get('stage_outcomes',[])
 if all(int((ended&(goal==v['segment']-1)).sum())==v['attempts'] and abs(float(passed[ended&(goal==v['segment']-1)].float().mean())-v['success_rate'])<1e-6 for v in os):matches.append(m)
assert len(matches)==1,('ambiguous replay policy',len(matches));m=matches[0];policy=m['collected_policy_updates']
args.output.mkdir(parents=True,exist_ok=True);labels=['左脚','右脚','左手','右手','左膝','右膝'];reports=[];cards=[]
pool=ThreadPoolExecutor(max_workers=3);renders=[]
selected=[1,3,4,6,7,8,9,10,11,12,18,19]
for segment in selected:
 g=segment-1;at=ended&(goal==g);failure=at&~passed;direct=goal[0]==g
 for kind,mask in [('direct',direct),('handoff',~direct)]:
  candidates=(failure&mask[None]).nonzero()
  attempts=int((at&mask[None]).sum());failures=len(candidates)
  if not failures:continue
  # Fixed middle world in the failure cohort, independent of visual appearance.
  t,w=map(int,candidates[len(candidates)//2]);local=int(bd['stage_step'][rows[t],w])+1;entry=t-local+1
  d={k:bd[k][rows[:t+1],w].clone() for k in ('s','s1','point','active','known','goal_index','stage_step','stage_success','done')}
  end=int(clock.event_ends[g]);new=clock.target_masks[end-1]&~clock.keep_masks[end-1];target=demo['point'][end-1];tol=cfg['contact_tolerances_m'][g]
  valid=d['active'][entry:]&d['known'][entry:];distance=(d['point'][entry:]-target).norm(dim=-1);parts=[]
  for j in new.nonzero().flatten().tolist():
   errors=distance[:,j].masked_fill(~valid[:,j],float('inf'));best=float(errors.min());hits=(valid[:,j]&(distance[:,j]<=tol)).nonzero().flatten()
   parts.append(dict(part=labels[j],best_contact_cm=best*100 if np.isfinite(best) else None,achieved=bool(len(hits)),first_valid=int(hits[0])+1 if len(hits) else None))
  from supermimic.mdp import unflatten_state
  from supermimic.rotations import rot_angle_sq
  final=unflatten_state(d['s1'][-1]);ref=unflatten_state(demo['s1'][end-1]);root=float((final[0]-ref[0]).norm())*100;angle=float(rot_angle_sq(final[1],ref[1]).sqrt());physical=not bool(torch.isfinite(d['s1'][-1]).all()) or float(d['s1'][-1,2])<.15
  unmet=[v['part']+'未接触目标区域' for v in parts if not v['achieved']]
  if root>=cfg['root_tolerance']*100:unmet.append(f'root误差 {root:.1f}cm')
  if angle>=.4:unmet.append(f'姿态误差 {angle:.3f}rad')
  reason=('物理失败' if physical else '达到固定时长上限')+'；'+'；'.join(unmet or ['末帧条件需逐帧查看'])
  stem=f'segment-{segment:02d}-{kind}';trace=torch.cat((d['s'][:1],d['s1']));goals=torch.cat((d['goal_index'][:1],d['goal_index']));ei=clock.event_ends[goals]-1;nm=clock.target_masks[ei]&~clock.keep_masks[ei];pts=demo['point'][ei].clone();pts[~nm]=torch.tensor([0.,0.,-1000.]);reftrace=demo['s1'][ei]
  notes=[f'Policy {policy} | segment {int(goals[0])+1} start']
  for k in range(t+1):
   cur=int(d['goal_index'][k]);cap=cfg['fixed_stage_budgets'][cur];status='PASS: switch target' if d['stage_success'][k] else 'FAIL: stop' if d['done'][k] else ''
   notes.append(f'Segment {cur+1}/19 | step {int(d["stage_step"][k])+1}/{cap} | {status}'+('\n'+reason if k==t else ''))
  npz=args.output/(stem+'.npz');np.savez_compressed(npz,names=np.array(['lowclimb_0000']),root_p=trace[:,:3].numpy()[:,None],root_quat=trace[:,3:7].numpy()[:,None],q=trace[:,13:42].numpy()[:,None],alive=np.ones((len(trace),1),bool),f=np.arange(len(trace))[:,None],ref_root_p=reftrace[:,:3].numpy(),ref_root_R=quat_to_mat(reftrace[:,3:7]).numpy(),ref_q=reftrace[:,13:42].numpy(),clip_start=np.array([0]),clip_len=np.array([len(trace)]),goal_points=pts[0].numpy(),goal_points_by_frame=pts.numpy(),goal_root=reftrace[0,:3].numpy(),goal_root_by_frame=reftrace[:,:3].numpy(),measured_contact_points=torch.cat((torch.zeros(1,6,3),d['point'])).numpy(),measured_contact_active=(torch.cat((torch.zeros(1,6,dtype=torch.bool),d['active']&d['known']))&nm).numpy(),frame_annotations=np.array(notes),**{k:np.asarray(v) for k,v in rec['metadata'].items() if isinstance(v,(str,int,float,bool))})
  renders.append(pool.submit(subprocess.run,[sys.executable,str(ROOT/'tools/render.py'),str(npz),str(args.output/(stem+'.mp4')),'--task_manifest',cfg['task_manifest'],'--seconds',str(len(trace)*.02+1),'--hide_reference','--fps','50','--width','1280','--height','720','--camera_distance','3.0','--title',f'Segment {segment} | {kind} | policy {policy}'],cwd=runtime_root(),check=True))
  report=dict(segment=segment,entry=kind,world=w,policy_updates=policy,local_steps=local,cap=cfg['fixed_stage_budgets'][g],entry_frame=entry,global_steps=t+1,attempts=attempts,failures=failures,reason=reason,root_cm=root,angle_rad=angle,parts=parts,video=stem+'.mp4');reports.append(report)
  label='示范起点' if kind=='direct' else '真实衔接入口';detail='；'.join(v['part']+'：'+(f'最近接触 {v["best_contact_cm"]:.2f}cm' if v['best_contact_cm'] is not None else '未观测到实际接触') for v in parts)
  cards.append(f'<section id="s{segment}-{kind}"><h2>第{segment}段 · {label}</h2><p>本批失败 {failures}/{attempts}；固定选取失败组中间环境 {w}。该段执行 {local}/{report["cap"]} 帧。{html.escape(reason)}</p><video controls playsinline preload="metadata" src="{stem}.mp4"></video><button onclick="jump(this,{entry*.02})">跳到第{segment}段入口</button><button onclick="jump(this,{(t+1)*.02})">结束帧</button><p>{html.escape(detail)}；root {root:.2f}cm；姿态 {angle:.3f}rad。</p></section>')
  (args.output/'report.json').write_text(json.dumps(dict(source_run=str(args.run),sampled_policy_updates=policy,training_iteration=m['iteration'],examples=reports),ensure_ascii=False,indent=2))
  nav=' '.join(f'<a href="#s{x["segment"]}-{x["entry"]}">{x["segment"]} {"示范" if x["entry"]=="direct" else "衔接"}</a>' for x in reports)
  page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>连续训练失败录像</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1200px;margin:auto;padding:20px}section{background:#203047;padding:18px;margin:20px 0;border-radius:12px}video{width:100%}a{color:#9cf;margin:8px}button,select{font:inherit;margin:6px;padding:5px}</style><h1>高失败动作：真实训练录像</h1><p>直接导出保存的最后一批仿真状态，没有重跑策略。此批使用第'+str(policy)+'次更新后的策略（训练记录第'+str(m['iteration'])+'轮、更新前采集）。衔接样本包含前面的完整执行，点击按钮可跳到失败段入口。多部位接触可先后达成；目标接触容差5cm，第12段8cm；root 25cm、姿态0.4rad。箱体使用已修正渲染。</p><p>播放速度 <select onchange="document.querySelectorAll(\'video\').forEach(v=>v.playbackRate=+this.value)"><option>.25</option><option>.5</option><option>1</option></select> <a href="report.json">失败原因与来源</a></p><nav>'+nav+'</nav>'+''.join(cards)+'<script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.25);function jump(b,t){const v=b.parentNode.querySelector("video");v.pause();v.currentTime=Math.min(t,v.duration-.02)}</script></html>'
  (args.output/'index.html').write_text(page);print('PUBLISHED',segment,kind,reason,flush=True)
for future in renders:future.result()
pool.shutdown()
print('COMPLETE',len(reports),flush=True)
