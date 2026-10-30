"""Render already-collected trimming comparison; no new simulation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,json,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];base=runtime_path('tmp/contact-tracker-20261027-v1');tracking=bool(os.environ.get('TRIM_TRACKING_TEACHER'));web=runtime_path('results/mixed_comparison_web')/('contact-stationary-trim-tracking' if tracking else 'contact-stationary-trim');report=json.loads((base/('stationary-trim-tracking-v1' if tracking else 'stationary-trim-v1')/'trim-report.json').read_text());metadata=torch.load(base/'demo0-roles-v1/rollout.pt',map_location='cpu',weights_only=False)['metadata'];pool=ThreadPoolExecutor(max_workers=3);jobs=[];cards=[]
labels={'demo-1-0':'第1段示范：原切分','demo-1-1':'第1段示范：裁剪后','demo-19-0':'第19段示范：原切分','demo-19-1':'第19段示范：裁剪后','policy-first-0':'冻结第40轮策略：原起点','policy-first-1':'冻结第40轮策略：新起点与相应缩短的时限'}
if tracking:labels={k:v.replace('冻结第40轮策略','原始tracking teacher') for k,v in labels.items()}
for stem,label in labels.items():
 p=web/(stem+'.npz');d=np.load(p);a={k:d[k] for k in d.files};d.close();
 if tracking:a['frame_annotations']=np.array([str(v).replace('Frozen policy40','Tracking teacher6000') for v in a['frame_annotations']])
 a.update({k:np.asarray(v) for k,v in metadata.items() if isinstance(v,(str,int,float,bool))});np.savez_compressed(p,**a)
 cmd=[sys.executable,str(ROOT/'tools/render.py'),str(p),str(web/(stem+'.mp4')),'--task_manifest','data/lowclimb_boxes/task.json','--seconds',str(len(a['f'])*.02),'--hide_reference','--fps','50','--width','960','--height','540','--camera_distance','3.0','--title',stem];
 if not tracking or stem.startswith('policy') or not (web/(stem+'.mp4')).exists():jobs.append(pool.submit(subprocess.run,cmd,cwd=runtime_root(),check=True))
 cards.append(f'<section><h2>{label}</h2><p>{len(a["f"])-1}帧，{(len(a["f"])-1)*.02:.2f}秒</p><video controls playsinline src="{stem}.mp4"></video></section>')
for job in jobs:job.result()
pool.shutdown();table=''.join(f'<tr><td>{x["segment"]}</td><td>{x["original_end"]-x["original_start"]}</td><td>{x["trimmed_end"]-x["trimmed_start"]}</td><td>{x["removed_head"]}</td><td>{x["removed_tail"]}</td></tr>' for x in report['spans']);first=[x for x in report['results'] if x['segment']==1];note='；'.join(('新起点' if x['trimmed'] else '原起点')+f'成功率{x["success_rate"]*100:.1f}%，平均{x["mean_frames"]:.1f}帧，上限{x["budget"]}帧' for x in first)
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>裁剪站立段对照</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1100px;margin:auto;padding:24px}section{background:#203047;padding:16px;margin:18px 0}video{width:100%}td,th{padding:6px 15px;border-bottom:1px solid #456}a{color:#9cf}</style><h1>切分首尾站立裁剪实验</h1><p>示范第1段删去开头44帧（包括初始化沉降和等待），保留正式运动前2帧上下文；最后一段删去尾部67帧站立。中间各段没有满足条件的长时间低速首尾，保持原切分。原基线和目标点未修改。</p><p>'+note+'</p><p>冻结策略的新起点保留真实历史；此对照尚未对裁剪后的技能重新训练。末段策略本来约15帧就完成，原示范109帧的长尾已经不计入其实际执行。</p><p>播放速度 <select onchange="document.querySelectorAll(\'video\').forEach(v=>v.playbackRate=+this.value)"><option>1</option><option>.5</option><option>.25</option></select></p><table><tr><th>段</th><th>原帧数</th><th>裁剪后</th><th>删开头</th><th>删结尾</th></tr>'+table+'</table>'+''.join(cards)+'<p><a href="report.json">完整切分与冻结策略结果</a></p></html>'

if tracking:page=page.replace('冻结策略的新起点','Tracking teacher的新起点').replace('此对照尚未对裁剪后的技能重新训练。','Teacher权重未更新，新起点直接恢复其原参考时钟和历史。')
(web/'index.html').write_text(page);print('COMPLETE 6 trimming videos',flush=True)
