"""Publish recorded whole-sequence executions, including handoff diagnostics."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from supermimic.contact_control import ContactClock
    from supermimic.rotations import quat_to_mat
    torch.set_num_threads(4)
    report=json.loads((args.run/'report.json').read_text());cfg=report['config']
    recording=torch.load(cfg['recording'],map_location='cpu',weights_only=False)
    demo=recording['data'];clock=ContactClock(demo,hold=1,role_aware=True)
    args.output.mkdir(parents=True,exist_ok=True)
    cards=[];summary=[]
    for result in report['results']:
        saved=torch.load(Path(result['output'])/'rollout.pt',map_location='cpu',weights_only=False)
        d=saved['data'];updates=result['policy_updates']
        summary.append(f'<tr><td>{updates}</td><td>{result["completed"]}/{result["attempts"]}</td><td>{result["mean_completed_segments"]:.2f}/19</td><td>{result["max_completed_segments"]}/19</td></tr>')
        for number,example in enumerate(result['examples']):
            i=example['world'];steps=example['global_steps'];stem=f'update-{updates}-world-{i}'
            trace=torch.cat((d['s'][:1,i],d['s1'][:steps,i]))
            segments=torch.cat((d['goal_index'][:1,i],d['goal_index'][:steps,i])).long()
            endpoints=clock.event_ends[segments]-1
            new=clock.target_masks[endpoints]&~clock.keep_masks[endpoints]
            goals=demo['point'][endpoints].clone();goals[~new]=torch.tensor([0.,0.,-1000.])
            roots=demo['s1'][endpoints,:3]
            ref=demo['s1'][endpoints]
            points=torch.cat((torch.zeros(1,6,3),d['point'][:steps,i]))
            active=torch.cat((torch.zeros(1,6,dtype=torch.bool),d['active'][:steps,i]&d['known'][:steps,i]))&new
            notes=['Sequence start | single initialization; no resets between stages']
            completed=0;handoffs=[]
            for t in range(steps):
                g=int(d['goal_index'][t,i])+1
                if bool(d['stage_success'][t,i]):
                    completed+=1
                    handoffs.append(dict(segment=g,global_step=t+1,local_steps=int(d['stage_step'][t,i])+1,
                                         root_cm=float(d['root_error'][t,i])*100,angle_rad=float(d['angle_error'][t,i])))
                state=' | PASS, switch target' if bool(d['stage_success'][t,i]) else ' | STOP' if bool(d['done'][t,i]) else ''
                budget=clock.bounds[g]-clock.bounds[g-1]+cfg['grace_steps']
                notes.append(f'Stage {g}/19 | stage step {int(d["stage_step"][t,i])+1}/{budget} | completed {completed}/19{state}\n'
                             f'Root error {float(d["root_error"][t,i])*100:.2f} cm | orientation {float(d["angle_error"][t,i]):.3f} rad')
            npz=args.output/f'{stem}.npz'
            np.savez_compressed(npz,names=np.array(['lowclimb_0000']),root_p=trace[:,:3].numpy()[:,None],
                root_quat=trace[:,3:7].numpy()[:,None],q=trace[:,13:42].numpy()[:,None],alive=np.ones((len(trace),1),bool),
                f=np.arange(len(trace))[:,None],ref_root_p=ref[:,:3].numpy(),ref_root_R=quat_to_mat(ref[:,3:7]).numpy(),ref_q=ref[:,13:42].numpy(),
                clip_start=np.array([0]),clip_len=np.array([len(trace)]),goal_points=goals[0].numpy(),goal_points_by_frame=goals.numpy(),
                goal_root=roots[0].numpy(),goal_root_by_frame=roots.numpy(),measured_contact_points=points.numpy(),
                measured_contact_active=active.numpy(),frame_annotations=np.array(notes),
                **{k:np.asarray(v) for k,v in recording['metadata'].items() if isinstance(v,(str,int,float,bool))})
            subprocess.run([sys.executable,str(ROOT/'tools/render.py'),str(npz),str(args.output/f'{stem}.mp4'),
                '--task_manifest',cfg['task_manifest'],'--seconds',str((steps+1)*.02+1),'--hide_reference','--fps','50',
                '--width','1280','--height','720','--camera_distance','3.0','--title',f'Continuous execution | update {updates} | world {i}'],check=True,cwd=runtime_root())
            example.update(video=f'{stem}.mp4',handoffs=handoffs)
            selection='完成段数中位样本' if number==0 else '完成段数最多的样本'
            reasons='；'.join(f'{p["part"]}：'+(f'最接近 {p["best_contact_cm"]:.2f} cm' if p['best_contact_cm'] is not None else '未观测到接触')+('，已达标' if p['achieved'] else '，未达标') for p in example['parts'])
            links=' '.join(f'<button onclick="jump(\'{stem}\',{h["global_step"]*.02})">第{h["segment"]}段切换</button>' for h in handoffs)
            cards.append(f'<section><h2>更新 {updates} 次 · {selection}</h2><p>完成 {example["completed_segments"]}/19 段；{example["reason"]}，停在第 {example["stopped_segment"]} 段。执行 {steps} 帧（{steps*.02:.2f}s），当前段用了 {example["stage_steps"]} 帧。</p><video id="{stem}" controls playsinline preload="metadata" src="{stem}.mp4"></video><p>{links}</p><p>{html.escape(reasons)}。结束时root误差 {example["root_cm"]:.2f} cm，姿态误差 {example["angle_rad"]:.3f} rad。</p></section>')
        rows=''.join(f'<tr><td>{s["segment"]}</td><td>{s["entered"]}</td><td>{s["passed"]}</td><td>{s["failed"]}</td><td>{s["mean_entry_root_cm"]:.2f}</td><td>{s["mean_entry_velocity_error_m_s"]:.3f}</td></tr>' for s in result['segments'] if s['entered'])
        cards.append(f'<details><summary>更新 {updates} 次：逐段衔接统计</summary><p>入口误差相对于该段训练时的示范初始状态，仅作诊断，不参与判定。</p><table><tr><th>段</th><th>进入</th><th>完成</th><th>失败</th><th>入口root误差/cm</th><th>入口线速度误差/m·s⁻¹</th></tr>{rows}</table></details>')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>19段连续执行结果</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1280px;margin:24px auto;padding:0 18px}section,details{background:#203047;padding:16px;border-radius:10px;margin:20px 0}video{width:100%}a{color:#9cf}td,th{padding:5px 14px;border-bottom:1px solid #456}button,select{font:inherit;margin:4px;padding:5px}</style><h1>19段连续执行：真实衔接结果</h1><p>4096个环境，其中4057个接触任务世界。只在整段开头初始化；达成后只切换目标，保留真实位置、速度、上一步动作和传感器历史。阶段间没有重置或回到示范状态，策略权重没有更新。</p><p>第12段接触容差8cm，其余5cm；root 25cm / 0.4rad；每段仍为示范段长+10帧，真实接触累计达成。橙色目标和绿色root目标随阶段切换。</p><table><tr><th>策略更新数</th><th>全部完成</th><th>平均完成段数</th><th>最多完成段数</th></tr>'''+''.join(summary)+'''</table><p>播放速度 <select onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=+this.value)"><option value="0.5">0.5×</option><option value="0.25">0.25×</option><option value="1">1×</option></select> · <a href="report.json">完整结果与状态连续性检查</a></p>'''+''.join(cards)+'''<script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.5);function jump(id,t){const v=document.getElementById(id);v.pause();v.currentTime=Math.min(t,v.duration-.02)}</script></html>'''
    page=page.replace('每段仍为示范段长+10帧',f"每段为示范段长+{cfg['grace_steps']}帧")
    if cfg.get('contact_only'):
        page=page.replace('第12段接触容差8cm，其余5cm；root 25cm / 0.4rad；', '本次为接触即通过诊断：只要求指定部位实际接触，可先后累计；不检查落点距离、root或姿态误差；')
        page=page.replace('真实接触累计达成。橙色目标', '保留物理失败与阶段超时。已存在的指定部位接触也计入，可能立即切段。橙色目标')
    (args.output/'index.html').write_text(page)
    print(json.dumps(dict(published=str(args.output.resolve()),variants=len(report['results'])),ensure_ascii=False))


if __name__=='__main__':
    main()
