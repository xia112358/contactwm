"""Render saved native contact-policy executions using the canonical robot."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_root',type=Path,required=True)
    parser.add_argument('--publish_dir',type=Path,required=True)
    parser.add_argument('--task_manifest',type=Path,default=runtime_path('data/lowclimb_boxes/task.json'))
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--native_iteration',type=int,default=700)
    parser.add_argument('--initial_iteration',type=int,default=500)
    parser.add_argument('--recording',type=Path)
    args=parser.parse_args()
    import torch
    import numpy as np
    from supermimic.rotations import quat_to_mat
    recording=args.recording or args.run_root/'demo0-seed0/rollout.pt'
    demo=torch.load(recording,map_location='cpu',weights_only=False)
    source=demo['data']
    ref=torch.cat((source['s'][:1],source['s1']),0)
    args.publish_dir.mkdir(parents=True,exist_ok=True)
    variants=[('initial','学习前 · 53 步',args.run_root/'demo0-fixed-demo/native_000000.pt'),
              ('update300','训练 300 次 · 88 步',args.run_root/'demo0-fixed-demo/native_000300.pt'),
              ('tracker','原始追踪示范 · 799 步',args.run_root/'demo0-seed0/rollout.pt')]
    if args.experiment:
        variants=[('initial','BC 初始化',args.experiment/f'native_{args.initial_iteration:06d}.pt'),
                  ('optimized','单步接触优化 · 首个接触达成',args.experiment/f'native_{args.native_iteration:06d}.pt'),
                  ('tracker','完整追踪示范',recording)]
    reports=[]
    for name,title,path in variants:
        item=torch.load(path,map_location='cpu',weights_only=False)
        d=item['data'];states=torch.cat((d['s'][:1],d['s1']),0)
        count=len(states)
        alive=np.ones((count,1),bool)
        alive[-1,0]=not bool(d['term'][-1])
        data=dict(names=np.array(['lowclimb_0000']),root_p=states[:,:3].numpy()[:,None],
            root_quat=states[:,3:7].numpy()[:,None],q=states[:,13:42].numpy()[:,None],
            f=np.arange(count,dtype=np.int64)[:,None],alive=alive,
            ref_root_p=ref[:,:3].numpy(),ref_root_R=quat_to_mat(ref[:,3:7]).numpy(),
            ref_q=ref[:,13:42].numpy(),clip_start=np.array([0]),clip_len=np.array([len(ref)]),
            **{k:np.asarray(v) for k,v in demo['metadata'].items() if isinstance(v,(str,int,float,bool))})
        if args.experiment and name!='tracker' and 'goal_index' in d:
            from supermimic.contact_control import ContactClock
            clock=ContactClock(source,hold=1,role_aware=True)
            goals=torch.cat((d['goal_index'],d['goal_index'][-1:]))
            frames=clock.event_ends[goals.long()]-1
            parts=clock.target_masks[frames].float().argmax(-1)
            data['goal_point']=source['point'][frames,parts].numpy()[:,None]
            data['goal_point'][~clock.target_masks[frames].any(-1).numpy(),0]=[0,0,-1000]
        np.savez_compressed(args.publish_dir/f'{name}.npz',**data)
        command=[sys.executable,str(ROOT/'tools/render.py'),str(args.publish_dir/f'{name}.npz'),
            str(args.publish_dir/f'{name}.mp4'),'--task_manifest',str(args.task_manifest),
            '--clips','lowclimb_0000','--seconds','20','--width','960','--height','540',
            '--title',(f'{name} | {len(d["s"])} native steps' if args.experiment else f'{name} | {len(d["s"])}/799 native steps')]
        if name=='tracker':
            command.append('--hide_reference')
        subprocess.run(command,cwd=runtime_root(),check=True)
        reports.append(dict(name=name,title=title,frames=len(d['s']),seconds=len(d['s'])*.02,
                            source=str(path.resolve()),completed=item.get('report',{}).get('completed',not bool(d['term'].any()) and len(d['s'])==799),
                            native_report=item.get('report')))
    (args.publish_dir/'provenance.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    cards=''.join(f'<section><h2>{r["title"]}</h2><video id="{r["name"]}" controls loop playsinline preload="metadata" src="{r["name"]}.mp4"></video><p>{r["seconds"]:.2f} 秒 · {r["frames"]}/799 个实际控制步</p></section>' for r in (reports[1],reports[0],reports[2]))
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原生接触策略回放 · 88 步</title>
<style>body{margin:0;background:#10151e;color:#e8edf5;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:28px 20px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px}p{color:#b3bfd2}.note{border-left:3px solid #e6ac57;padding:10px 16px;background:#212632}section{background:#1a2230;border:1px solid #344154;border-radius:12px;padding:14px 18px;margin:22px 0}video{display:block;width:100%;max-height:620px;background:#000;border-radius:8px}button,select{font:inherit;color:#e8edf5;border:1px solid #556783;border-radius:7px;background:#253249;padding:7px 14px;margin-right:10px}a{color:#96c8ff}</style>
<main><h1>接触策略：训练 300 次后的真实执行</h1><p>低攀爬示范 motion 0。实心机器人是学习策略在 Newton/MJWarp 中实际执行的状态；蓝色半透明机器人是原始追踪示范。</p>
<div class="note">这一版实际执行了 <b>88 步（1.76 秒）</b> 后提前终止。完整示范为 799 步（15.98 秒），目前尚未学会完整动作。下方是保存的原生轨迹回放。</div>
<p><button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play()})">从头播放全部</button><label>播放速度 <select onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=Number(this.value))"><option value="0.25">0.25× 慢放</option><option value="0.5">0.5×</option><option value="1">1×</option></select></label></p>'''+cards+'''<p><a href="provenance.json">轨迹来源与控制步数</a></p></main><script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.25);</script></html>'''
    (args.publish_dir/'index.html').write_text(html)
    if args.experiment:
        native=reports[1]['native_report']
        heading='单步接触优化：第一个接触的真实执行'
        note=(f'固定检查点完成了 <b>{native["successful_goals"]}/{native["total_goals"]} 个接触阶段</b>。'
              '新接触到达后立即切换，没有统一停留时间。'
              +('整段接触序列已完成。' if native['completed'] else '整段示范尚未完成，后续仍会失稳。'))
        new_cards=''.join(f'<section><h2>{r["title"]}</h2><video controls loop playsinline preload="metadata" src="{r["name"]}.mp4"></video><p>{r["seconds"]:.2f} 秒 · {r["frames"]} 个实际控制步'+
            (f' · {r["native_report"]["successful_goals"]}/{r["native_report"]["total_goals"]} 接触阶段达成 · {r["native_report"]["policy_updates"]} 次策略更新' if r['native_report'] else ' · 示范参考')+'</p></section>' for r in (reports[1],reports[0],reports[2]))
        first=html.index('<main>')
        html=html[:first]+f'<main><h1>{heading}</h1><p>实心机器人：学习策略的原生执行。蓝色：追踪示范。橙色：当前新接触目标，到达后切换。</p><div class="note">{note}</div><p><label>播放速度 <select onchange="document.querySelectorAll(\'video\').forEach(v=>v.playbackRate=Number(this.value))"><option value="0.25">0.25× 慢放</option><option value="0.5">0.5×</option><option value="1">1×</option></select></label></p>'+new_cards+'<p><a href="provenance.json">原生报告与轨迹来源</a></p></main><script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.25);</script></html>'
        (args.publish_dir/'index.html').write_text(html)
    print(json.dumps(dict(published=str(args.publish_dir.resolve()),reports=reports),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
