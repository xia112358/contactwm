"""Publish a representative failure from the latest complete recorded attempt batch."""
from supermimic.paths import runtime_root
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--segment', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection',choices=('median_failure','median_success','first'),default='median_failure')
    parser.add_argument('--label',default='接触策略执行录像')
    args = parser.parse_args()
    import numpy as np
    import torch
    from supermimic.contact_control import ContactClock
    from supermimic.contact_dmo import native_attempt_metrics
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import quat_to_mat, rot_angle_sq
    torch.set_num_threads(4)
    cfg = json.loads((args.run / 'config.json').read_text())
    assert cfg['completion_mode'] == 'accumulated_target' and not cfg['contact_face_matching']
    metrics = [json.loads(x) for x in (args.run / 'metrics.jsonl').read_text().splitlines()]
    recording = torch.load(cfg['recording'], map_location='cpu', weights_only=False)
    demo = recording['data']
    clock = ContactClock(demo, hold=1, role_aware=True)
    buffer = torch.load(args.run / 'replay.pt', map_location='cpu', weights_only=False)
    index = (buffer['ptr'] - buffer['size'] + torch.arange(buffer['size'])) % buffer['T']
    data = {k: v[index] for k, v in buffer['data'].items()}
    del buffer
    starts = (data['ep_step'] == 0).all(-1).nonzero().flatten().tolist()
    complete = [(s, e) for s, e in zip(starts, starts[1:] + [len(index)]) if data['done'][s:e].any(0).all()]
    assert complete, 'No complete synchronized first-attempt batch remains in replay'
    start, end = complete[-1]
    data = {k: v[start:end] for k, v in data.items()}
    actual = native_attempt_metrics(data, clock, None)
    # Establish provenance from the recorded batch, rather than the save filename.
    matches = [row for row in metrics if row.get('native_batch') and
               all(abs(a['success_rate'] - b['success_rate']) < 1e-6 and
                   abs(a['root_error_cm'] - b['root_error_cm']) < 1e-4 and
                   abs(a['mean_terminal_step'] - b['mean_terminal_step']) < 1e-4
                   for a, b in zip(actual['segments'], row['native_batch']['segments']))]
    assert len(matches) == 1, 'Cannot identify the policy used for this replay batch'
    sampled_update = matches[0]['collected_policy_updates']
    segment = args.segment - 1
    tolerance = cfg.get('contact_tolerances_m', [.05]*len(clock.event_ends))[segment]
    slots = (data['goal_index'][0] == segment).nonzero().flatten()
    d = {k: v[:, slots] for k, v in data.items()}
    stop = d['done'].long().argmax(0)
    batch = torch.arange(len(slots))
    live = torch.arange(len(d['done']))[:, None] <= stop
    start, end = clock.bounds[segment:segment+2]
    new = clock.target_masks[end-1] & ~clock.keep_masks[end-1]
    ids = new.nonzero().flatten().tolist()
    target = demo['point'][end-1]
    state = unflatten_state(d['s1'])
    reference = unflatten_state(demo['s1'][end-1])
    root = (state[0] - reference[0]).norm(dim=-1)
    angle = rot_angle_sq(state[1], reference[1]).sqrt()
    observed = d['active'] & d['known'] & live[..., None]
    distance = (d['point'] - target).norm(dim=-1)
    valid = observed & (distance <= tolerance)
    accumulated = valid.long().cummax(0).values.bool()
    physical = (~torch.isfinite(d['s1']).all(-1)) | (state[0][..., 2] < .15)
    complete_now = ((~new) | accumulated).all(-1) & (root < cfg['root_tolerance']) & (angle < .4) & ~physical
    success = ~d['term'][stop, batch]
    assert torch.equal(success, complete_now[stop, batch])
    assert not bool((complete_now & (torch.arange(len(live))[:, None] < stop)).any())
    best = distance.masked_fill(~observed, float('inf')).amin(0)
    failed = (~success).nonzero().flatten()
    if args.selection=='first':
        chosen=0;selection_text='固定首个环境，不按成功或失败挑选'
    else:
        candidates=failed if args.selection=='median_failure' else success.nonzero().flatten()
        assert len(candidates), 'No attempts matching the requested selection'
        rank=best[candidates][:,new].amax(-1).argsort()
        chosen=int(candidates[rank[len(rank)//2]])
        selection_text=('失败组' if args.selection=='median_failure' else '成功组')+'按最近接触误差排序的中位样本'
    selected_success=bool(success[chosen])
    last = int(stop[chosen])
    names = ['左脚', '右脚', '左手', '右手', '左膝', '右膝']
    english = ['Left foot', 'Right foot', 'Left hand', 'Right hand', 'Left knee', 'Right knee']
    parts = []
    for part in ids:
        active = observed[:last+1, chosen, part]
        errors = distance[:last+1, chosen, part].masked_fill(~active, float('inf'))
        t = int(errors.argmin()) if bool(active.any()) else last
        first = active.nonzero().flatten()
        achieved = valid[:last+1, chosen, part].nonzero().flatten()
        parts.append(dict(part=names[part],first_contact_step=int(first[0])+1 if len(first) else None,
                          first_valid_step=int(achieved[0])+1 if len(achieved) else None,
                          best_contact_step=t+1,best_contact_cm=float(errors[t])*100 if bool(active.any()) else None,
                          best_error_xyz_cm=((d['point'][t,chosen,part]-target[part])*100).tolist() if bool(active.any()) else None,
                          target_xyz_m=target[part].tolist(), actual_best_xyz_m=d['point'][t,chosen,part].tolist() if bool(active.any()) else None,
                          batch_observed=int(observed[:,:,part].any(0).sum()),
                          batch_valid=int(valid[:,:,part].any(0).sum())))
    frames = []
    annotations = ['Initial state; orange: targets | cyan / magenta: measured contacts']
    for t in range(last+1):
        part_rows=[]
        text=[]
        for part in ids:
            contact=bool(observed[t,chosen,part]);achieved=bool(accumulated[t,chosen,part])
            cm=float(distance[t,chosen,part])*100 if contact else None
            part_rows.append(dict(part=names[part],contact=contact,error_cm=cm,achieved=achieved))
            value=f'{cm:.2f} cm' if contact else 'no contact'
            text.append(f'{english[part]}: {value} | target achieved: {"yes" if achieved else "no"}')
        frames.append(dict(step=t+1,root_cm=float(root[t,chosen])*100,angle_rad=float(angle[t,chosen]),parts=part_rows))
        annotations.append('\n'.join([f'Step {t+1}/{last+1} | root {float(root[t,chosen])*100:.2f} cm | angle {float(angle[t,chosen]):.3f} rad']+text))
    report=dict(segment=args.segment,source_run=str(args.run.resolve()),sampled_policy_updates=sampled_update,
                scope='Render existing recorded native attempts without changing states or weights.',
                selection=args.selection,selection_description=selection_text,success=selected_success,checkpoint=cfg['resume'],
                frozen_actor=cfg.get('freeze_actor',False),frozen_wm=cfg.get('freeze_wm',False),
                attempts=len(slots),successes=int(success.sum()),failures=len(failed),source_world=int(slots[chosen]),
                demonstration_steps=end-start,budget_steps=cfg['segment_budgets'][segment],steps=last+1,
                end_reason='接触及root条件达成' if selected_success else '达到帧数上限' if last+1==cfg['segment_budgets'][segment] else '物理失败',
                root_cm=float(root[last,chosen])*100,angle_rad=float(angle[last,chosen]),parts=parts,
                failure_counts=dict(root=int((root[stop,batch]>=cfg['root_tolerance']).sum()),
                                    orientation=int((angle[stop,batch]>=.4).sum()),physical=int(physical[stop,batch].sum())),
                criteria=dict(contact_radius_cm=tolerance*100,root_tolerance_cm=cfg['root_tolerance']*100,orientation_rad=.4,sequential_contacts=True),frames=frames)
    args.output.mkdir(parents=True,exist_ok=True)
    trace=torch.cat((d['s'][:1,chosen],d['s1'][:last+1,chosen]))
    ref=torch.cat((demo['s'][start:start+1],demo['s1'][start:end]))
    ref=ref[torch.arange(len(trace)).clamp_max(len(ref)-1)]
    npz=args.output/'attempt.npz'
    np.savez_compressed(npz,names=np.array(['lowclimb_0000']),root_p=trace[:,:3].numpy()[:,None],
                        root_quat=trace[:,3:7].numpy()[:,None],q=trace[:,13:42].numpy()[:,None],alive=np.ones((len(trace),1),bool),
                        f=np.arange(len(trace))[:,None],ref_root_p=ref[:,:3].numpy(),ref_root_R=quat_to_mat(ref[:,3:7]).numpy(),ref_q=ref[:,13:42].numpy(),
                        clip_start=np.array([0]),clip_len=np.array([len(trace)]),goal_points=target[new].numpy(),goal_root=reference[0].numpy(),
                        measured_contact_points=torch.cat((torch.zeros(1,len(ids),3),d['point'][:last+1,chosen][:,ids])).numpy(),
                        measured_contact_active=torch.cat((torch.zeros(1,len(ids),dtype=torch.bool),observed[:last+1,chosen][:,ids])).numpy(),
                        frame_annotations=np.array(annotations),
                        **{k:np.asarray(v) for k,v in recording['metadata'].items() if isinstance(v,(str,int,float,bool))})
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for view,azimuth in [('main',135),('side',225)]:
        subprocess.run([sys.executable,str(ROOT/'tools/render.py'),str(npz),str(args.output/f'{view}.mp4'),
                        '--task_manifest',cfg['task_manifest'],'--seconds','5','--hide_reference','--fps','50',
                        '--width','1280','--height','720','--camera_azimuth',str(azimuth),'--camera_distance','3.0',
                        '--title',f'Segment {args.segment} | update {sampled_update} | world {int(slots[chosen])} | {'PASS' if selected_success else 'FAIL'}'],check=True,cwd=runtime_root())
    rows=''.join(f'<tr><td>{p["part"]}</td><td>{p["best_contact_step"]}</td><td>{p["best_contact_cm"]:.2f} cm</td><td>{p["first_valid_step"] or "未达标"}</td><td>{p["batch_valid"]}/{len(slots)}</td></tr>' for p in parts)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(args.label)} · 第{args.segment}段</title>
<style>body{{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1380px;margin:24px auto;padding:0 18px}}.videos{{display:flex;gap:16px;flex-wrap:wrap}}section{{flex:1;min-width:340px;background:#203047;padding:12px;border-radius:10px}}video{{width:100%;display:block}}td,th{{padding:6px 12px;border-bottom:1px solid #456}}a{{color:#9cf}}button,select{{font:inherit;padding:6px;margin:5px}}#frame{{white-space:pre-wrap;background:#203047;padding:12px}}small{{color:#bac8db}}</style>
<h1>{html.escape(args.label)} · 第{args.segment}段</h1><p>本次记录的策略DMO更新数：{sampled_update}；{len(slots)} 次尝试，{len(failed)} 次失败，{int(success.sum())} 次成功。录像选择：{selection_text}（环境 {int(slots[chosen])}）。结束原因：{report["end_reason"]}。</p>
<p>示范 {end-start} 帧；本次执行 {last+1} 帧。末帧 root 误差 {report['root_cm']:.2f} cm，姿态误差 {report['angle_rad']:.3f} rad。橙色球是目标点；青色球是左侧部位的真实接触点，紫色球是右侧部位的真实接触点；绿色球是 root 目标。</p>
<table><tr><th>部位</th><th>最接近帧</th><th>最小真实接触误差</th><th>首次达标帧</th><th>整批达标数</th></tr>{rows}</table>
<p>本次记录的接触位置容差为 {tolerance*100:g} cm。多部位目标允许先后达成；不要求同步。本批 root / 姿态 / 物理失败计数分别为 {report['failure_counts']['root']} / {report['failure_counts']['orientation']} / {report['failure_counts']['physical']}。录像使用已校验箱体位置和朝向的渲染器，隐藏示范虚影。</p>
<p><button onclick="restart()">从头慢放</button><button onclick="step(-1)">上一帧</button><button onclick="step(1)">下一帧</button><button onclick="last()">结束帧</button><select onchange="speed=+this.value;videos.forEach(v=>v.playbackRate=speed)"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select></p>
<div class="videos"><section><h2>主视角</h2><video controls playsinline preload="auto" src="main.mp4"></video></section><section><h2>另一侧视角</h2><video controls playsinline preload="auto" src="side.mp4"></video></section></div><p id="frame">拖动视频或逐帧查看接触状态。</p>
<p><a href="main.mp4">主视角录像</a> · <a href="side.mp4">侧视角录像</a> · <a href="report.json">逐帧诊断与来源</a></p><small>以上视频按保存的仿真状态渲染。策略来源：{html.escape(cfg["resume"])}；此批DMO更新数 {sampled_update}。本轮是否冻结actor：{cfg.get("freeze_actor",False)}，是否冻结WM：{cfg.get("freeze_wm",False)}。</small>
<script>const frames={json.dumps(frames,ensure_ascii=False)},videos=[...document.querySelectorAll('video')];let speed=.25;videos.forEach(v=>v.playbackRate=speed);function seek(t){{videos.forEach(v=>{{v.pause();v.currentTime=Math.min(Math.max(t,0),v.duration-.02)}});show()}}function restart(){{videos.forEach(v=>{{v.currentTime=0;v.playbackRate=speed;v.play()}})}}function step(d){{seek(Math.round(videos[0].currentTime*50)/50+d/50)}}function last(){{seek(frames.length/50)}}function show(){{let i=Math.round(videos[0].currentTime*50)-1,r=frames[i];document.getElementById('frame').textContent=r?`第 ${{r.step}} 帧 | root ${{r.root_cm.toFixed(2)}} cm | 姿态 ${{r.angle_rad.toFixed(3)}} rad\n`+r.parts.map(p=>`${{p.part}}：${{p.contact?p.error_cm.toFixed(2)+' cm':'无实际接触'}}；累计达标：${{p.achieved?'是':'否'}}`).join('\n'):'初始状态'}}videos[0].addEventListener('timeupdate',show);</script></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
