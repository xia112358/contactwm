"""Publish failed first attempts from a frozen-policy contact replay."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--segments',type=int,nargs='+',help='Render these 1-based segments, including successes')
    parser.add_argument('--no_render',action='store_true',help='Only publish trajectory diagnostics')
    args = parser.parse_args()
    import numpy as np
    import torch
    from supermimic.contact_control import ContactClock, assess_region_arrival, new_part_contact
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import quat_to_mat, rot_angle_sq
    torch.set_num_threads(4)
    cfg = json.loads((args.run/'config.json').read_text())
    root_tolerance=cfg.get('root_tolerance',.1)  # Historical runs retain their recorded criterion.
    status = json.loads((args.run/'status.json').read_text())
    assert status['phase'] == 'complete'
    assert status['actor_frozen_unchanged'] and status['wm_frozen_unchanged']
    assert 'contact_tolerances_m' in cfg or 'within 5cm target region' in cfg['completion']
    buf = torch.load(args.run/'replay.pt', map_location='cpu', weights_only=False)
    # Require complete initial attempts; a wrapped buffer cannot establish this.
    assert buf['ptr'] == buf['size'] and buf['size'] < buf['T']
    data = {k:v[:buf['size']] for k,v in buf['data'].items()}
    assert bool((data['ep_step'][0] == 0).all())
    rec = torch.load(cfg['recording'], map_location='cpu', weights_only=False)
    demo = rec['data']; clock = ContactClock(demo, hold=1, role_aware=True)
    parts = ['左脚', '右脚', '左手', '右手', '左膝', '右膝']
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for g in data['goal_index'][0].unique().tolist():
        if args.segments and g+1 not in args.segments: continue
        slots = (data['goal_index'][0] == g).nonzero().flatten()
        assert len(slots) == 1, 'This publication describes one first attempt per segment'
        slot = int(slots[0]); d = {k:v[:,slot] for k,v in data.items()}
        last = int(d['done'].nonzero()[0]); steps = last+1
        d = {k:v[:steps] for k,v in d.items()}
        start, end = clock.bounds[g:g+2]
        tolerance=cfg.get('contact_tolerances_m',[.05]*len(clock.event_ends))[g]
        keep = clock.keep_masks[end-1]; new = clock.target_masks[end-1] & ~keep
        target = demo['point'][end-1]
        face_matching=cfg.get('contact_face_matching',True)
        faces=torch.tensor(cfg['target_faces_by_segment'][g]) if face_matching else None
        arrived, retained = assess_region_arrival(d, target, faces, new, keep, clock.support_masks[end-1] if face_matching else None,tolerance=tolerance)
        state = unflatten_state(d['s1']); ref = unflatten_state(demo['s1'][end-1])
        root = (state[0]-ref[0]).norm(dim=-1)
        angle = rot_angle_sq(state[1], ref[1]).sqrt()
        distances = (d['point']-target).norm(dim=-1)
        point = distances.masked_fill(~new, 0).max(-1).values
        observed = d['known'] & d['active']
        same_face = d['surface_local'] == faces if face_matching else torch.ones_like(observed)
        eligible = ((~new) | (observed & same_face)).all(-1)
        physical = ~torch.isfinite(d['s1']).all(-1) | (state[0][:,2] < .15)
        completed = arrived & (root < root_tolerance) & (angle < .4) & ~physical
        contact_events=torch.zeros_like(arrived)
        onset_steps=torch.zeros_like(new,dtype=torch.long)
        mode=cfg.get('contact_stop_mode')
        if mode in ('requested_part_onset','requested_part_onset_accumulated'):
            initial=max(0,start-1)
            previous=demo['active'][initial]&demo['known'][initial]
            renewed=torch.zeros_like(previous)
            saved_point=torch.zeros_like(target);saved_face=torch.full_like(onset_steps,-1)
            for t in range(steps):
                current=observed[t].clone()
                first=current&~previous&~renewed&new
                onset_steps=torch.where(first,t+1,onset_steps)
                saved_point=torch.where(first[:,None],d['point'][t],saved_point)
                saved_face=torch.where(first,d['surface_local'][t],saved_face)
                contact_events[t],renewed=new_part_contact(current,previous,renewed,new)
                if mode=='requested_part_onset':
                    renewed &= current
                    contact_events[t]=((~new)|renewed).all()&new.any()
                else:
                    observed[t]=torch.where(new,renewed,observed[t])
                    same_face[t]=torch.where(new,saved_face==faces if face_matching else torch.ones_like(new),same_face[t])
                    distances[t]=torch.where(new,(saved_point-target).norm(dim=-1),distances[t])
                previous=current
            point=distances.masked_fill(~new,0).max(-1).values
            eligible=((~new)|(observed&same_face)).all(-1)
            arrived=((~new)|(observed&same_face&(distances<=tolerance))).all(-1)
            completed=arrived&(root<root_tolerance)&(angle<.4)&~physical
            if new.any(): completed &= contact_events
        if mode=='accumulated_target':
            arrived=((~new)|(observed&same_face&(distances<=tolerance)).long().cummax(0).values.bool()).all(-1)
            completed=arrived&(root<root_tolerance)&(angle<.4)&~physical
        if not new.any(): completed &= retained
        assert not completed[:-1].any(), 'Attempt continued after success'
        success = bool(completed[-1])
        assert steps <= cfg['segment_budgets'][g]
        # Select the closest simultaneous target/body state, not separate minima.
        score = torch.maximum(torch.maximum(root/root_tolerance, angle/.4), point/tolerance)
        score = score.masked_fill(~eligible, float('inf'))
        best = int(score.argmin()) if score.isfinite().any() else steps-1
        def describe(t):
            issues = []
            for part in new.nonzero().flatten().tolist():
                if mode=='accumulated_target' and bool((observed[:t+1,part]&same_face[:t+1,part]&(distances[:t+1,part]<=tolerance)).any()): continue
                if not observed[t,part]: issues.append(parts[part]+'未实际接触')
                elif not same_face[t,part]: issues.append(parts[part]+'接触了错误台面')
                elif distances[t,part] > tolerance: issues.append(f'{parts[part]}落点误差 {float(distances[t,part])*100:.1f} cm')
            if root[t] >= root_tolerance: issues.append(f'root 误差 {float(root[t])*100:.1f} cm')
            if angle[t] >= .4: issues.append(f'姿态误差 {float(angle[t]):.2f} rad')
            if not new.any() and not retained[t]: issues.append('当前支撑条件未满足')
            if physical[t]: issues.append('物理失败：非有限状态或 root 高度低于 15 cm')
            return dict(step=t+1, root_cm=float(root[t])*100, contact_cm=float(point[t])*100 if bool(((~new)|observed[t]).all()) else None,
                        angle_rad=float(angle[t]), issues=issues)
        report = dict(segment=g+1, success=success, steps=steps, demonstration_steps=end-start,contact_tolerance_m=tolerance,
                      budget_steps=cfg['segment_budgets'][g], new_parts=[parts[i] for i in new.nonzero().flatten().tolist()],
                      end_reason='成功' if success else '物理失败' if physical[-1] else '指定部位新接触已发生，但目标条件未达标' if contact_events[-1] else '目标部位接触已发生，但落点或 root 未达标' if eligible[-1] and new.any() and steps<cfg['segment_budgets'][g] else '达到帧数上限',
                      closest=describe(best), final=describe(steps-1),
                      ever_contact_arrived=bool(arrived.any()), ever_root_arrived=bool((root<root_tolerance).any()),
                      ever_target_face_contact=bool(eligible.any()),
                      closest_target_face_contact=(describe(int(point.masked_fill(~eligible,float('inf')).argmin())) if eligible.any() else None),
                      per_part=[dict(part=parts[i],ever_contact=bool(observed[:,i].any()),
                          ever_target_face=bool((observed[:,i]&same_face[:,i]).any())) for i in new.nonzero().flatten().tolist()],
                      onset_steps={parts[i]:int(onset_steps[i]) for i in new.nonzero().flatten().tolist()},
                      source_world=slot)
        if not args.no_render and (not success or args.segments):
            trace = torch.cat([d['s'][:1], d['s1']])
            reference = torch.cat([demo['s'][start:start+1], demo['s1'][start:end]])
            reference = reference[torch.arange(len(trace)).clamp_max(len(reference)-1)]
            stem = f'segment-{g+1:02d}'
            np.savez_compressed(args.output/(stem+'.npz'), names=np.array(['lowclimb_0000']),
                root_p=trace[:,:3].numpy()[:,None], root_quat=trace[:,3:7].numpy()[:,None], q=trace[:,13:42].numpy()[:,None],
                alive=np.ones((len(trace),1),bool), f=np.arange(len(trace))[:,None],
                ref_root_p=reference[:,:3].numpy(), ref_root_R=quat_to_mat(reference[:,3:7]).numpy(), ref_q=reference[:,13:42].numpy(),
                clip_start=np.array([0]), clip_len=np.array([len(reference)]), goal_points=target[new].numpy(), goal_root=ref[0].numpy(),
                **{k:np.asarray(v) for k,v in rec['metadata'].items() if isinstance(v,(str,int,float,bool))})
            report['video'] = stem+'.mp4'
            subprocess.run([sys.executable, str(ROOT/'tools/render.py'), str(args.output/(stem+'.npz')),
                str(args.output/(stem+'.mp4')), '--task_manifest', cfg['task_manifest'], '--seconds', '5',
                '--hide_reference', '--width', '960', '--height', '540', '--title', f'Segment {g+1} | current criteria | '+('SUCCESS' if success else 'FAILED')], check=True, cwd=runtime_root())
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    report = dict(run=str(args.run.resolve()), policy_updates=status['policy_updates'], wm_updates=status['wm_updates'], checkpoint=cfg['resume'], completion=cfg['completion'], root_tolerance=root_tolerance, stop_rule=cfg.get('stop_rule'),
                  grace_steps=cfg['segment_grace_steps'], scope='One first attempt per segment, not an estimated success rate', segments=reports)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    failures = [r for r in reports if not r['success']]
    rows = ''.join(f'<tr><td>{r["segment"]}</td><td>{r["contact_tolerance_m"]*100:g} cm</td><td>{"成功" if r["success"] else "失败"}</td><td>{r["steps"]}/{r["budget_steps"]}</td><td>{html.escape("；".join(r["closest"]["issues"])) if not r["success"] else "—"}</td></tr>' for r in reports)
    cards = ''.join(f'<section id="segment-{r["segment"]}"><h2>第 {r["segment"]} 段：{r["end_reason"]}</h2><video controls playsinline preload="metadata" src="{r["video"]}"></video><p>示范 {r["demonstration_steps"]} 帧；本次 {r["steps"]} 帧；上限 {r["budget_steps"]} 帧。</p><p>诊断帧：第 {r["closest"]["step"]} 帧，{html.escape("；".join(r["closest"]["issues"]))}。</p><p>结束时：{html.escape("；".join(r["final"]["issues"]))}。</p></section>' for r in reports if 'video' in r)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新标准下的失败段</title><style>body{{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1100px;margin:24px auto;padding:0 16px}}section{{background:#203047;padding:18px;margin:24px 0;border-radius:12px}}video{{width:100%}}td,th{{padding:7px 14px;border-bottom:1px solid #456}}a{{color:#9cf}}</style><h1>新标准：{len(reports)-len(failures)}/{len(reports)} 段本次成功</h1><p>每段独立执行一次，使用保存的策略权重（累计更新 {status["policy_updates"]} 次），本次回放不更新权重。指定部位实际接触目标区域（各段容差见表）（{"匹配目标面" if cfg.get("contact_face_matching",True) else "不做面匹配"}）；root 误差小于 {root_tolerance*100:g} cm / 0.4 rad；示范段长 + {cfg['segment_grace_steps']} 帧；之前短暂断触不再永久否决成功。</p><p>录像已修正箱体朝向，隐藏示范虚影；橙色是接触目标，绿色是 root 目标。展示所选段的本次执行，不代表长期成功率。截断规则：{html.escape(cfg.get("stop_rule","见原运行配置"))}。诊断帧选各目标部位同时具有有效接触时最接近达标的一帧；若从未出现，则显示结束帧。接触缺失时不把无效接触点坐标当作落点误差。</p><p>播放速度 <select onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=+this.value)"><option value="0.5">0.5×</option><option value="0.25">0.25×</option><option value="1">1×</option></select> · <a href="report.json">详细结果</a></p><table><tr><th>段</th><th>接触容差</th><th>结果</th><th>执行帧 / 上限</th><th>最接近时仍未满足的条件</th></tr>{rows}</table>{cards}<script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.5)</script></html>'''
    (args.output/'index.html').write_text(page)


if __name__ == '__main__':
    main()
