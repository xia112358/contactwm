"""Audit complete frozen-policy replay and publish first-attempt segment videos."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import torch
    from supermimic.contact_control import ContactClock, PARTS, assess_arrival
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import quat_to_mat, rot_angle_sq
    torch.set_num_threads(4)
    cfg = json.loads((args.run / 'config.json').read_text())
    status = json.loads((args.run / 'status.json').read_text())
    assert status['actor_frozen_unchanged'] and status['wm_frozen_unchanged']
    rec = torch.load(cfg['recording'], map_location='cpu', weights_only=False)
    demo = rec['data']; clock = ContactClock(demo, hold=1, role_aware=True)
    replay = torch.load(args.run / 'replay.pt', map_location='cpu', weights_only=False)
    assert replay['size'] == 200 and replay['ptr'] == 0, 'Requires complete 200-step capture'
    d = replay['data']; T, N = d['done'].shape
    assert bool((d['ep_step'][0] == 0).all())
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(run=str(args.run.resolve()), checkpoint=cfg['resume'],
                  actor_unchanged=True, wm_unchanged=True, segments=[])
    cn = ['左脚', '右脚', '左手', '右手', '左膝', '右膝']
    names = lambda mask: [cn[i] for i in mask.nonzero().flatten().tolist()]
    cards = []
    for g, (begin, end) in enumerate(zip(clock.bounds, clock.bounds[1:])):
        slots = (d['goal_index'][0] == g).nonzero().flatten()
        target = demo['point'][end-1]; keep = clock.keep_masks[end-1]
        new = clock.target_masks[end-1] & ~keep; support = clock.support_masks[end-1]
        e = {k: d[k][:, slots] for k in ('active', 'known', 'point', 'load')}
        arrived, kept = assess_arrival(e, target, new, keep, support)
        states = unflatten_state(d['s1'][:, slots])
        ref = unflatten_state(demo['s1'][end-1])
        root = (states[0]-ref[0]).norm(dim=-1)
        angle = rot_angle_sq(states[1], ref[1]).sqrt()
        point = (e['point']-target).norm(dim=-1).masked_fill(~new, 0).max(-1).values
        failed = ~torch.isfinite(d['s1'][:, slots]).all(-1) | (states[0][..., 2] < .15)
        completed = arrived & (root < .1) & (angle < .4) & (True if new.any() else kept) & ~failed
        done = d['done'][:, slots]
        handoff = torch.zeros(len(slots), dtype=torch.bool)
        violation = handoff.clone(); success = torch.zeros_like(done); violated = done.clone()
        for t in range(T):
            handoff |= arrived[t] & new.any()
            violation |= ~handoff & ~kept[t]
            violated[t] = violation
            success[t] = completed[t] & ~violation
            expected = completed[t] | failed[t] | (d['ep_step'][t, slots]+1 >= cfg['segment_budget'])
            assert torch.equal(done[t], expected), (g, t, 'done mismatch')
            handoff[done[t]] = False; violation[done[t]] = False
        assert int(success.sum()) == status['native']['segments'][g]['successful_attempts']
        first_end = done.to(torch.int64).argmax(0)
        assert done.any(0).all()
        first_success = success[first_end, torch.arange(len(slots))]
        reasons = dict(success=int(success.sum()), physical_failure=int((done & failed).sum()),
                       reached_but_keep_violated=int((done & completed & violated).sum()),
                       timeout=int((done & ~completed & ~failed).sum()))
        segment = dict(segment=g+1, new=names(new), keep=names(keep), support=names(support),
                       attempts=int(done.sum()), reasons=reasons,
                       first_attempts=len(slots), first_successes=int(first_success.sum()), videos=[])
        # Deterministic examples: first world, plus the opposite first-attempt outcome when available.
        chosen = [0]
        opposite = (first_success != first_success[0]).nonzero().flatten()
        if len(opposite): chosen.append(int(opposite[0]))
        first_category = torch.where(first_success, 0, torch.where(
            failed[first_end, torch.arange(len(slots))], 1, torch.where(
                completed[first_end, torch.arange(len(slots))], 2, 3)))
        for category in first_category.unique().tolist():
            if not any(int(first_category[i]) == category for i in chosen):
                chosen.append(int((first_category == category).nonzero()[0]))
        for local in chosen:
            slot = int(slots[local]); last = int(first_end[local]); length = last+1
            ok = bool(first_success[local])
            observed = e['known'][:length, local] & e['active'][:length, local]
            missing = keep & ~observed
            unloaded = support & ~(e['known'][:length, local] & (e['load'][:length, local] >= 1))
            eligible = ~torch.cummax((arrived[:length, local] & new.any()).int(), 0).values.bool()
            broken = (missing | unloaded) & eligible[:, None]
            broken_at = broken.any(-1).nonzero().flatten()
            reason = ('成功' if ok else '物理失败（非有限状态或 root 高度低于 15cm）' if failed[last, local]
                      else '已到达，但此前保持接触/承重约束被违反' if completed[last, local]
                      else '200 步超时，未同时满足目标条件')
            info = dict(world_slot=slot, steps=length, success=ok, reason=reason,
                        root_cm=float(root[last, local]*100), angle_rad=float(angle[last, local]),
                        contact_cm=float(point[last, local]*100), arrived=bool(arrived[last, local]),
                        ever_arrived=bool(arrived[:length, local].any()),
                        min_root_cm=float(root[:length, local].min()*100),
                        min_angle_rad=float(angle[:length, local].min()),
                        minimum_height_m=float(states[0][:length, local, 2].min()),
                        first_keep_violation_step=int(broken_at[0])+1 if len(broken_at) else None,
                        violated_parts=names(broken.any(0)),
                        final_parts=[dict(part=cn[i], active=bool(observed[-1,i]),
                                          error_cm=float((e['point'][last,local,i]-target[i]).norm()*100))
                                     for i in new.nonzero().flatten().tolist()])
            info['arrival_frames'] = [dict(step=t+1, root_cm=float(root[t,local]*100),
                angle_rad=float(angle[t,local]), contact_cm=float(point[t,local]*100))
                for t in arrived[:length,local].nonzero().flatten().tolist()]
            info['keep_violation_frames'] = [dict(step=t+1,
                missing_contact=names(missing[t] & eligible[t]),
                insufficient_load=names(unloaded[t] & eligible[t]),
                load_N=e['load'][t,local].tolist()) for t in broken_at.tolist()]
            trace = torch.cat((d['s'][:1, slot], d['s1'][:length, slot]))
            reference = torch.cat((demo['s'][begin:begin+1], demo['s1'][begin:end]))
            frame = np.minimum(np.arange(len(trace)), len(reference)-1)
            stem = f'segment-{g+1:02d}-world-{slot}'
            data = dict(names=np.array(['lowclimb_0000']), root_p=trace[:,:3].numpy()[:,None],
                        root_quat=trace[:,3:7].numpy()[:,None], q=trace[:,13:42].numpy()[:,None],
                        alive=np.ones((len(trace),1),bool), f=frame[:,None],
                        ref_root_p=reference[:,:3].numpy(), ref_root_R=quat_to_mat(reference[:,3:7]).numpy(),
                        ref_q=reference[:,13:42].numpy(), clip_start=np.array([0]), clip_len=np.array([len(reference)]),
                        **{k:np.asarray(v) for k,v in rec['metadata'].items() if isinstance(v,(str,int,float,bool))})
            # All new target points, including multi-part arrivals, are shown.
            data['goal_points'] = target[new].numpy()
            data['goal_root'] = ref[0].numpy()
            np.savez_compressed(args.output / (stem+'.npz'), **data)
            info['video'] = stem+'.mp4'; segment['videos'].append(info)
            command = [sys.executable, str(ROOT/'tools/render.py'), str(args.output/(stem+'.npz')),
                       str(args.output/(stem+'.mp4')), '--task_manifest', cfg['task_manifest'],
                       '--seconds', '5', '--width', '960', '--height', '540', '--fps', '25',
                       '--title', f'Segment {g+1} | {"SUCCESS" if ok else "FAILED"} | native replay']
            if not (args.output/(stem+'.mp4')).exists():
                subprocess.run(command, cwd=runtime_root(), check=True)
        report['segments'].append(segment)
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(segment, ensure_ascii=False), flush=True)
    def details(v):
        text = ''
        if not v['success'] and v['arrival_frames']:
            a = v['arrival_frames'][0]
            text += f"首次接触到达在第 {a['step']} 步，当时 root 误差 {a['root_cm']:.1f} cm、姿态误差 {a['angle_rad']:.2f} rad。"
        if v['keep_violation_frames']:
            b = v['keep_violation_frames'][0]
            text += f"首次违规：接触缺失 {'、'.join(b['missing_contact']) or '无'}；承重不足 {'、'.join(b['insufficient_load']) or '无'}。到达前共 {len(v['keep_violation_frames'])} 帧违规。"
        return html.escape(text)

    for s in report['segments']:
        videos = ''.join('<div><video controls playsinline preload="metadata" src="'+v['video']+'"></video><p>'+
                        html.escape(f"{v['reason']}；{v['steps']} 步。终点 root {v['root_cm']:.1f} cm，姿态误差 {v['angle_rad']:.2f} rad，接触点误差 {v['contact_cm']:.1f} cm。")+
                        (html.escape(f"第 {v['first_keep_violation_step']} 步首次违反保持/承重要求：{'、'.join(v['violated_parts'])}。") if v['first_keep_violation_step'] else '')+details(v)+'</p></div>' for v in s['videos'])
        r = s['reasons']
        cards.append(f'<section id="segment-{s["segment"]}"><h2>第 {s["segment"]} 段</h2><p>首次尝试成功 {s["first_successes"]}/{s["first_attempts"]}；新接触：{"、".join(s["new"]) or "无"}；保持：{"、".join(s["keep"]) or "无"}；承重：{"、".join(s["support"]) or "无"}。</p><p>本轮全部已结束尝试：成功 {r["success"]}，物理失败 {r["physical_failure"]}，到达但违反保持/承重 {r["reached_but_keep_violated"]}，超时 {r["timeout"]}。</p>'+videos+'</section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>纯 DAgger · 全部 19 段录像与失败原因</title><style>body{background:#111925;color:#eee;font:16px/1.7 system-ui;margin:24px auto;max-width:1100px;padding:0 16px}section{background:#202d40;padding:18px;margin:20px 0;border-radius:12px}video{width:100%;max-height:540px}a{color:#9cf}select{font:inherit}</style><h1>纯 DAgger：19 段原生执行录像</h1><p>策略和 WM 均冻结。每段从记录的示范状态独立出发，最多 200 步（4 秒），没有绝对倾斜终止。每段展示固定首个环境；若首次尝试有成功/失败两类，再展示一个相反结果。没有挑选最佳动作。</p><p>实心机器人为真实物理执行；蓝色为对应示范，示范结束后停在终点，仅作对照，不参与逐帧判定。橙色为全部新接触点；绿色为 root 终点。接触点误差需结合实际接触是否存在一起看。</p><p>播放速度 <select onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=+this.value)"><option value="0.5">0.5×</option><option value="0.25">0.25×</option><option value="1">1×</option></select> · <a href="report.json">完整诊断数据</a></p>'''+''.join(cards)+'''<script>document.querySelectorAll('video').forEach(v=>v.playbackRate=.5)</script></html>'''
    page = page.replace('若首次尝试有成功/失败两类，再展示一个相反结果', '补充首次尝试中不同成功/失败原因的首个样例')
    page = page.replace('<h1>', '<nav>'+ ' · '.join(f'<a href="#segment-{i}">第 {i} 段</a>' for i in range(1,20))+'</nav><h1>', 1)
    (args.output/'index.html').write_text(page)


if __name__ == '__main__':
    main()
