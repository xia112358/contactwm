"""Recheck recorded attempts under a completion profile without rerunning physics."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--criteria',type=Path,default=ROOT/'experiments/contact_completion.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample_world',type=int)
    args=parser.parse_args()
    import torch
    from supermimic.contact_control import ContactClock,goal_contact_mask,load_contact_tolerances
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import rot_angle_sq
    torch.set_num_threads(4)
    cfg=json.loads((args.run/'config.json').read_text())
    assert cfg['completion_mode']=='accumulated_target' and not cfg['contact_face_matching']
    rec=torch.load(cfg['recording'],map_location='cpu',weights_only=False)
    clock=ContactClock(rec['data'],hold=1,role_aware=True)
    old_radii=cfg.get('contact_tolerances_m',[.05]*len(clock.event_ends))
    new_radii=load_contact_tolerances(clock,rec['metadata'],args.criteria)
    buf=torch.load(args.run/'replay.pt',map_location='cpu',weights_only=False)
    rows=(buf['ptr']-buf['size']+torch.arange(buf['size']))%buf['T']
    d={k:v[rows] for k,v in buf['data'].items()};del buf
    starts=(d['ep_step']==0).all(-1).nonzero().flatten().tolist()
    blocks=[(s,e) for s,e in zip(starts,starts[1:]+[len(rows)]) if d['done'][s:e].any(0).all()]
    assert blocks
    s,e=blocks[-1];d={k:v[s:e] for k,v in d.items()}
    event=d['goal_index'][0];end=clock.event_ends[event]
    new=clock.target_masks[end-1]&~clock.keep_masks[end-1]
    target=rec['data']['point'][end-1]
    state=unflatten_state(d['s1']);reference=unflatten_state(rec['data']['s1'][end-1])
    root=(state[0]-reference[0]).norm(dim=-1)
    angle=rot_angle_sq(state[1],reference[1]).sqrt()
    physical=~torch.isfinite(d['s1']).all(-1)|(state[0][...,2]<.15)
    stop=d['done'].long().argmax(0);batch=torch.arange(len(stop))
    live=torch.arange(len(d['done']))[:,None]<=stop
    observed=d['known']&d['active']

    def recheck(radii):
        radius=torch.tensor(radii)[event,None]
        hits=goal_contact_mask(d,target,tolerance=radius)&new&live[...,None]
        accumulated=hits.long().cummax(0).values.bool()
        arrived=((~new)|accumulated).all(-1)
        # Preserve the original retained-contact requirement for stages with no new goal.
        kept=((~clock.keep_masks[end-1])|observed).all(-1)
        completed=arrived&(root<cfg['root_tolerance'])&(angle<.4)&~physical&live
        completed &= torch.where(new.any(-1),True,kept)
        success=completed.any(0)
        first=torch.where(success,completed.long().argmax(0),stop)
        return success,first,hits

    before,old_first,_=recheck(old_radii)
    after,new_first,hits=recheck(new_radii)
    assert torch.equal(before,~d['term'][stop,batch]),'Old-rule reconstruction disagrees with simulator records'
    assert torch.equal(old_first[before],stop[before]),'Original attempts continued after completion'
    assert all(a>=b for a,b in zip(new_radii,old_radii)), 'This audit expects relaxation only'
    assert not bool((before&~after).any())
    segments=[]
    for g in event.unique().tolist():
        mask=event==g
        if new_radii[g]==old_radii[g]:
            assert torch.equal(before[mask],after[mask]) and torch.equal(old_first[mask],new_first[mask])
        segments.append(dict(segment=g+1,old_radius_cm=old_radii[g]*100,new_radius_cm=new_radii[g]*100,
                             attempts=int(mask.sum()),old_successes=int(before[mask].sum()),new_successes=int(after[mask].sum()),
                             newly_passing=int((~before&after&mask).sum()),remaining_failures=int((~after&mask).sum())))
    sample=None
    if args.sample_world is not None:
        i=args.sample_world;assert 0<=i<len(stop)
        first_hits={str(p):int(hits[:,i,p].long().argmax())+1 if bool(hits[:,i,p].any()) else None
                    for p in new[i].nonzero().flatten().tolist()}
        sample=dict(world=i,segment=int(event[i])+1,old_success=bool(before[i]),new_success=bool(after[i]),
                    old_stop_step=int(stop[i])+1,new_completion_step=int(new_first[i])+1 if after[i] else None,
                    first_valid_steps_by_part=first_hits,root_cm=float(root[new_first[i],i])*100,
                    orientation_rad=float(angle[new_first[i],i]))
    report=dict(run=str(args.run.resolve()),criteria_profile=str(args.criteria.resolve()),
                scope='Counterfactual completion on the same saved first attempts, bounded by original termination. No new simulation, loss recomputation or model update.',
                old_contact_tolerances_m=old_radii,new_contact_tolerances_m=new_radii,
                contact_loss_scale_m=.05,loss_formula_changed=False,root_tolerance_cm=cfg['root_tolerance']*100,
                orientation_tolerance_rad=.4,grace_steps=cfg['segment_grace_steps'],segments=segments,sample=sample,
                checks=dict(original_decisions_match=True,other_segments_unchanged=True,no_previous_success_lost=True))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    changed=[r for r in segments if r['old_radius_cm']!=r['new_radius_cm']]
    text=''.join(f'<p>第 {r["segment"]} 段：{r["old_radius_cm"]:g} → {r["new_radius_cm"]:g} cm；同一批 {r["attempts"]} 次尝试，通过数从 <b>{r["old_successes"]}</b> 变为 <b>{r["new_successes"]}</b>，仍失败 {r["remaining_failures"]} 次。</p>' for r in changed)
    sample_text=(f'<p>之前录像中的环境 {sample["world"]}：旧规则在第 {sample["old_stop_step"]} 帧超时；按新规则，第 <b>{sample["new_completion_step"]}</b> 帧即可完成。</p>'
                 if sample and sample['new_success'] and not sample['old_success'] else '')
    rows=''.join(f'<tr><td>{r["segment"]}</td><td>{r["old_radius_cm"]:g} → {r["new_radius_cm"]:g} cm</td><td>{r["old_successes"]}/{r["attempts"]}</td><td>{r["new_successes"]}/{r["attempts"]}</td></tr>' for r in segments)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>落地段容差调整</title><style>body{{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1000px;margin:24px auto;padding:0 18px}}a{{color:#9cf}}td,th{{padding:5px 16px;border-bottom:1px solid #456}}</style><h1>第12段落地：接触位置容差放宽到8cm</h1>{text}{sample_text}<p>其余段仍为5cm；root仍为25cm，姿态仍为0.4rad，截止时间仍为示范长度+10帧。两个接触可以先后达成。连续接触距离loss及其缩放没有修改。</p><p>这是同一批已保存轨迹的新旧判据对照，不代表重新训练产生了改善。历史训练指标和旧录像标签保留原始5cm含义。</p><p><a href="../segment12-late-training/">原失败录像</a> · <a href="../tracking-segment12-continuation/">Tracking后续</a> · <a href="report.json">详细核对结果</a></p><table><tr><th>段</th><th>旧→新容差</th><th>旧规则通过</th><th>新规则通过</th></tr>{rows}</table></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps(dict(changed=changed,sample=sample,checks=report['checks']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
