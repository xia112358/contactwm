"""Compare contact criteria on identical saved first attempts, without simulation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import html
import json
from pathlib import Path
import sys
pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from supermimic.contact_control import ContactClock,assess_region_arrival
    from supermimic.mdp import unflatten_state
    from supermimic.rotations import rot_angle_sq
    torch.set_num_threads(4)
    cfg=json.loads((args.run/'config.json').read_text())
    status=json.loads((args.run/'status.json').read_text())
    assert status['actor_frozen_unchanged'] and status['wm_frozen_unchanged']
    assert cfg.get('contact_stop_mode') is None, 'Use the original replay, before onset truncation changes'
    buf=torch.load(args.run/'replay.pt',map_location='cpu',weights_only=False)
    assert buf['ptr']==buf['size'] and buf['size']<buf['T']
    data={k:v[:buf['size']] for k,v in buf['data'].items()}
    demo=torch.load(cfg['recording'],map_location='cpu',weights_only=False)['data']
    clock=ContactClock(demo,hold=1,role_aware=True)
    parts=['左脚','右脚','左手','右手','左膝','右膝'];reports=[]
    for g in data['goal_index'][0].unique().tolist():
        slots=(data['goal_index'][0]==g).nonzero().flatten();assert len(slots)==1
        slot=int(slots[0]);last=int(data['done'][:,slot].nonzero()[0])+1
        d={k:v[:last,slot] for k,v in data.items()};assert int(d['ep_step'][0])==0
        end=int(clock.event_ends[g]);keep=clock.keep_masks[end-1]
        new=clock.target_masks[end-1]&~keep;target=demo['point'][end-1]
        faces=torch.tensor(cfg['target_faces_by_segment'][g])
        arrival,retained=assess_region_arrival(d,target,faces,new,keep,clock.support_masks[end-1])
        state=unflatten_state(d['s1']);ref=unflatten_state(demo['s1'][end-1])
        root=(state[0]-ref[0]).norm(dim=-1);angle=rot_angle_sq(state[1],ref[1]).sqrt()
        safe=torch.isfinite(d['s1']).all(-1)&(state[0][:,2]>=.15)
        observed=d['known']&d['active'];same=(d['surface_local']==faces)&(faces>=0)
        distance=(d['point']-target).norm(dim=-1)
        valid=observed&same&(distance<=.05)
        accumulated=valid.long().cummax(0).values.bool()
        sequential=((~new)|accumulated).all(-1)
        support=torch.ones_like(safe) if new.any() else retained
        old=arrival&(root<.1)&(angle<.4)&safe&support
        root25=arrival&(root<.25)&(angle<.4)&safe&support
        relaxed=sequential&(root<.25)&(angle<.4)&safe&support
        assert not bool((old&~root25).any()) and not bool((root25&~relaxed).any())
        def first(mask):
            indices=mask.nonzero().flatten()
            return int(indices[0])+1 if len(indices) else None
        details=[];issues=[]
        for i in new.nonzero().flatten().tolist():
            onface=observed[:,i]&same[:,i]
            best=float(distance[onface,i].min())*100 if onface.any() else None
            hit=first(valid[:,i]);details.append(dict(part=parts[i],first_valid_step=hit,best_target_face_error_cm=best))
            if hit is None:
                if not observed[:,i].any():issues.append(parts[i]+'未检测到实际接触')
                elif not onface.any():issues.append(parts[i]+'未接触正确目标面')
                else:issues.append(f'{parts[i]}最近落点仍差 {best:.1f} cm')
        if not relaxed.any() and sequential.any():
            eligible=sequential&safe&support
            if not (eligible&(root<.25)).any():issues.append('接触累计达标后 root 仍超出 25 cm')
            elif not (eligible&(root<.25)&(angle<.4)).any():issues.append('接触累计达标且 root 合格时，姿态仍超出 0.4 rad')
        reports.append(dict(segment=g+1,recorded_steps=last,old_pass=bool(old.any()),root25_pass=bool(root25.any()),relaxed_pass=bool(relaxed.any()),old_success_step=first(old),root25_success_step=first(root25),relaxed_success_step=first(relaxed),parts=details,issues=issues))
    result=dict(run=str(args.run.resolve()),checkpoint=cfg['resume'],policy_updates=status['policy_updates'],scope='Identical saved first attempt per segment; no simulation, no new truncation, no weight updates',criteria=dict(old='same-frame contacts, root <0.10m',root25='same-frame contacts, root <0.25m',relaxed='per-part valid target contacts accumulated through current frame, root <0.25m at success frame',unchanged='actual contact, same target face, distance <=0.05m, orientation <0.4rad, physical safety and original recorded horizon',limitation='No inference beyond the original recorded episode end'),counts={key:sum(r[key] for r in reports) for key in ['old_pass','root25_pass','relaxed_pass']},segments=reports)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    rows=''.join('<tr><td>'+str(r['segment'])+'</td>'+''.join('<td>'+('通过' if r[k] else '失败')+'</td>' for k in ['old_pass','root25_pass','relaxed_pass'])+'<td>'+str(r['recorded_steps'])+'</td><td>'+html.escape('；'.join(r['issues']) or '—')+'</td></tr>' for r in reports)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>同一批轨迹重新计分</title><style>body{font:16px/1.7 system-ui;background:#111925;color:#eee;max-width:1100px;margin:24px auto;padding:16px}td,th{padding:8px;border-bottom:1px solid #567}a{color:#9cf}</style><h1>同一批轨迹重新计分</h1><p>{UPDATES} 次更新的同一份权重、同一批 19 段首次尝试。未重新仿真，未改变结束帧，未更新权重。</p><p>依次只放宽 root 到 25 cm，以及允许各目标部位先后进入正确目标面的 5 cm 区域，达成后在本段内累计。不新增“首次碰触就失败”的判据。姿态仍须小于 0.4 rad；root 和姿态在判成功的当帧检查。</p><p>原记录已结束后的动作无法推断；这里只重新计分已保存的轨迹。</p><p><a href="report.json">详细结果</a></p><table><tr><th>段</th><th>旧标准</th><th>只放宽 root</th><th>再放宽同步</th><th>原记录帧数</th><th>仍未通过原因</th></tr>'''+rows+'</table></html>'
    (args.output/'index.html').write_text(page.replace('{UPDATES}',str(status['policy_updates'])))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
