"""Publish this run's read-only dashboard beneath the existing static server."""
import argparse
import json
from pathlib import Path
import re
import shutil
import time

from monitor_training import rows,read_json,tail


def alive(record):
    try:
        proc=Path('/proc')/str(record['pid'])
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return stat[19]==record['proc_start_ticks'] and stat[0]!='Z'
    except (OSError,KeyError,ValueError):return False


def snapshot(run):
    plan=read_json(run/'plan.json');flow=read_json(run/'status.json');tracking=run/'tracking'
    train=rows(tracking/'train_log.csv');latest=train[-1] if train else {}
    owner=read_json(tracking/'process.json');live=alive(owner)
    failure=(tracking/'invalid_transition.pt').exists()
    total=plan.get('tracking',{}).get('iterations',24000)
    state=('已按要求停止' if flow.get('status')=='stopped' else '数值异常停机' if failure else '训练中' if live else 'tracking 已完成'
           if latest.get('iteration',0)>=total else '训练进程已退出')
    evaluations=[]
    for path in sorted([*run.glob('*_eval.json'),*tracking.glob('ckpt_*_eval.json')]):
        match=re.search(r'(\d{6})_eval.json$',path.name)
        detail=read_json(path)
        if not match or not detail.get('per_clip'):continue
        n=len(detail['per_clip']);evaluations.append(dict(iteration=int(match.group(1)),total=n,
            passed=n-int(detail['n_failed']),norm_return=detail['norm_return'],
            mean_length_ratio=detail['mean_length_ratio'],groups=detail.get('groups',{}),evaluated_at=path.stat().st_mtime))
    args=read_json(tracking/'args.json');manifest=read_json(Path(plan.get('task_manifest','missing')))
    log=tracking/'train_log.csv'
    catalogue=read_json(run/'whole_transfers/catalogue.json')
    return dict(run=run.name,updated=time.time(),state=state,alive=live,failure=failure,
        total=total,num_envs=plan.get('num_envs'),collision_profile=args.get('collision_profile'),
        contact_model=args.get('contact_model'),case_count=len(manifest.get('tasks',[])),gpu=plan.get('gpu'),
        whole_transfer_count=catalogue.get('whole_transfer_count'),whole_transfer_motions=catalogue.get('motions'),
        whole_transfer_gate=plan.get('whole_transfer_coverage',{}),
        tracking_only=plan.get('tracking_only',False),eval_every=plan.get('evaluation_interval',2000),
        dataset_label=plan.get('dataset_label','lowclimb'),probe_envs=plan.get('statistics_only_probes',0),
        sampler_stats=read_json(tracking/'sampler_stats.json'),
        group_counts=manifest.get('group_counts',{}),
        phase=flow.get('stage'),flow_status=flow.get('status'),flow_error=flow.get('error'),
        latest=latest,train=train,evaluations=sorted(evaluations,key=lambda x:x['iteration']),
        eval_alive=alive(read_json(run/'eval-monitor-process.json')),
        checkpoints=[p.name for p in sorted(tracking.glob('ckpt_*.pt'))[-6:]],
        log_age=time.time()-log.stat().st_mtime if log.exists() else None,
        log=tail(Path(owner.get('log',str(tracking/'stdout-retry1.log')))),pipeline_log=tail(Path(plan.get('pipeline_log',str(run/'flow.log')))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval',type=float,default=10.);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name('surface_flow_monitor.html'),a.output/'index.html')
    while True:
        data=snapshot(a.run)
        temp=a.output/'status.tmp';temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
        temp.replace(a.output/'status.json')
        if data['flow_status'] in ('complete','failed','stopped') and not data['alive']:break
        time.sleep(a.interval)


if __name__=='__main__':main()
