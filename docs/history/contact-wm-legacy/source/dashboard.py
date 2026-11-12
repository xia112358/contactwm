"""Export a small read-only monitoring snapshot to the existing static server."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shutil
import time


def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return default


def alive(record):
    if not record:return False
    try:
        value=Path(f"/proc/{record['pid']}/stat").read_text().split()
        return value[21]==record['start_ticks'] and value[2]!='Z'
    except OSError:return False


def snapshot(root, runs):
    stream=runs/'stream-4096-v2'
    if not stream.exists():stream=runs/'stream-4096-v1'
    current=read_json(stream/'status.json')
    if current:
        metrics=[]
        if (stream/'metrics.jsonl').exists():
            for line in (stream/'metrics.jsonl').read_text().splitlines():
                try:metrics.append(json.loads(line))
                except json.JSONDecodeError:pass
        checkpoints=[dict(step=int(p.stem.split('_')[-1]),kind='train',response=read_json(p)) for p in sorted((stream/'policy').glob('response_*.json'))]
        return dict(updated_at=datetime.now(timezone.utc).isoformat(),total_steps=read_json(stream/'config.json',{}).get('iterations',20000),
            training_status='running' if current['status'] in ('starting','running') and alive(read_json(root/'tmp/contact-wm-stream-4096-process.json')) else 'completed' if current['status']=='completed' else 'stopped',
            evaluation_status='waiting',metrics=metrics,checkpoints=checkpoints,native_rows=[],native_summary=None,
            training_done=current['status']=='completed',evaluation_done=False,stream=current)
    base=runs;train=base/'policy-multitarget-v2';evaluation=base/'multitarget-native-evaluation-v2'
    online_root=base/'online-contact-long-v1';online=read_json(online_root/'status.json')
    metric_dirs=[train]
    if online:
        metric_dirs+=sorted(online_root.glob('round_*/policy'))
        if online.get('active_policy_directory') and Path(online['active_policy_directory']).exists():train=Path(online['active_policy_directory'])
        else:train=Path(online['selected_policy']).parent
        evaluation=Path(online.get('active_evaluation_directory') or online_root/'pending-evaluation')
    metrics=[]
    for metric_dir in metric_dirs:
        try:
            for line in (metric_dir/'metrics.jsonl').read_text().splitlines():
                try:metrics.append(json.loads(line))
                except json.JSONDecodeError:pass
        except OSError:pass
    checkpoints=[]
    for path in sorted(train.glob('response_*.json')):
        item=read_json(path)
        if item:checkpoints.append(dict(step=int(path.stem.split('_')[-1]),kind='train',response=item))
    report=read_json(train/'report.json');native_report=read_json(evaluation/'report.json')
    if report and report.get('heldout_response_after'):
        checkpoints.append(dict(step=report['steps'],kind='heldout',response=report['heldout_response_after']))
    progress=read_json(evaluation/'progress.json',[])
    rows=[]
    for row in progress:
        result=row.get('result',{})
        rows.append(dict(label=row['label'],index=row['index'],error=bool(row.get('error')),
            success=result.get('success'),checks=result.get('checks'),reason=result.get('reason'),
            max_position_error=result.get('max_position_error'),max_slip=result.get('max_slip')))
    train_live=alive(read_json(root/'tmp/contact-wm-policy-multitarget-v2-process.json'))
    eval_live=alive(read_json(root/'tmp/contact-wm-multitarget-native-evaluation-v2-process.json'))
    value=dict(updated_at=datetime.now(timezone.utc).isoformat(),total_steps=400,
        training_status='completed' if report else 'running' if train_live else 'stopped',
        evaluation_status='completed' if native_report else ('running' if progress or list(evaluation.glob('*-process.json')) else 'waiting') if eval_live else 'stopped',
        metrics=metrics,checkpoints=checkpoints,native_rows=rows,
        native_summary={name:native_report.get(name) for name in ('prior','trained')} if native_report else None,
        training_done=bool(report),evaluation_done=bool(native_report))
    if online:
        value.update(total_steps=online['total_policy_iteration'],training_done=online['status']=='completed',
            training_status='completed' if online['status']=='completed' else 'stopped' if online['status']=='failed' else 'running',
            evaluation_status='completed' if native_report else 'running' if online['phase']=='evaluate' else 'waiting',
            online={k:online.get(k) for k in ('status','phase','round','rounds','policy_iteration')})
        current=online_root/f"round_{online['round']-1:03d}"
        value['online']['collected_targets']=len(list(current.glob('native_*/summary.json')))
        wm_report=read_json(current/'wm/report.json')
        if wm_report:value['online']['wm']={k:wm_report.get(k) for k in ('sequence_before','sequence_after','recent_fraction')}
        value['online']['wm_accepted']=online.get('wm_gate',{}).get('accepted')
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True,help='Existing contact-wm run directory')
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name('dashboard.html'),args.output/'index.html')
    shutil.copyfile(Path(__file__).with_name('native_player.html'),args.output/'native.html')
    while True:
        run=args.runs/'stream-4096-v2'
        if not run.exists():run=args.runs/'stream-4096-v1'
        native=run/'native-latest.json'
        if native.exists():
            shutil.copyfile(native,args.output/'native.json.tmp');(args.output/'native.json.tmp').replace(args.output/'native.json')
        elif (args.output/'native.json').exists():(args.output/'native.json').unlink()
        baseline=native.with_name('native-baseline.json')
        if baseline.exists():shutil.copyfile(baseline,args.output/'native-baseline.json')
        elif (args.output/'native-baseline.json').exists():(args.output/'native-baseline.json').unlink()
        value=snapshot(args.root,args.runs)
        value['watch_complete']=(value['online']['status']!='running') if 'online' in value else value['evaluation_done'] or (value['training_status']=='stopped' and value['evaluation_status']=='stopped')
        temp=args.output/'data.json.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False));temp.replace(args.output/'data.json')
        if args.once or value['watch_complete']:break
        time.sleep(5)


if __name__=='__main__':main()
