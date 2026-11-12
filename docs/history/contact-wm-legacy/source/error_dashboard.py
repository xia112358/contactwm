"""Read-only contact error curves exported to the existing loopback web server."""
import argparse
import json
from pathlib import Path
import shutil
import time


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def rows(path):
    try:lines=path.read_text().splitlines()
    except OSError:return []
    result=[]
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:pass  # Writer may still be appending its final line.
    return result


def lineage(run):
    """Include earlier segments of a resumed run, without mixing other trials."""
    chain=[];seen=set();current=run
    while current.resolve() not in seen:
        seen.add(current.resolve());chain.append(current)
        resume=read(current/'config.json',{}).get('resume')
        if not resume:break
        previous=Path(resume).parent
        if previous.parent.resolve()!=run.parent.resolve() or not (previous/'status.json').is_file():break
        current=previous
    return list(reversed(chain))


def snapshot(run):
    status=read(run/'status.json',{})
    if run.name.startswith('frozen-policy-long-'):
        metrics=rows(run/'metrics.jsonl')
        status={**status,'iteration':status.get('step',0),'policy_updates':status.get('step',0)}
        def point(step,**values):return dict(iteration=step,updates=step,**values)
        native=[]
        for value in rows(run/'native-results.jsonl'):
            step=value['step'];episodes=rows(run/f'native-{step:04d}'/'native-metrics.jsonl')
            results=[r for episode in episodes for r in episode.get('results',[])]
            native.append(point(step,error=value.get('mean_contact_error_m'),
                success=sum(value.get('successes',[])),total=sum(e.get('total',0) for e in episodes),
                complete_contacts=sum(r.get('observed_hold_steps',0)==10 for r in results),
                per_target=[dict(index=r['index'],error=r.get('max_position_error'),hold=r.get('observed_hold_steps',0),success=r.get('success',False))
                    for r in (episodes[-1]['results'] if episodes else [])]))
        return dict(name=run.name,status=status,
            wm=[point(m['step'],error=m['train'].get('mean_endpoint_error_m')) for m in metrics],native=native,
            losses=[point(m['step'],**{k:m['train'].get(k) for k in ('geometry','target_geometry','support_geometry')}) for m in metrics])
    segments=lineage(run)
    metrics=[row for segment in segments for row in rows(segment/'metrics.jsonl')]
    updates={0:0,**{m['iteration']:m.get('policy_updates',0) for m in metrics}}
    def point(iteration,**values):return dict(iteration=iteration,updates=updates.get(iteration),**values)
    wm=[]
    for path in [p for segment in segments for p in sorted((segment/'policy').glob('response_*.json'))]:
        value=read(path)
        if value:
            wm.append(point(int(path.stem.split('_')[-1]),error=value.get('mean_target_error_m')))
    native={}
    for value in [row for segment in segments for row in rows(segment/'native-metrics.jsonl')]:
        results=value.get('results',[]);n=value.get('total',len(results))
        native[value['iteration']]=point(value['iteration'],error=value.get('mean_contact_error_m'),
            success=value.get('successes',0),total=n,
            complete_contacts=sum(r.get('observed_hold_steps',0)==10 for r in results),
            worst=max((r['max_position_error'] for r in results if r.get('max_position_error') is not None),default=None),
            per_target=[dict(index=r['index'],error=r.get('max_position_error'),hold=r.get('observed_hold_steps',0),success=r.get('success',False)) for r in results])
    return dict(name=run.name,status=status,wm=list({p['iteration']:p for p in wm}.values()),native=sorted(native.values(),key=lambda x:x['iteration']),
        losses=[point(m['iteration'],geometry=m['geometry'],target_geometry=m.get('target_geometry'),support_geometry=m.get('support_geometry')) for m in metrics if m.get('geometry') is not None])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__).with_name('error_dashboard.html'),args.output/'index.html')
    while True:
        runs=sorted((p for pattern in ('height-contact-*','frozen-policy-long-*') for p in args.runs.glob(pattern)
            if (p/'status.json').is_file()),key=lambda p:(p/'status.json').stat().st_mtime)
        data=dict(exported_at=time.time(),runs=[snapshot(p) for p in runs])
        temp=args.output/'data.json.tmp';temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False));temp.replace(args.output/'data.json')
        if args.once:break
        time.sleep(3)


if __name__=='__main__':main()
