"""Finite matched-initial-state native target sweep, including prior control.

Can wait for a specific training process identity and its completion report.
Never certifies target response when initial states differ or episodes fail.
"""
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import subprocess
import sys
import time
import numpy as np
import torch


@dataclass(frozen=True)
class Options:
    checkpoint:Path
    prior_checkpoint:Path
    initial_recording:Path
    output_dir:Path
    training_process:Path|None=None
    native_launcher:Path=Path('tmp/run_contact_wm_native.py')
    span:float=.02
    workers:int=2


def main():
    import tyro
    o=tyro.cli(Options);out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    if o.training_process:
        process=json.loads(o.training_process.read_text())
        while not (o.checkpoint.parent/'report.json').is_file():
            stat=Path(f"/proc/{process['pid']}/stat")
            if not stat.exists() or stat.read_text().split()[21]!=process['start_ticks']:
                raise RuntimeError('Training ended without a completion report')
            time.sleep(5)
    offsets=[(0.,0.,0.),(-o.span,0.,0.),(o.span,0.,0.),(0.,-o.span,0.),(0.,o.span,0.)]
    def execute(label,checkpoint,index):
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        directory=out/f'{label}_{index}';log=out/f'{label}_{index}.log'
        command=[sys.executable,str(o.native_launcher),'holosoma.contact_wm.collect',
            '--checkpoint',c['prior']['source'],'--contact-policy',str(checkpoint),
            '--output-dir',str(directory),'--num-envs','1','--matched-motion',str(c['motion_index']),
            '--max-steps',str(5+c['horizon']),'--task-initial-recording',str(o.initial_recording),
            '--support-contract','sustained_contact','--contact-goal-offset',*map(str,offsets[index])]
        with log.open('xb') as stream:
            p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            (out/f'{label}_{index}-process.json').write_text(json.dumps(dict(pid=p.pid,
                start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],command=command,log=str(log)),indent=2))
            code=p.wait()
        if code or not (directory/'summary.json').exists():return dict(label=label,index=index,error=str(log))
        recording=directory/f"motion_{c['motion_index']:03d}"
        score=subprocess.run([sys.executable,'-m','holosoma.contact_wm.evaluate_policy',
            '--recording',str(recording),'--output',str(directory/'contact-result.json')],capture_output=True,text=True)
        if score.returncode:return dict(label=label,index=index,error=score.stderr)
        return dict(label=label,index=index,recording=str(recording),result=json.loads((directory/'contact-result.json').read_text()))
    results=[]
    with ThreadPoolExecutor(max_workers=o.workers) as pool:
        futures=[pool.submit(execute,label,checkpoint,i) for label,checkpoint in
                 [('prior',o.prior_checkpoint),('trained',o.checkpoint)] for i in range(5)]
        for f in as_completed(futures):
            results.append(f.result());temp=out/'progress.json.tmp';temp.write_text(json.dumps(results,indent=2));temp.replace(out/'progress.json')
            print(json.dumps(dict(completed=len(results),total=10,last=results[-1])),flush=True)
    report=dict(results=results,goal_offsets=offsets,oracle_terrain=True,
                evidence='native execution; identical requested state and five replayed warmup actions')
    for label in ('prior','trained'):
        rows=sorted([r for r in results if r['label']==label],key=lambda r:r['index'])
        if any('error' in row for row in rows):report[label]=dict(complete=False);continue
        initial=[np.load(Path(r['recording']).parent/'task-initial-state.npz') for r in rows]
        starts=[r['state'][0] for r in initial]
        state_error=float(np.max(np.abs(np.stack(starts)-starts[0])))
        force_error=float(max(np.abs(r['previous_force_w']-initial[0]['previous_force_w']).max() for r in initial))
        contacts_match=all(np.array_equal(r[k],initial[0][k]) for r in initial for k in ('active','known'))
        matched=state_error<=1e-4 and force_error<=.1 and contacts_match
        endpoints=[];targets=[];complete=True
        for row in rows:
            directory=Path(row['recording']);m=json.loads((directory/'report.json').read_text());z=np.load(directory/'transitions.npz')
            start=m['execution_start'];g=m['execution_goal'];end=start+m['execution_horizon'];arrival=start+g['arrival_step']-1
            part=int(np.flatnonzero(np.array(g['establish'])[0])[0])
            if len(z['state'])<end or not (z['active'][arrival:end,part]&z['known'][arrival:end,part]).all():complete=False;break
            endpoints.append(z['solver_point_w'][arrival:end,part].mean(0));targets.append(np.array(g['position_w'])[0,part])
        item=dict(complete=complete,initial_state_max_difference=state_error,initial_force_max_difference_N=force_error,
                  initial_contacts_match=contacts_match,matched_initial_state=matched,
                  successes=sum(r['result']['success'] for r in rows),trials=5)
        if complete:
            endpoints=np.array(endpoints);targets=np.array(targets);x=(targets-targets[0])[:,:2];y=(endpoints-endpoints[0])[:,:2]
            item.update(diagnostic_target_gain_xy=np.linalg.lstsq(x,y,rcond=None)[0].T.tolist(),
                        endpoints=endpoints.tolist(),targets=targets.tolist(),
                        response_error_m=float(np.linalg.norm(y-x,axis=-1)[1:].mean()),
                        causal_response_valid=matched)
        report[label]=item
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
