"""Finite alternating native collection / WM sequence fit / goal-policy BPTT.

Uses SuperMimic's replay-and-alternate training pattern with the direct PHP
contact policy. It does not import the velocity encoder or motion-goal policy.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import torch
from holosoma.contact_wm.learn_loop import candidate_gate


@dataclass(frozen=True)
class Options:
    policy:Path
    wm:Path
    recording:Path
    atlas:Path
    output_dir:Path
    replay:tuple[Path,...]=()
    rounds:int=20
    policy_steps:int=200
    wm_steps:int=200
    wm_window:int=5
    noise:float=.02
    workers:int=3
    evaluate_every:int=5
    max_wm_regression:float=.2
    device:str='cuda:0'
    native_launcher:Path=Path('tmp/run_contact_wm_native.py')


def main():
    import tyro
    o=tyro.cli(Options)
    if min(o.rounds,o.policy_steps,o.wm_steps,o.wm_window,o.workers,o.evaluate_every)<1:raise ValueError('Positive loop settings required')
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    policy=o.policy.resolve();wm=o.wm.resolve();replay=list(o.replay)
    initial=torch.load(policy,map_location='cpu',weights_only=False)
    initial_iteration=initial['iteration'];iteration=initial_iteration
    state=dict(status='running',phase='starting',round=0,rounds=o.rounds,policy_iteration=iteration,
        total_policy_iteration=iteration+o.rounds*o.policy_steps,selected_policy=str(policy),selected_wm=str(wm),
        completed_rounds=[],options={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,tuple) else v for k,v in asdict(o).items()})
    def save():
        p=out/'status.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
    def run(command,log,required):
        with log.open('xb') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            log.with_suffix('.process.json').write_text(json.dumps(dict(pid=child.pid,
                start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=command,cwd=str(Path.cwd()),log=str(log)),indent=2))
            code=child.wait()
        if code or not required.exists():raise RuntimeError(f'Child failed or missing result: {log}')
    save()
    try:
        for round_id in range(o.rounds):
            directory=out/f'round_{round_id:03d}';directory.mkdir()
            state.update(round=round_id+1,phase='collect',active_policy_directory=None);save()
            current=torch.load(policy,map_location='cpu',weights_only=False)
            offsets=[(x,y,0.) for x in (-.04,0.,.04) for y in (-.04,0.,.04)]
            def collect(index):
                destination=directory/f'native_{index:02d}'
                run([sys.executable,str(o.native_launcher),'holosoma.contact_wm.collect',
                    '--checkpoint',current['prior']['source'],'--contact-policy',str(policy),
                    '--output-dir',str(destination),'--num-envs','1','--matched-motion',str(current['motion_index']),
                    '--max-steps',str(5+current['horizon']),'--task-initial-recording',str(o.recording),
                    '--support-contract','sustained_contact','--action-noise-std',str(o.noise),
                    '--noise-seed',str(1000+round_id*9+index),'--contact-goal-offset',*map(str,offsets[index])],
                    directory/f'native_{index:02d}.log',destination/'summary.json')
                return destination
            with ThreadPoolExecutor(max_workers=o.workers) as pool:new_data=list(pool.map(collect,range(9)))
            replay.extend(new_data);state.update(phase='wm',new_recordings=[str(p) for p in new_data]);save()
            fit=directory/'wm'
            run([sys.executable,'-m','holosoma.contact_wm.calibrate','--recording',str(o.recording.parent),
                '--resume',str(wm),'--output-dir',str(fit),'--steps',str(o.wm_steps),'--device',o.device,
                '--wm-window',str(o.wm_window),'--additional-recordings',*map(str,replay),
                '--recent-recordings',*map(str,new_data)],directory/'wm.log',fit/'report.json')
            report=json.loads((fit/'report.json').read_text());gate=candidate_gate(report,o.max_wm_regression)
            before=report['sequence_before'];after=report['sequence_after']
            if after['final_joint_rmse_rad']>before['final_joint_rmse_rad']*(1+o.max_wm_regression):
                gate['accepted']=False;gate['failures'].append(dict(metric='multi_step_joint',before=before,after=after))
            if gate['accepted']:wm=fit/'model.pt'
            train=directory/'policy';state.update(phase='policy',selected_wm=str(wm),wm_gate=gate,active_policy_directory=str(train));save()
            run([sys.executable,'-m','holosoma.contact_wm.train_policy','--recording',str(o.recording),
                '--atlas',str(o.atlas),'--wm-checkpoint',str(wm),'--output-dir',str(train),
                '--resume',str(policy),'--steps',str(o.policy_steps),'--device',o.device,
                '--multi-target','--goal-offset','.05','0','0','--target-span','.04','--arrival-delay','5',
                '--save-every','100','--learning-rate','.000001','--target-learning-rate-scale','250',
                '--response-weight','20'],directory/'policy.log',train/'report.json')
            iteration+=o.policy_steps;policy=train/f'policy_{iteration:04d}.pt'
            state.update(policy_iteration=iteration,selected_policy=str(policy));save()
            if (round_id+1)%o.evaluate_every==0 or round_id+1==o.rounds:
                evaluation=directory/'evaluation';state.update(phase='evaluate',active_evaluation_directory=str(evaluation));save()
                run([sys.executable,'-m','holosoma.contact_wm.evaluate_targets','--checkpoint',str(policy),
                    '--prior-checkpoint',str(o.policy.parent/'policy_0000.pt'),'--initial-recording',str(o.recording),
                    '--output-dir',str(evaluation),'--workers',str(o.workers)],directory/'evaluation.log',evaluation/'report.json')
            state['completed_rounds'].append(dict(round=round_id+1,policy=str(policy),wm=str(wm),wm_gate=gate))
            save();print(json.dumps(dict(round=round_id+1,policy_iteration=iteration,wm_accepted=gate['accepted'])),flush=True)
        state.update(status='completed',phase='completed');save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
