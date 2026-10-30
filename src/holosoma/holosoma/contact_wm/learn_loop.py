"""Finite native-data aggregation loop for the inherited contact world model.

The policy remains frozen. Candidate WMs are retained even when held-out error
regresses; only candidates passing the error gate initialize the next round.
This gate does not certify policy gradients or physical contact success.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys


@dataclass(frozen=True)
class Options:
    policy_checkpoint:Path
    wm_checkpoint:Path
    recording:Path
    output_dir:Path
    additional_recordings:tuple[Path,...]=()
    native_launcher:Path=Path('tmp/run_contact_wm_native.py')
    rounds:int=2
    updates_per_round:int=400
    noise_std:float=.03
    seed:int=37
    max_regression:float=.10
    device:str='cuda:0'


def candidate_gate(report,tolerance):
    """Every held-out trajectory must pass; averages cannot hide regressions."""
    if tolerance<0 or not report['before_by_recording']:raise ValueError('Need held-out trajectories and nonnegative tolerance')
    failures=[]
    for name,before in report['before_by_recording'].items():
        after=report['after_by_recording'][name]
        for subset in ('all','switch'):
            if before[subset] is None:continue
            for metric in ('joint_rmse_rad','force_rmse_N'):
                b=before[subset][metric];a=after[subset][metric]
                if not (0<=a<=max(b,1e-9)*(1+tolerance)):
                    failures.append(dict(recording=name,subset=subset,metric=metric,before=b,after=a))
    return dict(accepted=not failures,failures=failures,
                meaning='prediction-error gate only; policy gradient validity not certified')


def main():
    import tyro
    o=tyro.cli(Options)
    if o.rounds<1 or o.updates_per_round<1 or o.max_regression<0:raise ValueError('Invalid loop limits')
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    extra=list(o.additional_recordings);wm=o.wm_checkpoint.resolve();rounds=[]
    def run(command,log,required):
        with log.open('xb') as stream:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            identity=dict(pid=process.pid,command=command,log=str(log),cwd=str(Path.cwd()),
                          start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21])
            (out/'active-process.json').write_text(json.dumps(identity,indent=2))
            code=process.wait()
        if code or not required.is_file():raise RuntimeError(f'Child failed or missing artifact: {log}')
    for index in range(o.rounds):
        directory=out/f'round_{index:02d}';directory.mkdir();data=directory/'native';fit=directory/'wm'
        run([sys.executable,str(o.native_launcher),'holosoma.contact_wm.collect',
             '--checkpoint',str(o.policy_checkpoint),'--output-dir',str(data),'--num-envs','8',
             '--action-noise-std',str(o.noise_std),'--noise-seed',str(o.seed+index)],
            directory/'native.log',data/'summary.json')
        extra.append(data)
        run([sys.executable,'-m','holosoma.contact_wm.calibrate','--recording',str(o.recording),
             '--resume',str(wm),'--output-dir',str(fit),'--steps',str(o.updates_per_round),
             '--device',o.device,'--additional-recordings',*map(str,extra)],
            directory/'wm.log',fit/'report.json')
        report=json.loads((fit/'report.json').read_text());gate=candidate_gate(report,o.max_regression)
        if gate['accepted']:wm=fit/'model.pt'
        rounds.append(dict(round=index,recording=str(data),candidate=str(fit/'model.pt'),gate=gate))
        status=dict(completed_rounds=rounds,selected_wm=str(wm),policy_updated=False,
                    simulator_gradient_certified=False,remaining_rounds=o.rounds-index-1)
        temp=out/'status.json.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json')
        print(json.dumps(status),flush=True)


if __name__=='__main__':main()
