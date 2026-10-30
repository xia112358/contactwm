"""Run independent same-policy-physics Newton relabel jobs, never training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--stop',type=int,default=28)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--query-nconmax',type=int,default=None)
    parser.add_argument('--query-njmax',type=int,default=None)
    args=parser.parse_args()
    if not 0 <= args.start < args.stop <= 28 or not 1 <= args.workers <= 2:
        raise ValueError('Expected original28 range and at most two concurrent GPU query workers')
    args.output.mkdir(parents=True,exist_ok=True)
    def run(i):
        name=f'climb_{i:02d}'
        motion=Path(f'runtime/current/motions/{name}_z_scale_1.0.npz')
        labels=args.output/f'{name}_z_scale_1.0.npz'
        from somaforge_core.newton_contact_data import load_contact_labels
        if labels.exists():
            verified=load_contact_labels(motion,labels)
            return dict(motion=name,status='already_verified',frames=len(verified['contact_part_mask']))
        work=args.output/name
        # Retain every failed attempt; never overwrite its diagnostics.
        attempt=0
        while (attempt_dir:=work/f'attempt_{attempt:03d}').exists():
            attempt+=1
        attempt_dir.mkdir(parents=True)
        command=[sys.executable,'scripts/serve_newton_contact_queries.py',
            '--checkpoint',str(args.checkpoint),'--motion-manifest',
            f'runtime/current/rollout/newton_contact_force/manifests/{name}_manifest.json',
            '--binding',str(attempt_dir/'binding.json'),'--create-native-binding',
            '--inspection-output',str(attempt_dir/'model.json'),'--relabel-motion',str(motion),
            '--labels-output',str(labels),'--device','cuda:0']
        if args.query_nconmax is not None:
            command.extend(['--query-nconmax',str(args.query_nconmax)])
        if args.query_njmax is not None:
            command.extend(['--query-njmax',str(args.query_njmax)])
        with (attempt_dir/'worker.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        # Kit can swallow process exit errors; only validated artifacts count.
        try:
            verified=load_contact_labels(motion,labels)
            return dict(motion=name,status='verified',frames=len(verified['contact_part_mask']),
                        active_part_frames=int(verified['contact_part_mask'].sum()),log=str(attempt_dir/'worker.log'))
        except Exception as exc:
            return dict(motion=name,status='failed',error=str(exc),exit_code=result.returncode,log=str(attempt_dir/'worker.log'))
    reports=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,i) for i in range(args.start,args.stop)]):
            report=future.result();reports.append(report);print(json.dumps(report),flush=True)
    if any(r['status']=='failed' for r in reports):
        raise SystemExit(1)


if __name__=='__main__':
    main()
