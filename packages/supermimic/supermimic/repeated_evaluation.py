"""Independent-process nominal tracking repeats; CPU-only checkpoint watcher."""
from pathlib import Path
import argparse
import json
import signal
import statistics
import subprocess
import sys
import time


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def summarize(results, seeds):
    names = list(results[0]['per_clip_lengths'])
    if len(results) != len(seeds) or any(set(r['per_clip_lengths']) != set(names) for r in results):
        raise ValueError('Repeat count or motion coverage differs')
    count = len(results)
    def stats(values):
        return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else 0.,
                    minimum=min(values), maximum=max(values), values=values)
    passed = stats([len(names)-r['n_failed'] for r in results])
    summary = dict(repeats=count, seeds=seeds, protocol='independent processes; nominal evaluation without added noise',
        passed_stats=passed, n_failed=len(names)-passed['mean'],
        norm_return_stats=stats([r['norm_return'] for r in results]), per_clip_lengths={})
    for field in ('norm_return', 'norm_return_framew', 'mean_length_ratio'):
        summary[field] = statistics.mean(r[field] for r in results)
    for name in names:
        rows = [r['per_clip_lengths'][name] for r in results]
        successes = sum(not r['failed'] for r in rows)
        item = {k: statistics.mean(r[k] for r in rows)
                for k in ('steps','seconds','reference_steps','reference_seconds','length_ratio','norm_return')}
        item.update(successes=successes, repeats=count, pass_rate=successes/count, failed=successes<count,
                    length_ratio_stats=stats([r['length_ratio'] for r in rows]))
        summary['per_clip_lengths'][name] = item
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_dir', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--iteration_interval', type=int, default=1000)
    p.add_argument('--stop_after', type=int, default=20000)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or a.iteration_interval < 1:
        p.error('Use distinct seeds and a positive checkpoint interval')
    from .evaluation_queue import pending_checkpoints
    folder = a.run_dir.resolve(); output = folder.parent/'repeated_eval'; output.mkdir(exist_ok=True)
    def stop(*_): raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        if (folder/'invalid_transition.pt').exists() or (folder/'STOPPED.json').exists(): return
        paths = pending_checkpoints(folder, set(), iteration_interval=a.iteration_interval, stop_after=a.stop_after)
        for checkpoint in paths:
            iteration = int(Path(checkpoint).stem[5:]); dest = output/f'ckpt_{iteration:06d}'; dest.mkdir(exist_ok=True)
            summary_path = dest/'summary.json'
            if summary_path.exists() and json.loads(summary_path.read_text())['seeds']==a.seeds: continue
            results = []
            for seed in a.seeds:
                base = dest/f'seed_{seed}'; result_path = dest/f'seed_{seed}_eval.json'
                if not result_path.exists():
                    cmd = [sys.executable, '-u', '-m', 'supermimic.evaluate', '--run_dir', str(folder),
                           '--ckpt', checkpoint, '--eval_seed', str(seed), '--result_base', str(base)]
                    print(f'[repeat] checkpoint {iteration}, seed {seed}', flush=True)
                    with (dest/f'seed_{seed}.log').open('w') as log:
                        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        write_json(output/'active.json', dict(iteration=iteration, seed=seed, pid=proc.pid,
                            start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19], command=cmd))
                        try:
                            if proc.wait() != 0: raise RuntimeError(f'Evaluation failed; see {dest}/seed_{seed}.log')
                        finally:
                            if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
                results.append(json.loads(result_path.read_text()))
            summary = summarize(results, a.seeds); summary['iteration'] = iteration
            write_json(summary_path, summary)
            print(f'[repeat] {iteration}: passed {summary["passed_stats"]}', flush=True)
        if not a.watch or (output/f'ckpt_{a.stop_after:06d}'/'summary.json').exists(): return
        time.sleep(30)


if __name__ == '__main__': main()
