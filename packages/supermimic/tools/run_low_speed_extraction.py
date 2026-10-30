"""Finish an existing native capture, retry failed low-speed motions, then slice and validate."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2)+'\n'); tmp.replace(path)


def identity(pid, command):
    return dict(pid=pid, start_ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                command=command, cwd=str(runtime_root()))


def alive(record):
    try:
        stat = Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return stat[19] == record['start_ticks'] and stat[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--max-attempts', type=int, default=6)
    a = p.parse_args(); run = a.run.absolute()
    with (run/'runner-process.json').open('x') as handle:
        json.dump(identity(os.getpid(), [sys.executable,*sys.argv]), handle, indent=2)
    first = json.loads((run/'capture_00-process.json').read_text())
    args = first['command']
    selection_path = Path(args[args.index('--selection_manifest')+1])
    selection = json.loads(selection_path.read_text())
    expected = {t['source_task_index'] for t in selection['tasks']}
    status = dict(status='running', stage='capture', selected_motions=len(expected), started_at=time.time())
    def update(**extra):
        status.update(updated_at=time.time(), **extra); write(run/'status.json',status)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
    def launch(name, command):
        log = run/(name+'.log')
        with log.open('xb') as handle:
            proc = subprocess.Popen(command,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        write(run/(name+'-process.json'),dict(**identity(proc.pid,command),log=str(log)))
        while proc.poll() is None:
            update(child_pid=proc.pid); time.sleep(5)
        if proc.returncode:
            raise RuntimeError(f'{name} exited {proc.returncode}; see {log}')
    try:
        update(child_pid=first['pid'])
        while alive(first):
            time.sleep(5)
        successful = set(); captures = []
        for attempt in range(a.max_attempts):
            capture = run/f'capture_{attempt:02d}'
            if attempt:
                remaining = expected-successful
                selected = copy.deepcopy(selection)
                selected['tasks'] = [t for t in selected['tasks'] if t['source_task_index'] in remaining]
                selected['group_counts'] = {g:sum(t['group']==g for t in selected['tasks']) for g in selection['group_counts']}
                retry_path = run/f'retry_selection_{attempt:02d}.json'; write(retry_path,selected)
                command = list(args)
                command[command.index('--selection_manifest')+1] = str(retry_path)
                command[command.index('--output')+1] = str(capture)
                # Nominal independent-process retry; no perturbation/noise or checkpoint change.
                update(stage='retry_capture', attempt=attempt+1, remaining=len(remaining))
                launch(f'capture_{attempt:02d}',command)
            report = json.loads((capture/'library_capture.json').read_text())
            successful.update(r['motion'] for r in report['motions'] if r['successful_rollout'])
            captures.append(capture)
            update(complete_motions=len(successful), remaining=len(expected-successful), attempt=attempt+1)
            if successful == expected:
                break
        if successful != expected:
            raise RuntimeError(f'Capture attempts exhausted; missing original motion IDs: {sorted(expected-successful)}')
        update(stage='segmentation')
        launch('segmentation',[str(ROOT/'tools/run_native.sh'),str(ROOT/'tools/extract_low_speed_contacts.py'),
            '--captures',*map(str,captures),'--selection',str(selection_path),'--output',str(run/'library')])
        update(stage='native_restart_validation')
        launch('native_restart_validation',[str(ROOT/'tools/run_native.sh'),str(ROOT/'tools/validate_contact_slices.py'),
            '--library',str(run/'library'),'--task_manifest',args[args.index('--task_manifest')+1],
            '--output',str(run/'restart_validation'),'--headless'])
        validation = json.loads((run/'restart_validation/report.json').read_text())
        catalogue_path = run/'library/catalogue.json'
        catalogue = json.loads(catalogue_path.read_text())
        catalogue['native_restart_validation'] = dict(report=str(run/'restart_validation/report.json'),
            passed=validation['passed'], failed=validation['failed'], total=validation['total'])
        for case, result in zip(catalogue['cases'],validation['cases']):
            if case['case'] != result['case'] or case['motion'] != result['motion']:
                raise ValueError('Validation case identity mismatch')
            case['native_restart_passed'] = result['native_restart_passed']
        write(catalogue_path,catalogue)
        update(status='complete',stage='complete',cases=validation['total'],
               validated_cases=validation['passed'],unvalidated_cases=validation['failed'])
    except Exception as error:
        update(status='failed',error=str(error)); raise


if __name__ == '__main__':
    main()
