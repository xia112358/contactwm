"""Authorized staged surface-v2 training, with explicit gates and run ownership.

Adopts the already launched fresh tracking process; trains DAgger and independent
segments only after a newly collected, successful surface-native demonstration.
This runner does not stop any process, restart across reboots, or auto-relabel data.
"""
from __future__ import annotations
from supermimic.paths import runtime_root, workspace_root
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)


def identity(pid):
    p=Path('/proc')/str(pid)
    stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,proc_start_ticks=stat[19],state=stat[0],
        command=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0'))


def same_live_process(record):
    try:current=identity(record['pid'])
    except (OSError,ValueError):return False
    return current['proc_start_ticks']==record['proc_start_ticks'] and current['state']!='Z'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();plan=json.loads((run/'plan.json').read_text())
    import torch
    pass
    spec=json.loads(Path(plan['task_manifest']).read_text())
    pass
    from somaforge_core.robot_assets import validate_g1_asset_metadata
    from somaforge_core.collision_profiles import require_profile
    manifest_hash=hashlib.sha256(Path(plan['task_manifest']).read_bytes()).hexdigest()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(plan['gpu']),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',TORCHINDUCTOR_COMPILE_THREADS='4')
    env['SOMAFORGE_ROOT']=str(workspace_root())
    source_files=[ROOT/'supermimic'/name for name in ('train.py','lowstep.py','networks.py','contact_collect.py',
        'contact_initialize.py','contact_dagger.py','contact_train.py','contact_surface.py','contact_native.py','contact_library.py','contact_task.py','contact_control.py')]
    source_files.append(ROOT/'tools/build_contact_transfer_library.py')
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    write(run/'source_contract.json',sources)
    state=dict(status='running',stage='tracking',started_at=time.time(),runner=identity(os.getpid()),
        num_envs=plan['num_envs'],scale_axis=plan['scale_axis'],completed=[])
    write(run/'runner-process.json',state['runner'])
    def status(stage,**extra):
        state.update(stage=stage,updated_at=time.time(),**extra);write(run/'status.json',state)
    def check_sources():
        for rel,digest in sources.items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Source changed during staged run; review before advancing: '+rel)
    def checkpoint(path):
        value=torch.load(path,map_location='cpu',weights_only=False)
        asset=json.loads(value['robot_asset_json']) if 'robot_asset_json' in value else json.loads(value['source_metadata']['robot_asset_json'])
        validate_g1_asset_metadata(asset,context=str(path));require_profile(asset,'surface_v2',context=str(path))
        return value
    def execute(stage,argv):
        check_sources()
        if hashlib.sha256(Path(plan['task_manifest']).read_bytes()).hexdigest()!=manifest_hash:
            raise RuntimeError('Task manifest changed')
        command=[str(ROOT/'tools/run_native.sh'),*map(str,argv)]
        log_path=run/(stage+'.log')
        with log_path.open('xb') as log:
            process=subprocess.Popen(command,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record={**identity(process.pid),'cwd':str(ROOT),'gpu':plan['gpu'],'log':str(log_path)}
        write(run/(stage+'-process.json'),record);status(stage,child=record)
        while process.poll() is None:
            time.sleep(10)
            state['updated_at']=time.time();write(run/'status.json',state)
        if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; inspect {log_path}')
        state['completed'].append(stage);status(stage,child=None)
    try:
        owner=json.loads((run/'tracking/process.json').read_text())
        status('tracking',child=owner)
        while same_live_process(owner):
            progress={}
            csv_path=run/'tracking/train_log.csv'
            if csv_path.exists():
                rows=list(csv.DictReader(csv_path.open()))
                if rows:
                    last=rows[-1];progress=dict(iteration=int(last['iteration']),sim_steps=int(last['sim_steps']))
            status('tracking',tracking_progress=progress)
            time.sleep(10)
        tracking=run/'tracking'
        if (tracking/'invalid_transition.pt').exists():raise RuntimeError('Tracking stopped on invalid physics; no downstream training started')
        final=checkpoint(tracking/'last.pt')
        if int(final['iteration'])!=plan['tracking']['iterations']:raise RuntimeError('Tracking ended before its configured budget')
        state['completed'].append('tracking');status('tracking_complete')
        target=int(plan['tracking']['iterations'])
        evaluations=[]
        for it in (target,target-2000,target-4000):
            teacher=tracking/f'ckpt_{it:06d}.pt';checkpoint(teacher)
            name=f'eval_tracking_{it:06d}'
            execute(name,['-m','supermimic.evaluate','--run_dir',tracking,'--ckpt',teacher,
                '--eval_seed','0','--result_base',run/name])
            score=json.loads((run/(name+'_eval.json')).read_text())
            evaluations.append((int(score['n_failed']),-float(score['norm_return']),it,teacher))
        _,_,it,teacher=min(evaluations)
        expected=max(len(__import__('numpy').loadtxt(task['csv'],delimiter=','))-1 for task in spec['tasks'])
        capture=run/'native_motion_library'
        execute('collect_native_library',['-m','supermimic.contact_collect','--task_manifest',plan['task_manifest'],
            '--checkpoint',teacher,'--collision_profile','surface_v2','--output',capture,
            '--num_envs',len(spec['tasks']),'--all_motions','--steps',expected+1,'--seed','0','--headless'])
        library=run/'whole_transfers';coverage=plan['whole_transfer_coverage']
        execute('build_whole_transfers',['tools/build_contact_transfer_library.py','--capture',capture,'--output',library,
            '--min-motions',coverage['minimum_motions'],'--min-transfers',coverage['minimum_transfers']])
        catalogue=json.loads((library/'catalogue.json').read_text())
        accepted=dict(teacher=str(teacher),recording=str(library/'rollout.pt'),iteration=it,
            whole_transfer_count=catalogue['whole_transfer_count'],motions=catalogue['motions'])
        write(run/'accepted_tracking.json',accepted);status('whole_transfers_accepted',accepted_tracking=accepted)
        dagger=run/'dagger';d=plan['dagger']
        execute('dagger',['-m','supermimic.contact_initialize','--task_manifest',plan['task_manifest'],
            '--checkpoint',accepted['teacher'],'--recording',accepted['recording'],'--output',dagger,
            '--collision_profile','surface_v2','--num_envs',plan['num_envs'],'--bc_steps','2000',
            '--dagger_rounds',d['rounds'],'--dagger_min_rounds',d['minimum_rounds'],
            '--dagger_updates_per_round',d['updates_per_round'],'--dagger_rollouts_per_segment',d['student_rollouts_per_segment'],
            '--dagger_action_noise','.03','--bc_batch','4096','--headless'])
        initial=dagger/'checkpoint_dagger.pt';initial_state=checkpoint(initial)
        fresh=int(initial_state['dagger']['fresh_native_student_transitions'])
        if fresh<=0:raise RuntimeError('DAgger produced no fresh student transitions')
        write(run/'accepted_dagger.json',dict(checkpoint=str(initial),fresh_native_student_transitions=fresh,
            note='best fresh equal-segment action-MSE initialization, not a claim of native task success'))
        from supermimic.contact_library import build_contact_clock
        clock=build_contact_clock(torch.load(accepted['recording'],map_location='cpu',weights_only=False),hold=1,role_aware=True)
        budget=max(b-a for a,b in zip(clock.bounds,clock.bounds[1:]))+30
        buffer=max(128,((budget+63)//64)*64)
        s=plan['segments']
        execute('independent_segments',['-m','supermimic.contact_train','--task_manifest',plan['task_manifest'],
            '--checkpoint',accepted['teacher'],'--recording',accepted['recording'],'--resume',initial,
            '--prior_checkpoint',initial,'--output',run/'segments','--collision_profile','surface_v2',
            '--num_envs',plan['num_envs'],'--iterations',s['iterations'],'--policy_gradient','dmo',
            '--completion_mode','accumulated_target','--sequence_length','1','--dmo_microbatch','1024',
            '--contact_position_loss','surface_region','--surface_samples_per_part',s['surface_samples_per_part'],
            '--surface_temperature','.02','--contact_target_normal','0','0','1','--segment_grace_steps','30',
            '--buffer_steps',buffer,'--warmup_steps',buffer,'--wm_batches','16','--wm_batch','1024',
            '--learning_rate','3e-6','--wm_learning_rate','2e-5','--prior_weight','.01',
            '--save_every','50','--log_every','10','--check_dmo','--headless'])
        state.update(status='complete',completed_at=time.time());status('complete')
    except Exception as error:
        state.update(status='failed',error=str(error));status(state['stage']);raise


if __name__=='__main__':main()
