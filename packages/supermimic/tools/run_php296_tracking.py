"""Launch the authorized PHP-only surface-v2 tracker and owned evaluation watcher."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def identity(pid,command):
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,proc_start_ticks=stat[19],command=command,cwd=str(runtime_root()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--task_manifest',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=20000)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();run=a.run.absolute();manifest=a.task_manifest.absolute()
    if a.iterations<1000 or a.iterations%1000:raise ValueError('Budget must be a positive multiple of 1000')
    spec=json.loads(manifest.read_text())
    if spec.get('source_component')!='php_all_five_groups_296' or len(spec['tasks'])!=296:
        raise ValueError('Expected the complete PHP-only 296-motion manifest')
    if spec.get('excluded_datasets')!=['LAFAN1']:raise ValueError('LAFAN must be excluded')
    run.mkdir(parents=True,exist_ok=False);tracking=run/'tracking';tracking.mkdir()
    sources=run/'source';sources.mkdir()
    digests={}
    for source in sorted((ROOT/'supermimic').glob('*.py')):
        shutil.copyfile(source,sources/source.name);digests[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ['prepare_php296.py','run_php296_tracking.py','run_native.sh']:
        shutil.copyfile(ROOT/'tools'/name,sources/name)
    write(run/'source_contract.json',digests)
    plan=dict(schema='php296_surface_v2_tracking_v1',task_manifest=str(manifest),
        task_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        gpu=a.gpu,num_envs=4096,statistics_only_probes=888,physical_envs=4984,
        collision_profile='surface_v2',contact_model='force6',tracking_only=True,dataset_label='PHP 五组',
        excluded_datasets=['LAFAN1'],evaluation_interval=1000,
        pipeline_log=str(run.parent/(run.name+'-supervisor.log')),
        preflight_report=str(runtime_path('validation/php296_surface_v2_preflight.json')),
        tracking=dict(iterations=a.iterations,initialization='fresh random networks and optimizers',
            budget_semantics='upper limit; intermediate evaluations support an explicit early-stop decision'),
        variant='heightmap + force6 + all-from-start probes + recent30 failure-window retries',
        reference_run='mixed_terrain_walk_start_probe_main',reference_single_evaluation='136/145 at 6000 in legacy physics; not a surface-v2 result',
        probe_training_data=False,probe_normalizer_updates=False,group_counts=spec['group_counts'])
    write(run/'plan.json',plan)
    state=dict(status='running',stage='tracking_startup',started_at=time.time(),runner=identity(os.getpid(),[sys.executable,*sys.argv]))
    write(run/'runner-process.json',state['runner'])
    def status(stage,**extra):
        state.update(stage=stage,updated_at=time.time(),**extra);write(run/'status.json',state)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',TORCHINDUCTOR_COMPILE_THREADS='4')
    def launch(name,argv,path):
        command=[str(ROOT/'tools/run_native.sh'),*map(str,argv)]
        with path.open('xb') as stream:
            child=subprocess.Popen(command,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record={**identity(child.pid,command), 'log':str(path),'gpu':a.gpu,'started_at':time.time()}
        write(run/name,record);return child,record
    try:
        training,owner=launch('tracking/process.json',['-m','supermimic.train','--task_manifest',manifest,
            '--collision_profile','surface_v2','--contact_model','force6','--policy_force_observation','measured',
            '--perception','heightmap','--reset_sampler','completion_recent30_failure_window',
            '--probes_per_motion','3','--probe_mode','all_long','--num_envs','4096','--iterations',a.iterations,
            '--buffer_steps','1000','--wm_batches','10','--wm_batch','2048','--pi_batch','4096',
            '--horizon','32','--wm_window','8','--hist','10','--goal_frames','10','--utd','.4',
            '--warmup_steps','64','--wm_only_iters','500','--ckpt_every','500','--log_every','50',
            '--seed','0','--amp','bf16','--capture_physics_failure','--log_dir',tracking],tracking/'stdout.log')
        status('tracking_startup',child=owner)
        while not (tracking/'args.json').exists():
            if training.poll() is not None:raise RuntimeError('Training exited during startup; inspect tracking/stdout.log')
            time.sleep(1)
        watcher,_=launch('eval-monitor-process.json',['-m','supermimic.evaluate','--run_dir',tracking,
            '--watch','--iteration_interval','1000','--stop_after',a.iterations,'--eval_seed','0'],run/'evaluation.log')
        while training.poll() is None:
            status('tracking',evaluation_alive=watcher.poll() is None)
            time.sleep(10)
        if training.returncode:raise RuntimeError(f'Training exited {training.returncode}; inspect tracking/stdout.log')
        while watcher.poll() is None:
            status('final_evaluation');time.sleep(10)
        if watcher.returncode:raise RuntimeError(f'Evaluation exited {watcher.returncode}; inspect evaluation.log')
        evaluations=[]
        for path in tracking.glob('ckpt_*_eval.json'):
            result=json.loads(path.read_text())
            if len(result['per_clip'])!=296:raise ValueError('Incomplete native evaluation')
            evaluations.append((result['n_failed'],-result['norm_return'],path))
        if not evaluations:raise RuntimeError('No full native evaluation is available')
        _,_,best=min(evaluations)
        write(run/'best_evaluated.json',dict(checkpoint=str(best).replace('_eval.json','.pt'),evaluation=str(best),
            selection='fewest full-trajectory failures, then highest normalized return; fixed evaluation seed 0'))
        state['status']='complete';status('complete')
    except Exception as error:
        state.update(status='failed',error=str(error));status(state['stage'])
        write(tracking/'STOPPED.json',dict(reason=str(error),time=time.time()))
        raise


if __name__=='__main__':main()
