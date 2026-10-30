"""Run the two prepared input ablations sequentially on GPU3, with evaluation."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'logs/input_ablation_queue.json'
CONFIG = ROOT/'experiments/mixed_input_ablation.json'

def write(path, data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def identity(pid):
    p=Path('/proc')/str(pid)
    return {'pid':pid,'proc_start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19],
            'command':(p/'cmdline').read_bytes().decode().strip('\0').split('\0')}

def training_pid(run):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeError):continue
        if 'supermimic/train.py' in cmd and str(run) in cmd:return int(p.name)
    return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-first', action='store_true')
    args=parser.parse_args()
    config=json.loads(CONFIG.read_text());env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
    state={'status':'running','gpu':3,'queue':['height_only','force_only'],'completed':[],
           'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'runner':identity(os.getpid())}
    write(STATE,state)
    for model in state['queue']:
        item=config['new'][model];name=item['run'];run=ROOT/'logs'/name
        try:
            recovering=args.resume_first and model=='height_only'
            if recovering:
                recovery=json.loads((run/'recovery_after_disk_full.json').read_text())
                assert Path(recovery['checkpoint']).is_file()
            else:
                assert not run.exists(),f'Run already exists: {run}'
            assert hashlib.sha256(Path(config['task_manifest']).read_bytes()).hexdigest()==config['task_manifest_sha256']
            run.mkdir(exist_ok=recovering)
            meta={**config['common'],'prepared_at':state['started_at'],'physical_gpu':3,
                  'contact_model':model,'iterations':6000,'num_envs':4096,'physical_envs':4096,
                  'statistics_only_probes':0,'seed':0,'group_counts':{'lowstep':56,'lowclimb':40,'highclimb':40,'flatwalk':9},
                  'comparison_family':'mixed145_v1','comparison_label':{'height_only':'仅高度图＋原采样器','force_only':'仅六部位力＋原采样器'}[model],
                  'policy_observation_dim':item['policy_observation_dim'],'world_model_input_dim':item['world_model_input_dim'],
                  'task_manifest_sha256':config['task_manifest_sha256'],'current_state':'starting','processes':{}}
            write(run/'experiment.json',meta)
            cmd=item['launch_argv'].copy();cmd[1]='experiments/run_input_ablation_frozen.sh'
            if recovering:
                cmd[6]=recovery['checkpoint']
                meta.update(resume_iteration=recovery['iteration'], recovery=recovery)
                write(run/'experiment.json',meta)
            with (run/'pipeline.log').open('ab') as log:
                pipe=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(run/'launch_state.json',{**identity(pipe.pid),'cwd':str(ROOT),'gpu':3})
            state.update(active=model,pipeline=identity(pipe.pid));write(STATE,state)
            for _ in range(120):
                if (run/'args.json').exists() and training_pid(run) is not None:break
                if pipe.poll() is not None:raise RuntimeError('Pipeline exited before args.json')
                time.sleep(1)
            assert (run/'args.json').exists(),'Training initialization timed out'
            with (run/'auto_eval.log').open('ab') as log:
                watch=subprocess.Popen(item['evaluation_argv'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['processes']={'training':training_pid(run),'evaluation':watch.pid};meta['current_state']='running'
            write(run/'experiment.json',meta)
            print('STARTED',model,meta['processes'],flush=True)
            code=pipe.wait()
            if code:
                write(run/'STOPPED.json',{'reason':'pipeline failed','exitcode':code})
                watch.wait(timeout=180)
                raise RuntimeError(f'{model} pipeline exit {code}')
            watch.wait(timeout=180)
            assert watch.returncode==0,'Automatic evaluation failed'
            result=json.loads((run/'ckpt_006000_eval.json').read_text())
            assert len(result['per_clip'])==145,'Incomplete final evaluation'
            meta['current_state']='completed';meta['completed_at']=time.strftime('%Y-%m-%d %H:%M:%S');write(run/'experiment.json',meta)
            state['completed'].append(model);write(STATE,state);print('COMPLETED',model,flush=True)
        except Exception as exc:
            state.update(status='failed',error=str(exc));write(STATE,state);raise
    state.update(status='completed',active=None,completed_at=time.strftime('%Y-%m-%d %H:%M:%S'));write(STATE,state)
    print('QUEUE COMPLETE',flush=True)

if __name__=='__main__':main()
