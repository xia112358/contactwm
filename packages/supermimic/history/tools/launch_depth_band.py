"""Launch the validated lower-band oracle experiment and its evaluation watcher."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
run='mixed_terrain_walk_depth_band_noforce_main'
p=ROOT/'logs'/run
if (p/'args.json').exists():raise SystemExit('Refusing to overwrite an existing training run')
p.mkdir(exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',SM_PROFILE='1')
cmd=['bash','experiments/run_depth_band_frozen.sh',run,'6000','data/mixed_terrain_walk/task.json','force6','','completion_recent30_failure_window','all_long','none','panorama_oracle']
meta=json.loads((ROOT/'experiments/oracle_depth_preoptimization.json').read_text())
now=datetime.datetime.now().isoformat();meta.update(prepared_at=now,started_at=now,processes={},current_state='running',depth_layout='lower_band',depth_encoder='mlp',depth_encoder_layers=[5120,128,63],comparison_label='规则低仰角真值深度＋WM力／策略无力',policy_image_resolution=[40,128],world_model_image_resolution=[40,128],ray_count=5120,azimuth_degrees=[-180,180],elevation_degrees=[-90,15],compiled_components=['actor depth MLP','WM depth MLP','critic kinematics','WM state transition','actor decoder','critic'],optimization_validation='validation/depth_optimized_validation.json')
meta['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['supermimic/train.py','supermimic/panorama.py','supermimic/lowstep.py','supermimic/networks.py','supermimic/evaluate.py','experiments/run_depth_band_frozen.sh']}
(p/'experiment.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
with (p/'pipeline.log').open('ab') as log:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
state=dict(pid=proc.pid,command=cmd,cwd=str(ROOT),proc_start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21],gpu=3,started_at=now)
(p/'launch_state.json').write_text(json.dumps(state,indent=2))
watch=['tools/run_native.sh','supermimic/evaluate.py','--run_dir',str(p),'--watch','--iteration_interval','1000','--stop_after','5000']
with (p/'auto_eval.log').open('ab') as log:e=subprocess.Popen(watch,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
meta['processes']['evaluation']=e.pid
(p/'experiment.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
(p/'eval_watch_state.json').write_text(json.dumps(dict(pid=e.pid,command=watch,proc_start_ticks=Path(f'/proc/{e.pid}/stat').read_text().split()[21]),indent=2))
print(json.dumps({'run':run,'pipeline_pid':proc.pid,'evaluation_pid':e.pid}))
