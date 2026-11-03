"""Independent-process nominal evaluations; preserve official single-run results."""
from supermimic.paths import runtime_path, runtime_root
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = runtime_path('results/mixed_repeat_6000')
RUNS = {'force6': '新版本＋原采样器', 'hybrid_probe': '新版本＋混合 probe'}
LOCK = threading.Lock()

def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else 0,
                minimum=min(values), maximum=max(values), values=values)

def publish():
    with LOCK:
        report = {'protocol': 'nominal, independent process per repeat, no observation/reset/domain noise',
                  'checkpoint': 6000, 'requested_repeats': 10, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': {}}
        rows=[]
        for key,label in RUNS.items():
            results=[json.loads(p.read_text()) for p in sorted((OUT/key).glob('seed_*_eval.json'))]
            if not results: continue
            counts=[145-r['n_failed'] for r in results]
            group={g: {'passed': stats([r['groups'][g]['n_clips']-r['groups'][g]['n_failed'] for r in results]),
                       'return': stats([r['groups'][g]['norm_return'] for r in results])} for g in results[0]['groups']}
            clips={c: sum(not r['per_clip_lengths'][c]['failed'] for r in results)/len(results) for c in results[0]['per_clip_lengths']}
            report['runs'][key]={'label':label,'completed':len(results),'passed':stats(counts),
                'success_rate':stats([n/145 for n in counts]),'groups':group,'per_clip_success_rate':clips}
            s=stats(counts)
            cells=''.join(f'<td>{group[g]["passed"]["mean"]:.1f} ± {group[g]["passed"]["std"]:.1f}</td>' for g in ['flatwalk','lowstep','lowclimb','highclimb'])
            rows.append(f'<tr><th>{label}</th><td>{len(results)}/10</td><td>{s["mean"]:.1f} ± {s["std"]:.1f} /145</td><td>{min(counts)}–{max(counts)}</td>{cells}</tr>')
        temp=OUT/'summary.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(OUT/'summary.json')
        page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>重复评估</title><style>body{font:17px system-ui;background:#101722;color:#e6edf5;margin:40px}table{border-collapse:collapse}td,th{padding:16px;border:1px solid #456}a{color:#8acaff}</style><h1>6000 步模型 · 各 10 轮独立评估</h1><p>相同种子 0–9；每轮独立进程，每条轨迹从头跑一次。保持原正式评估条件：无初始扰动、无观测噪声、无物理参数随机化。此实验测数值重复性，不代表有扰动鲁棒性或多训练种子结果。</p><p>均值 ± 轮间样本标准差；不是置信区间。每 30 秒自动刷新。</p><table><tr><th>版本</th><th>完成轮数</th><th>总通过数</th><th>范围</th><th>平地 /9</th><th>低台阶 /56</th><th>低攀爬 /40</th><th>高攀爬 /40</th></tr>'''+''.join(rows)+'</table><p>更新时间：'+report['updated_at']+'</p><a href="/">返回单轮结果和视频</a>'
        (runtime_path('results/mixed_comparison_web/repeats.html')).write_text(page)

def run(key,gpu):
    folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
    name='mixed_terrain_walk_'+key+'_main'
    for seed in range(10):
        base=folder/f'seed_{seed:02d}'
        if Path(str(base)+'_eval.json').exists():continue
        cmd=[str(ROOT/'tools/run_native.sh'),'supermimic/evaluate.py','--run_dir','logs/'+name,'--ckpt','logs/'+name+'/ckpt_006000.pt','--eval_seed',str(seed),'--result_base',str(base)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with base.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(key,seed,'complete',flush=True);publish()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,key,gpu) for key,gpu in [('force6',1),('hybrid_probe',3)]]
        for job in jobs: job.result()
    publish();print('ALL COMPLETE',flush=True)
