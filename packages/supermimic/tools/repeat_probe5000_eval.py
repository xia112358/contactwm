"""Repeat hybrid 5000 on GPU3, matching the existing 6000 evaluation protocol."""
from supermimic.paths import runtime_path, runtime_root
import json
import os
from pathlib import Path
import subprocess
import time
from repeat_mixed_eval import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = runtime_path('results/mixed_repeat_5000/hybrid_probe')
WEB = runtime_path('results/mixed_comparison_web')

def publish():
    data=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed_*_eval.json'))]
    old=json.loads((runtime_path('results/mixed_repeat_6000/summary.json')).read_text())['runs']['hybrid_probe']
    report={'checkpoint':5000,'requested_repeats':10,'completed':len(data),
            'protocol':'Independent processes on GPU3; seeds 0-9; nominal from-start evaluation without reset/observation/domain noise',
            'comparison_6000':old,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    rows=[]
    current=None
    if data:
        current={'passed':stats([145-d['n_failed'] for d in data]),
                 'groups':{g:{'passed':stats([d['groups'][g]['n_clips']-d['groups'][g]['n_failed'] for d in data]),
                              'return':stats([d['groups'][g]['norm_return'] for d in data])} for g in data[0]['groups']},
                 'per_clip_success_rate':{c:sum(not d['per_clip_lengths'][c]['failed'] for d in data)/len(data) for c in data[0]['per_clip_lengths']}}
        report.update(current)
    for label,d,n in [('5000',current,len(data)),('6000',old,old['completed'])]:
        if d is None:continue
        s=d['passed'];cells=''.join(f'<td>{d["groups"][g]["passed"]["mean"]:.1f} ± {d["groups"][g]["passed"]["std"]:.1f}</td>' for g in ['flatwalk','lowstep','lowclimb','highclimb'])
        rows.append(f'<tr><th>{label}</th><td>{n}/10</td><td>{s["mean"]:.1f} ± {s["std"]:.1f}</td><td>{s["mean"]/145:.1%}</td><td>{s["minimum"]}–{s["maximum"]}</td>{cells}</tr>')
    tmp=OUT.parent/'summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(OUT.parent/'summary.json')
    page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>混合 probe 后期退化复核</title><style>body{font:17px system-ui;background:#101722;color:#e6edf5;margin:40px}table{border-collapse:collapse}td,th{padding:14px;border:1px solid #456}a{color:#8acaff}</style><h1>混合 probe：5000 与 6000 步重复评估</h1><p>各 10 轮，同一 GPU3，种子 0–9，每轮独立进程，每条轨迹从头跑一次；没有初始扰动、观测噪声或物理参数随机化。均值 ± 轮间样本标准差。这衡量固定模型的数值重复性，不是多训练种子或带扰动鲁棒性结果。</p><table><tr><th>训练步数</th><th>评估轮数</th><th>通过条数 /145</th><th>平均成功率</th><th>范围</th><th>平地 /9</th><th>低台阶 /56</th><th>低攀爬 /40</th><th>高攀爬 /40</th></tr>'''+''.join(rows)+'</table><p>更新时间：'+report['updated_at']+'</p><a href="/repeats.html">6000 步不同采样器对比</a> · <a href="/">视频与单轮结果</a>'
    (WEB/'probe_regression.html').write_text(page)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);publish()
    name='mixed_terrain_walk_hybrid_probe_main'
    for seed in range(10):
        base=OUT/f'seed_{seed:02d}'
        if Path(str(base)+'_eval.json').exists():continue
        cmd=[str(ROOT/'tools/run_native.sh'),'supermimic/evaluate.py','--run_dir','logs/'+name,'--ckpt','logs/'+name+'/ckpt_005000.pt','--eval_seed',str(seed),'--result_base',str(base)]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
        with base.with_suffix('.log').open('w') as log:
            subprocess.run(cmd,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(seed,'complete',flush=True);publish()
    print('ALL COMPLETE',flush=True)
