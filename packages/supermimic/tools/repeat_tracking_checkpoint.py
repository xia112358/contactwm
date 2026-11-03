"""Independent-process nominal repeats of a fixed tracking checkpoint."""
from __future__ import annotations
from supermimic.paths import runtime_root
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def identity(pid,command):
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,proc_start_ticks=stat[19],command=command,cwd=str(runtime_root()))


def stats(values):
    return dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values),maximum=max(values),values=values)


def summarize(results):
    if not results:return None
    size=len(results[0]['per_clip_lengths'])
    names=list(results[0]['per_clip_lengths'])
    if any(set(r['per_clip_lengths'])!=set(names) for r in results):raise ValueError('Inconsistent evaluation coverage')
    return dict(completed=len(results),total=size,passed=stats([size-r['n_failed'] for r in results]),
        norm_return=stats([r['norm_return'] for r in results]),
        length_ratio=stats([r['mean_length_ratio'] for r in results]),
        groups={g:dict(total=results[0]['groups'][g]['n_clips'],
            passed=stats([r['groups'][g]['n_clips']-r['groups'][g]['n_failed'] for r in results])) for g in results[0]['groups']},
        clips={name:dict(passed=sum(not r['per_clip_lengths'][name]['failed'] for r in results),
            repeats=len(results),length_ratio=stats([r['per_clip_lengths'][name]['length_ratio'] for r in results])) for name in names})


def publish(output,web,plan,state):
    observations=[];buckets={}
    for item in plan['jobs']:
        path=output/(item['name']+'_eval.json')
        if not path.exists():continue
        result=json.loads(path.read_text());iteration=item['iteration']
        if len(result['per_clip_lengths'])!=296:raise ValueError('Incomplete PHP296 native evaluation')
        buckets.setdefault(iteration,[]).append(result)
        observations.append(dict(**item,passed=296-result['n_failed'],norm_return=result['norm_return'],
            length_ratio=result['mean_length_ratio']))
    summaries={str(k):summarize(v) for k,v in buckets.items()}
    fixed=[json.loads((output/(j['name']+'_eval.json')).read_text()) for j in plan['jobs']
           if j['iteration']==11000 and j['seed']==0 and (output/(j['name']+'_eval.json')).exists()]
    report=dict(schema='tracking_checkpoint_repeats_v1',updated_at=time.time(),status=state['status'],
        completed=len(observations),requested=len(plan['jobs']),protocol=plan['protocol'],
        checkpoint_sha256=plan['checkpoint_sha256'],summaries=summaries,
        fixed_seed_11000=summarize(fixed),observations=observations,official=plan['official'])
    write(output/'summary.json',report);web.mkdir(parents=True,exist_ok=True);write(web/'summary.json',report)
    rows=''.join(f'<tr><td>{r["iteration"]}</td><td>{html.escape(r["name"])}</td><td>{r["seed"]}</td><td>{r["passed"]}/296</td><td>{r["norm_return"]:.4f}</td><td>{r["length_ratio"]:.1%}</td></tr>' for r in observations)
    summaries_html=''
    labels={'locomotion':'移动','lowstep':'低台阶','highstep':'高台阶','lowclimb':'低攀爬','highclimb':'高攀爬'}
    for it,s in summaries.items():
        v=s['passed'];spread=f'{v["std"]:.2f}' if v['std'] is not None else '待多轮'
        groups=''.join(f'<li>{labels.get(g,g)}：{d["passed"]["mean"]:.1f}/{d["total"]}</li>' for g,d in s['groups'].items())
        summaries_html+=f'<section><h2>{it} · 已完成 {s["completed"]} 轮</h2><p>通过数均值 {v["mean"]:.1f}/296，轮间标准差 {spread}，范围 {v["minimum"]}–{v["maximum"]}</p><ul>{groups}</ul></section>'
    unstable=[]
    s=summaries.get('11000')
    if s:
        unstable=sorted([(k,v) for k,v in s['clips'].items() if 0<v['passed']<v['repeats']],key=lambda x:x[1]['length_ratio']['std'] or 0,reverse=True)
    mixed=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["passed"]}/{v["repeats"]}</td><td>{v["length_ratio"]["minimum"]:.1%}–{v["length_ratio"]["maximum"]:.1%}</td></tr>' for k,v in unstable)
    page=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>11k 固定权重重复评估</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#101722;color:#e6edf5}}a{{color:#8acaff}}section{{background:#192433;padding:16px;margin:16px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #456;text-align:left}}</style>
<h1>11k 固定权重 · 独立进程重复评估</h1><p><a href="../">返回训练监测</a> · <a href="summary.json">完整统计 JSON</a></p>
<p>状态：{html.escape(state['status'])}，已完成 {len(observations)}/{len(plan['jobs'])} 轮；每 20 秒刷新。</p>
<p>11k 共 10 轮：前 5 轮固定 seed 0，后 5 轮 seed 1–5；另测 10k 固定 seed 0 三轮作为对照。每轮独立进程重建相同仿真环境，全部 296 条从头执行。无初始扰动、观测噪声或物理随机化。保持原评估精度与成功判据，训练继续运行。</p>
<p>标准差衡量固定权重的执行重复性，不是置信区间、多训练种子结果或带扰动鲁棒性。原自动评估复用仿真环境，原结果完整保留。</p>
<p>原自动评估：10k {plan['official']['10000']['passed']}/296；11k {plan['official']['11000']['passed']}/296。</p>
{summaries_html}<section><h2>每轮独立结果</h2><table><tr><th>Checkpoint</th><th>轮次</th><th>Seed</th><th>整段通过</th><th>回报</th><th>长度覆盖</th></tr>{rows}</table></section>
<section><h2>11k 成败反复的动作</h2><table><tr><th>动作</th><th>通过轮数</th><th>执行长度范围</th></tr>{mixed}</table></section>'''
    temp=web/'index.tmp';temp.write_text(page);temp.replace(web/'index.html')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--web',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();run=a.run.absolute();output=a.output.absolute();web=a.web.absolute()
    output.mkdir(parents=True,exist_ok=False)
    tracking=run/'tracking';checkpoints={it:tracking/f'ckpt_{it:06d}.pt' for it in [10000,11000]}
    hashes={str(it):hashlib.sha256(path.read_bytes()).hexdigest() for it,path in checkpoints.items()}
    official={}
    for it in checkpoints:
        result=json.loads((tracking/f'ckpt_{it:06d}_eval.json').read_text())
        official[str(it)]=dict(passed=296-result['n_failed'],norm_return=result['norm_return'])
    # Interleave an unchanged-weight control to separate evaluation variance from learning changes.
    jobs=[]
    for i,seed in enumerate([0]*5+[1,2,3,4,5]):
        jobs.append(dict(iteration=11000,seed=seed,name=f'ckpt11000_repeat_{i:02d}_seed{seed}'))
        if i in [0,2,4]:jobs.append(dict(iteration=10000,seed=0,name=f'ckpt10000_control_{i//2:02d}_seed0'))
    plan=dict(protocol='Independent process each repeat; original nominal native evaluation with no observation/reset/domain noise; 11k fixed-seed and seed-sweep cohorts plus 10k controls',
        jobs=jobs,checkpoint_sha256=hashes,official=official,gpu=a.gpu,run=str(run),web=str(web))
    write(output/'plan.json',plan)
    state=dict(status='running',started_at=time.time(),runner=identity(os.getpid(),['repeat_tracking_checkpoint.py']))
    write(output/'status.json',state);publish(output,web,plan,state)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
    try:
        for item in jobs:
            checkpoint=checkpoints[item['iteration']]
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=hashes[str(item['iteration'])]:raise ValueError('Checkpoint changed')
            command=[str(ROOT/'tools/run_native.sh'),'-m','supermimic.evaluate','--run_dir',str(tracking),
                '--ckpt',str(checkpoint),'--eval_seed',str(item['seed']),'--result_base',str(output/item['name'])]
            log=output/(item['name']+'.log')
            with log.open('xb') as stream:
                child=subprocess.Popen(command,cwd=runtime_root(),env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record={**identity(child.pid,command),'log':str(log),'gpu':a.gpu}
            write(output/(item['name']+'-process.json'),record)
            state.update(active=item,child=record,updated_at=time.time());write(output/'status.json',state)
            code=child.wait()
            if code:raise RuntimeError(f'{item["name"]} exited {code}; see {log}')
            report=publish(output,web,plan,state)
            print(json.dumps(report['observations'][-1]),flush=True)
        state.update(status='complete',completed_at=time.time(),child=None);write(output/'status.json',state)
        publish(output,web,plan,state)
    except Exception as error:
        state.update(status='failed',error=str(error),updated_at=time.time());write(output/'status.json',state)
        publish(output,web,plan,state);raise


if __name__=='__main__':main()
