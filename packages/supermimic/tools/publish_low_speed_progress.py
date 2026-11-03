"""Publish the current pooled-contact training status to the existing local web service."""
from supermimic.paths import runtime_path
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def rows(path):
    result=[]
    try:
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    except FileNotFoundError:pass
    return result


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>低速接触训练进度</title>
<style>body{background:#101722;color:#e5edf7;font:16px/1.6 system-ui;margin:auto;max-width:1250px;padding:24px}h1{margin-bottom:4px}small,.muted{color:#a5b5c9}section,.card{background:#1a2636;border:1px solid #314054;border-radius:12px;padding:18px;margin:16px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card{margin:0}.value{font-size:28px;color:#65d7b2}progress{width:100%;height:22px}svg{width:100%;height:230px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #344259}select,input{padding:8px;background:#24364d;color:white;border:1px solid #51647e;border-radius:6px}a{color:#7bc8ff}.scroll{max-height:490px;overflow:auto}pre{white-space:pre-wrap;font-size:12px}.warn{color:#ffbd79}</style>
<h1>低速接触训练</h1><div class="muted">109 条动作 · 1,312 个通过恢复检查的段 · 共享策略，单段独立重置 · 每 5 秒更新</div>
<section><h2 id="stage">加载中…</h2><progress id="progress" max="500" value="0"></progress><div id="time" class="muted"></div><div id="error" class="warn"></div></section>
<div class="cards"><div class="card">单段训练迭代<div class="value" id="iteration">—</div></div><div class="card">本批接触目标成功率<div class="value" id="success">—</div></div><div class="card">策略 / WM 更新<div class="value" id="updates">—</div></div><div class="card">DAgger 最佳轮次<div class="value" id="best">—</div></div></div>
<section><h2>单段原生接触目标成功率</h2><div class="muted">按段平均；来自训练采样，不是独立评测。以真实约束接触、目标区域及根姿态判定。</div><div id="native-chart"></div></section>
<section><h2>DAgger 已完成：40 轮</h2><div id="dagger-note"></div><div class="muted">动作 MSE 曲线使用对数纵轴。第 20 轮后发生显存不足，改为 CPU 回放，从第 19 轮最佳权重续训；同轮显示续训记录。</div><div id="dagger-chart"></div></section>
<section><h2>最近一批各段表现</h2><select id="group"><option value="">全部任务</option><option>locomotion</option><option>lowstep</option><option>lowclimb</option></select> <input id="search" placeholder="动作名或原始段号"> <label><input type="checkbox" id="failed">只看未全部成功</label><div id="count" class="muted"></div><div class="scroll"><table><thead><tr><th>原始段号</th><th>动作</th><th>成功率</th><th>尝试数</th><th>根误差 cm</th></tr></thead><tbody id="cases"></tbody></table></div></section>
<details><summary>运行信息和最新日志</summary><pre id="details"></pre><pre id="log"></pre></details><p><a href="data.json">查看原始进度数据</a></p>
<script>
let DATA;const el=x=>document.getElementById(x),fmt=(v,n=3)=>v==null?'—':Number(v).toFixed(n),pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
function chart(id,points,log=false){if(!points.length){el(id).textContent='等待首个完整训练批次…';return}const w=1080,h=230,l=65,r=20,t=18,b=35;const ys=points.map(p=>log?Math.log10(Math.max(p.y,1e-9)):p.y);let ymin=log?Math.min(...ys):0,ymax=log?Math.max(...ys):1;if(ymax===ymin)ymax=ymin+1;const xmax=Math.max(1,...points.map(p=>p.x));const X=x=>l+x/xmax*(w-l-r),Y=y=>h-b-(y-ymin)/(ymax-ymin)*(h-t-b);let svg=`<svg viewBox="0 0 ${w} ${h}" role="img">`;for(let i=0;i<5;i++){let y=ymin+(ymax-ymin)*i/4;svg+=`<line x1="${l}" x2="${w-r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#344259"/><text x="3" y="${Y(y)+5}" fill="#a5b5c9" font-size="13">${log?Math.pow(10,y).toFixed(3):(y*100).toFixed(0)+'%'}</text>`}svg+=`<polyline points="${points.map((p,i)=>X(p.x)+','+Y(ys[i])).join(' ')}" fill="none" stroke="#65d7b2" stroke-width="2.5"/>`;points.forEach((p,i)=>svg+=`<circle cx="${X(p.x)}" cy="${Y(ys[i])}" r="4" fill="#7bc8ff"><title>第 ${p.x} 轮/迭代：${log?fmt(p.y,5):pct(p.y)}</title></circle>`);svg+=`<text x="${l}" y="225" fill="#a5b5c9">0</text><text x="${w-65}" y="225" fill="#a5b5c9">${xmax}</text></svg>`;el(id).innerHTML=svg}
function cases(){if(!DATA)return;const list=DATA.cases.filter(c=>(!el('group').value||c.group===el('group').value)&&(!el('failed').checked||c.success_rate<1)&&(c.name+' '+c.case_id).includes(el('search').value));el('count').textContent=`显示 ${list.length} / ${DATA.cases.length} 段`;el('cases').replaceChildren(...list.map(c=>{const tr=document.createElement('tr');[c.case_id,c.name,pct(c.success_rate),c.attempts,fmt(c.root_error_cm,2)].forEach(v=>{let td=document.createElement('td');td.textContent=v;tr.append(td)});return tr}))}
async function refresh(){try{let d=await(await fetch('data.json?t='+Date.now(),{cache:'no-store'})).json();DATA=d;let s=d.status;const names={starting:'初始化',collecting:'原生采样',training:'更新策略 / 世界模型',complete:'已完成',failed:'运行失败',interrupted:'已停止'};el('stage').textContent='单段训练 · '+(names[s.phase]||s.phase||'启动中')+(d.live?' · 进程运行中':' · 进程已退出');el('iteration').textContent=(s.iteration||0)+' / 500';el('progress').value=s.iteration||0;el('success').textContent=pct(d.latest_success);el('updates').textContent=(s.policy_updates||0)+' / '+(s.wm_updates||0);el('best').textContent=d.dagger_best.best_round;el('time').textContent='页面数据更新：'+new Date(d.published_at*1000).toLocaleString()+'；训练状态更新：'+(s.updated_at?new Date(s.updated_at*1000).toLocaleString():'等待')+(s.dmo_processed?'；本轮梯度处理 '+s.dmo_processed+'/'+s.dmo_total+' 个环境':'');el('error').textContent=s.error||(!d.live&&s.phase!=='complete'?'训练进程已退出，请检查日志。':'');el('dagger-note').textContent='最佳第 '+d.dagger_best.best_round+' 轮；动作 MSE '+fmt(d.dagger_best.mse,5)+'。DAgger 存活率不等同于接触目标成功率。';chart('native-chart',d.training);chart('dagger-chart',d.dagger,true);el('details').textContent=JSON.stringify(d.run,null,2);el('log').textContent=d.log_tail;cases()}catch(e){el('error').textContent='刷新失败：'+e.message}}
['group','search','failed'].forEach(x=>el(x).addEventListener('input',cases));refresh();setInterval(refresh,5000);
</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'index.html').write_text(HTML)
    lib=runtime_path('logs/php_low_speed_10k_extract_20261029_v1/library_final')
    selected=read(lib/'validated_cases.json')['cases']
    dag_runs=[runtime_path('logs/php_low_speed_10k_direct_dagger_20261029_v1'),runtime_path('logs/php_low_speed_10k_direct_dagger_20261029_resume_v2')]
    merged={r['round']:r for run in dag_runs for r in rows(run/'initialization/dagger/metrics.jsonl')}
    summary=read(dag_runs[-1]/'initialization/dagger/summary.json')
    while True:
        proc=read(a.run/'process.json',{});live=False
        try:
            stat=Path(f'/proc/{proc["pid"]}/stat').read_text().rsplit(')',1)[1].split()
            live=stat[19]==proc['start_ticks'] and stat[0]!='Z'
        except (FileNotFoundError,KeyError):pass
        status=read(a.run/'training/status.json',{})
        metrics=rows(a.run/'training/metrics.jsonl');last=metrics[-1].get('native_batch',{}) if metrics else {}
        cases=[]
        for row in last.get('segments',[]):
            spec=selected[row['segment']-1]
            cases.append(dict(case_id=spec['case'],name=spec['name'],group=spec['group'],**{k:row[k] for k in ('success_rate','attempts','root_error_cm')}))
        payload=dict(published_at=time.time(),live=live,status={k:v for k,v in status.items() if k not in ('native','timing')},
            latest_success=last.get('macro_success_rate'),cases=cases,
            training=[dict(x=r['iteration'],y=r['native_batch']['macro_success_rate']) for r in metrics if r.get('native_batch')],
            dagger=[dict(x=k,y=v['score']) for k,v in sorted(merged.items())],
            dagger_best=dict(best_round=summary['best_round'],mse=summary['best_report']['score']),
            run=dict(directory=str(a.run),pid=proc.get('pid'),gpu=0,shared_cases=1312,source_motions=109,independent_segments=True),
            log_tail=(a.run/'stdout.log').read_text()[-2500:])
        tmp=a.output/'data.tmp';tmp.write_text(json.dumps(payload,allow_nan=False));tmp.replace(a.output/'data.json')
        if not live:break
        time.sleep(5)


if __name__=='__main__':main()
