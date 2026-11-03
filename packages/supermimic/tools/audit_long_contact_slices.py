"""Export all >40-frame cases with measured motion/contact timelines; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from contact_boundary_waits import boundary_wait_bounds

PARTS=['左脚','右脚','左手','右手','左膝','右膝']

def runs(values):
    result=[]
    for i,v in enumerate(values):
        if not result or result[-1][2]!=v: result.append([i,i+1,v])
        else: result[-1][1]=i+1
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only-unexplained',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    rec=torch.load(a.library/'rollout.pt',map_location='cpu',weights_only=False)
    data=rec['data'];spans={s['motion']:s for s in rec['metadata']['recording_spans']}
    catalogue=json.loads((a.library/'validated_cases.json').read_text())['cases']
    out=[];proposals=[]
    for c in catalogue:
        b,e=c['pooled_frames'];nb,ne,wait=boundary_wait_bounds(data,b,e)
        proposals.append(dict(case=c['case'],old_bounds=[b,e],proposed_bounds=[nb,ne],**wait))
        if e-b<=40:continue
        s=data['s'][b:e];active=data['active'][b:e].bool();known=data['known'][b:e].bool()
        series=[];repeat=[];air=[];surface=data['surface'][b:e]
        for j,part in enumerate(PARTS):
            rr=runs(active[:,j].int().tolist());events=[y for x,y,v in rr if v==0 and y-x>=7 and y<e-b]
            if len(events)>1:repeat.append(dict(part=part,onsets=events))
            air.append(max((y-x for x,y,v in rr if v==0),default=0))
            series.append(dict(part=part,runs=rr,unknown=runs((~known[:,j]).int().tolist()),surface=surface[:,j].tolist()))
        tags=[]
        if wait['entire_wait'] or nb>b or ne<e:tags.append('可裁首尾等待')
        if c['original_source_frames'][1]==spans[c['motion']]['end']-spans[c['motion']]['begin']:tags.append('源轨迹尾段')
        if b==spans[c['motion']]['begin']:tags.append('源轨迹首段')
        if repeat:tags.append('多次重新接触')
        target_names=['left_foot','right_foot','left_hand','right_hand','left_knee','right_knee']
        long_air=[dict(part=PARTS[j],frames=air[j]) for j in range(6) if target_names[j] in c['new_contacts'] and air[j]>40]
        if long_air:tags.append('目标部位长时间离地')
        if not tags:tags=['其他长段']
        vel=s[:,7:10].norm(dim=-1);omega=s[:,10:13].norm(dim=-1);joint=s[:,42:71].square().mean(-1).sqrt()
        out.append(dict(**c,tags=tags,wait=wait,proposed_source_frames=[nb-spans[c['motion']]['begin'],ne-spans[c['motion']]['begin']],
            trimmed_frames=ne-nb,repeat=repeat,long_air=long_air,timeline=series,
            speed=vel.tolist(),angular=omega.tolist(),joint=joint.tolist(),root=s[:,:3].tolist(),
            root_path_m=float((s[1:,:3]-s[:-1,:3]).norm(dim=-1).sum()),
            peak_speed=float(vel.max()),peak_joint_speed=float(joint.max())))
    all_count=len(out)
    if a.only_unexplained:
        out=[c for c in out if not (c['wait']['entire_wait'] or c['trimmed_frames']<c['frames'] or c['repeat'] or c['long_air'])]
    out.sort(key=lambda c:-c['frames'])
    result=dict(count=len(out),total_overlong=all_count,excluded_explained=all_count-len(out),dt=rec['metadata']['dt'],source_library=str(a.library.resolve()),
        tags=dict(Counter(t for c in out for t in c['tags'])),cases=out,
        boundary_audit=dict(cases=len(proposals),changed=sum(c['old_bounds']!=c['proposed_bounds'] for c in proposals),entire_wait=sum(c['entire_wait'] for c in proposals)),
        note='标签是诊断线索，可重叠；源轨迹尾段不等于静止等待。裁剪为待重新验证的建议，未修改正式数据或重启训练。')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'data.json').write_text(json.dumps(result,ensure_ascii=False))
    (a.output/'boundary_proposals.json').write_text(json.dumps(proposals,ensure_ascii=False))
    (a.output/'index.html').write_text(HTML.replace('超过 40 帧的接触段 · 逐条审查', '原因待查的长段' if a.only_unexplained else '超过 40 帧的接触段 · 逐条审查'))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))

HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>长接触段逐条审查</title><style>
body{background:#101722;color:#dce6f2;font:15px system-ui;margin:24px}h1{font-size:24px}input,select,button{background:#243348;color:white;border:1px solid #526880;padding:9px;margin:4px}main{display:grid;grid-template-columns:340px 1fr;gap:20px}aside{max-height:78vh;overflow:auto}button{display:block;width:96%;text-align:left;cursor:pointer}button.sel{border-color:#42dfac}article{min-width:0}svg{width:100%;background:#172334}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #415167}small{color:#a9bdd2}.legend{color:#7ce0ba}a{color:#7ce0ba}@media(max-width:850px){main{grid-template-columns:1fr}aside{max-height:220px}}</style>
<h1>超过 40 帧的接触段 · 逐条审查</h1><p id="summary"></p><p>每次只展示当前筛选中最长的 8 段。这里展示原始记录的接触和运动曲线，不是渲染视频。所有帧均为段内相对帧；源帧采用左闭右开区间。</p><input id="search" placeholder="case 编号 / 动作名称"><select id="tag"><option value="">所有类型</option></select><main><aside id="list"></aside><article id="detail"></article></main>
<script>
let D,selected;const $=id=>document.getElementById(id);fetch('data.json').then(r=>r.json()).then(d=>{D=d;$('summary').textContent=`待查 ${d.count} 段，已隐藏 ${d.excluded_explained} 段有明确诊断线索的长段；${d.note}`;Object.entries(d.tags).forEach(([k,v])=>{$('tag').add(new Option(`${k} (${v})`,k))});list()});$('search').oninput=list;$('tag').onchange=list;
function list(){let q=$('search').value.toLowerCase(),t=$('tag').value;let a=D.cases.filter(c=>(!t||c.tags.includes(t))&&(`${c.case} ${c.name}`.toLowerCase().includes(q)));$('list').replaceChildren();a.slice(0,8).forEach(c=>{let b=document.createElement('button');b.textContent=`#${c.case} · ${c.frames} 帧 · ${c.name} — ${c.tags.join(' / ')}`;b.onclick=()=>show(c);$('list').append(b)});if(a.length)show(a[0]);else $('detail').textContent='没有匹配段';}
function curve(values,color,max,y,h,n){return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${values.map((v,i)=>`${110+i*850/n},${y+h-Math.min(v/max,1)*h}`).join(' ')}"/>`}
function show(c){let n=c.frames;let svg='<svg viewBox="0 0 1000 500">';c.timeline.forEach((p,j)=>{let y=25+j*35;svg+=`<text x="5" y="${y+15}" fill="#dce6f2">${p.part}</text>`;p.runs.forEach(([a,b,v])=>{svg+=`<rect x="${110+a*850/n}" y="${y}" width="${(b-a)*850/n}" height="22" fill="${v?'#42c99d':'#3b4c63'}"><title>${p.part} ${a}–${b}：${v?'接触':'未接触'}</title></rect>`});p.unknown.filter(r=>r[2]).forEach(([a,b])=>svg+=`<rect x="${110+a*850/n}" y="${y+18}" width="${(b-a)*850/n}" height="4" fill="#ffba60"/>`)});
[['speed','根部速度 m/s',.05,'#68c7ff'],['angular','角速度 rad/s',.15,'#db9fff'],['joint','关节 RMS rad/s',.2,'#f7bf74']].forEach(([k,label,threshold,color],j)=>{let y=250+j*75,m=Math.max(threshold*2,...c[k]);svg+=`<text x="5" y="${y}" fill="${color}" font-size="12">${label}</text><text x="5" y="${y+18}" fill="#aaa" font-size="11">上限 ${m.toFixed(2)}</text><line x1="110" x2="960" y1="${y+50-threshold/m*50}" y2="${y+50-threshold/m*50}" stroke="#889" stroke-dasharray="4"/>`;svg+=curve(c[k],color,m,y,50,n)});for(let i=0;i<=n;i+=10)svg+=`<text x="${110+i*850/n}" y="490" fill="#cbd5e1" font-size="12">${i}</text>`;svg+='</svg>';
$('detail').innerHTML=`<h2>#${c.case} ${c.name} · ${n} 帧 / ${(n*D.dt).toFixed(2)} 秒</h2><p>${c.tags.join(' ｜ ')}</p><p>源帧 [${c.source_frames}]；新接触：${c.new_contacts.join(', ')||'无'}；维持接触：${c.maintained_contacts.join(', ')||'无'}</p><p class="legend">绿：接触；灰：未接触；橙色底边：证据未知。悬停色块查看帧区间。虚线：低运动量阈值。</p>${svg}<table><tr><th>首端静稳连续帧</th><th>尾端静稳连续帧</th><th>裁剪建议剩余</th><th>根部累计路径</th></tr><tr><td>${c.wait.head_wait}</td><td>${c.wait.tail_wait}</td><td>${c.trimmed_frames} 帧</td><td>${c.root_path_m.toFixed(3)} m</td></tr></table><p>裁剪要求：接触及接触面不变、证据已知，根部速度 &lt;0.05 m/s、角速度 &lt;0.15 rad/s、关节 RMS 速度 &lt;0.2 rad/s，连续至少 10 帧；保留两帧边界。不是仅凭脚接触了就判定静止。</p><p>重复重新接触（离开至少 7 帧）：${c.repeat.map(x=>x.part+' @ '+x.onsets.join(', ')).join('；')||'无'}</p><p>目标部位连续离地超过 40 帧：${c.long_air.map(x=>x.part+' '+x.frames+' 帧').join('；')||'无'}</p><small>旧恢复验证结果仅对应原切片。裁剪建议尚未重新做原生恢复验证，不作为训练输入。</small>`;}
</script></html>'''
if __name__=='__main__':main()
