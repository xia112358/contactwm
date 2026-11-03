"""Render eight unexplained long recorded contact slices, without policy execution."""
from supermimic.paths import runtime_path, runtime_root
import json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
from supermimic.rotations import quat_to_mat

def main():
    torch.set_num_threads(4)
    web=runtime_path('results/mixed_comparison_web/unexplained-contact-videos-20261030')
    web.mkdir(parents=True,exist_ok=True)
    report=json.loads((runtime_path('results/mixed_comparison_web/unexplained-contact-20261030/data.json')).read_text())
    cases=report['cases'][:8]
    library=Path(report['source_library'])
    rec=torch.load(library/'rollout.pt',weights_only=False,map_location='cpu');d=rec['data'];meta=rec['metadata']
    spans={s['motion']:s for s in meta['recording_spans']}
    manifest=runtime_path('data/php296_surface_v2/task.json')
    cards=[];provenance=[]
    for c in cases:
        b,e=c['pooled_frames'];span=spans[c['motion']]
        begin=max(span['begin'],b-10);end=min(span['end'],e+10)
        states=torch.cat([d['s'][begin:end],d['s1'][end-1:end]])
        n=len(states);indices=torch.arange(begin,end+1).clamp_max(end-1)
        status=d['active'][indices]&d['known'][indices]
        labels=[];parts=['LF','RF','LH','RH','LK','RK']
        for i in range(n):
            frame=begin+i
            phase='SEGMENT' if b<=frame<e else ('CONTEXT BEFORE' if frame<b else 'CONTEXT AFTER / END')
            labels.append(f'{phase} | source frame {frame-span["begin"]} | relative {frame-b}/{e-b}\nContacts: '+','.join(p for j,p in enumerate(parts) if status[i,j]))
        arrays=dict(names=np.array([c['name']]),root_p=states[:,:3].numpy()[:,None],root_quat=states[:,3:7].numpy()[:,None],
            q=states[:,13:42].numpy()[:,None],f=np.arange(n)[:,None],alive=np.ones((n,1),bool),
            ref_root_p=states[:,:3].numpy(),ref_root_R=quat_to_mat(states[:,3:7]).numpy(),ref_q=states[:,13:42].numpy(),
            clip_start=np.array([0]),clip_len=np.array([n]),frame_annotations=np.array(labels),
            **{k:np.asarray(v) for k,v in meta.items() if isinstance(v,(str,int,float,bool))})
        stem=f'case_{c["case"]}';npz=web/(stem+'.npz');mp4=web/(stem+'.mp4');np.savez_compressed(npz,**arrays)
        rot=quat_to_mat(states[n//2,3:7]);yaw=math.degrees(math.atan2(float(rot[1,0]),float(rot[0,0])))
        cmd=[sys.executable,str(ROOT/'tools/render.py'),str(npz),str(mp4),'--task_manifest',str(manifest),
            '--hide_reference','--seconds',str(n*.02+.1),'--fps','50','--width','1280','--height','720',
            '--camera_azimuth',str(yaw+115),'--camera_elevation','-20','--camera_distance','3.3',
            '--title',f'Case {c["case"]} | {c["frames"]} frames | recorded teacher 10k']
        subprocess.run(cmd,cwd=runtime_root(),check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(mp4)]))['streams'][0]
        assert int(probe['nb_read_frames'])==n,(probe,n)
        info=dict(case=c['case'],name=c['name'],segment_frames=c['source_frames'],video_source_frames=[begin-span['begin'],end-span['begin']],
            segment_start_seconds=(b-begin)*.02,segment_end_seconds=(e-begin)*.02,video=mp4.name,frames=n,probe=probe,source=str(library/'rollout.pt'),
            source_checkpoint=meta['source_checkpoint'],playback='recorded native states; visualization only, no new simulation')
        provenance.append(info)
        cards.append(f'''<section><h2>#{c['case']} · {c['name']} · {c['frames']} 帧</h2><p>原段源帧 [{c['source_frames'][0]}, {c['source_frames'][1]})，视频保留前后最多 10 帧上下文。</p><video controls playsinline loop preload="metadata" src="{mp4.name}"></video><p><button onclick="jump(this,{info['segment_start_seconds']})">段起点</button><button onclick="jump(this,{max(info['segment_start_seconds'],info['segment_end_seconds']-.02)})">段终点</button><button onclick="step(this,-1)">前一帧</button><button onclick="step(this,1)">后一帧</button><a href="{mp4.name}">打开 / 下载视频</a></p></section>''')
        (web/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
        (web/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>待查长接触段视频</title><style>body{font:16px/1.6 system-ui;background:#101722;color:#e0e8f3;max-width:1200px;margin:auto;padding:24px}section{background:#1c2a3b;padding:18px;margin:24px 0;border-radius:12px}video{width:100%}button,select{font:inherit;background:#31475f;color:white;border:1px solid #7c94ad;padding:7px;margin:4px}a{color:#80d9ff}</style><h1>待查长接触段 · 实际记录回放</h1><p>仅展示原因尚未明确的最长 8 段。画面来自 10k tracking 教师的原始采集轨迹，不是后续单段策略的新执行。按原始机器人姿态和匹配地形渲染。</p><p>SEGMENT 是当前切片；CONTEXT 是段外上下文。接触缩写 LF/RF：左右脚，LH/RH：左右手，LK/RK：左右膝。</p><p>播放速度 <select id="rate" onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=+this.value)"><option value=".25">0.25×</option><option value=".5" selected>0.5×</option><option value="1">1×</option></select></p>'''+''.join(cards)+'''<p><a href="provenance.json">视频来源与帧范围</a></p><script>document.querySelectorAll('video').forEach(v=>{v.playbackRate=.5;v.addEventListener('play',()=>document.querySelectorAll('video').forEach(w=>{if(w!==v)w.pause()}))});function jump(b,t){let v=b.closest('section').querySelector('video');v.pause();v.currentTime=t;}function step(b,d){let v=b.closest('section').querySelector('video');v.pause();v.currentTime=Math.max(0,Math.min(v.duration-.02,v.currentTime+d*.02));}</script></html>''')
        print('READY',stem,flush=True)
    print('COMPLETE',web,flush=True)
if __name__=='__main__':main()
