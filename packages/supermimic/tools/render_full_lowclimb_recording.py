"""Publish a complete existing native tracking capture without rerunning training."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,json,math,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];pass
from supermimic.rotations import quat_to_mat

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recording',type=Path,default=runtime_path('logs/php_low_speed_10k_extract_20261029_v1/capture_00/motion_246/rollout.pt'))
    parser.add_argument('--publish_name',default='full-lowclimb-0030-20261030')
    parser.add_argument('--allow_partial',action='store_true')
    parser.add_argument('--title',default='Full low climb | recorded teacher 10k')
    args=parser.parse_args()
    if Path(args.publish_name).name!=args.publish_name:raise ValueError('Publish name must be a directory name')
    torch.set_num_threads(4)
    web=runtime_path('results/mixed_comparison_web')/args.publish_name;web.mkdir(exist_ok=True)
    source=args.recording
    saved=torch.load(source,map_location='cpu',weights_only=False);d=saved['data'];meta=saved['metadata']
    if not args.allow_partial:assert not bool(d['term'].any())
    if 'successful_rollout' in meta:assert meta['successful_rollout']
    states=torch.cat([d['s'][:1],d['s1']]);n=len(states)
    if not args.allow_partial:assert len(d['s'])==821, 'This renderer is for the complete lowclimb_0030 capture'
    active=torch.cat([d['active'][:1]&d['known'][:1],d['active']&d['known']]);parts=['LF','RF','LH','RH','LK','RK']
    notes=[f'Full trajectory | frame {i}/{n-1} | Contacts: '+','.join(p for j,p in enumerate(parts) if active[i,j]) for i in range(n)]
    data=dict(names=np.array(['lowclimb_0030']),root_p=states[:,:3].numpy()[:,None],root_quat=states[:,3:7].numpy()[:,None],q=states[:,13:42].numpy()[:,None],
        f=np.arange(n)[:,None],alive=np.ones((n,1),bool),ref_root_p=states[:,:3].numpy(),ref_root_R=quat_to_mat(states[:,3:7]).numpy(),ref_q=states[:,13:42].numpy(),
        clip_start=np.array([0]),clip_len=np.array([n]),frame_annotations=np.array(notes),
        **{k:np.asarray(v) for k,v in meta.items() if isinstance(v,(str,int,float,bool))})
    data['alive'][-1,0]=not bool(d['term'][-1])
    data['clip_len']=np.array([822])
    npz=web/'lowclimb_0030.npz';mp4=web/'lowclimb_0030.mp4';np.savez_compressed(npz,**data)
    r=quat_to_mat(states[n//2,3:7]);yaw=math.degrees(math.atan2(float(r[1,0]),float(r[0,0])))
    cmd=[sys.executable,str(ROOT/'tools/render.py'),str(npz),str(mp4),'--task_manifest',str(runtime_path('data/php296_surface_v2/task.json')),'--hide_reference',
        '--seconds',str(n*.02+.1),'--fps','50','--width','1280','--height','720','--camera_azimuth',str(yaw+110),'--camera_elevation','-18','--camera_distance','3.8',
        '--title',args.title]
    subprocess.run(cmd,cwd=runtime_root(),check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(mp4)]))['streams'][0]
    assert int(probe['nb_read_frames'])==n
    (web/'provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),contact_response=meta.get('contact_response_diagnostic','legacy'),checkpoint=meta['source_checkpoint'],motion=246,name='lowclimb_0030',control_steps=len(d['s']),video_frames=n,control_duration_seconds=len(d['s'])*.02,successful_original_rollout=not bool(d['term'].any()) and len(d['s'])==821,training_restarted=False,probe=probe,semantics='Complete saved native teacher execution; state playback only, no new simulation'),indent=2))
    (web/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整攀爬轨迹 lowclimb_0030</title><style>body{max-width:1280px;margin:auto;padding:24px;background:#101722;color:#e0e8f3;font:16px/1.7 system-ui}video{width:100%;background:black}button,select{font:inherit;padding:8px;margin:6px;background:#30445c;color:white;border:1px solid #7089a4}a{color:#80d9ff}</style><h1>完整攀爬轨迹 · lowclimb_0030</h1><p>10k tracking 教师保存的完整实际执行，821 个控制步，16.42 秒。包含原始开头和末尾，未裁剪；刚才 #1378 对应源帧 709–821。</p><video id="v" controls playsinline preload="metadata" src="lowclimb_0030.mp4"></video><p><button onclick="v.currentTime=0;v.play()">从头播放</button><button onclick="v.currentTime=14.18;v.play()">末尾站立段</button><button onclick="v.pause();v.currentTime=Math.max(0,v.currentTime-.02)">前一帧</button><button onclick="v.pause();v.currentTime=Math.min(v.duration-.02,v.currentTime+.02)">后一帧</button><select onchange="v.playbackRate=+this.value"><option value="1">1× 正常速度</option><option value=".5">0.5×</option><option value=".25">0.25×</option></select></p><p><a href="lowclimb_0030.mp4">打开 / 下载 MP4</a> · <a href="provenance.json">轨迹来源</a></p><script>const v=document.getElementById('v');</script></html>''')
    if meta.get('contact_response_diagnostic'):
        page=(web/'index.html').read_text().replace('10k tracking 教师保存的完整实际执行，', '修正接触响应后，冻结原 10k tracking 教师重新执行的完整轨迹（未重新训练），')
        page=page.replace('刚才 #1378 对应源帧 709–821。','物理版本 '+meta['contact_response_diagnostic']+'；不能视为重新训练后的策略。')
        page=page.replace('<video id=', '<p><a href="../full-lowclimb-0030-20261030/">对照：修正前完整视频</a></p><video id=')
        (web/'index.html').write_text(page)
    if args.allow_partial:
        page=(web/'index.html').read_text().replace('完整攀爬轨迹','修正物理后的攀爬尝试').replace('完整轨迹（未重新训练）','轨迹（未重新训练）')
        page=page.replace('821 个控制步，16.42 秒。包含原始开头和末尾，未裁剪；', f'{len(d["s"])} / 821 个控制步，{len(d["s"])*.02:.2f} 秒；已触发终止，未完成整段。')
        page=page.replace('<video id=', '<p><b>这是旧策略迁移到修正物理后的失败回放，不能当成通过验证的新策略。</b></p><video id=')
        page=page.replace('<button onclick="v.currentTime=14.18;v.play()">末尾站立段</button>', '')
        (web/'index.html').write_text(page)
    print('COMPLETE',n,web,flush=True)
if __name__=='__main__':main()
