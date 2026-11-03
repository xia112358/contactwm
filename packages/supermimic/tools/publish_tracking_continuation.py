"""Render the recorded tracking continuation around one contact segment."""
from supermimic.paths import runtime_root
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--task_manifest', type=Path, required=True)
    parser.add_argument('--segment', type=int, required=True)
    parser.add_argument('--lead_frames', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from supermimic.contact_control import ContactClock
    from supermimic.rotations import quat_to_mat
    torch.set_num_threads(4)
    rec = torch.load(args.recording, map_location='cpu', weights_only=False)
    d = rec['data']
    clock = ContactClock(d, hold=1, role_aware=True)
    start, endpoint = clock.bounds[args.segment-1:args.segment+1]
    begin = max(1, start-args.lead_frames)
    end = len(d['s1'])
    assert torch.allclose(d['s1'][:-1], d['s'][1:], atol=1e-5, rtol=0), 'Recording has a discontinuity'
    assert not bool(d['done'][:-1].any()), 'Continuation crosses an episode reset'
    states = torch.cat((d['s'][:1], d['s1']))
    trace = states[begin:end+1]
    frames = torch.arange(begin, end+1)
    evidence = frames-1
    observed = d['known'][evidence,:2] & d['active'][evidence,:2]
    points = d['point'][evidence,:2]
    target = d['point'][endpoint-1,:2]
    errors = (points-target).norm(dim=-1)*100
    annotations = []
    for i, frame in enumerate(frames.tolist()):
        stage = (f'Before segment {args.segment}' if frame < start else
                 f'Segment {args.segment} | step {frame-start}/{endpoint-start}' if frame < endpoint else
                 f'After segment {args.segment} endpoint: +{(frame-endpoint)*.02:.2f}s')
        contact = ['L' if bool(observed[i,0]) else '-', 'R' if bool(observed[i,1]) else '-']
        annotations.append(f'Global frame {frame} | {stage}\nRecorded foot contacts: {" / ".join(contact)} | orange: fixed segment {args.segment} targets')
    root0 = states[endpoint,:3]
    samples = []
    for offset in [0,1,5,10,25,50,100,200,end-endpoint]:
        frame = endpoint+offset
        if frame > end:
            continue
        i = frame-begin
        samples.append(dict(frames_after_endpoint=offset,seconds_after_endpoint=offset*.02,
                            root_displacement_cm=((states[frame,:3]-root0)*100).tolist(),
                            root_velocity_m_s=states[frame,7:10].tolist(),
                            contact_active=observed[i].tolist(),
                            contact_distance_to_segment12_target_cm=[float(errors[i,j]) if observed[i,j] else None for j in range(2)]))
    report = dict(recording=str(args.recording.resolve()),source_checkpoint=rec.get('report',{}).get('source_checkpoint'),
                  scope='Original tracking policy native rollout, continuous saved states. No contact-policy rollout or frozen endpoint extension.',
                  segment=args.segment,segment_start_frame=start,segment_end_frame=endpoint,
                  video_start_frame=begin,video_end_frame=end,video_fps=50,
                  endpoint_video_time_s=(endpoint-begin)*.02,following_duration_s=(end-endpoint)*.02,
                  contact_marker_note='Recorded native contact points may change as the contact patch changes; this is not whole-foot slip.',samples=samples)
    source_report = args.recording.parent/'report.json'
    if source_report.exists():
        report['source_checkpoint'] = json.loads(source_report.read_text()).get('source_checkpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    npz = args.output/'tracking.npz'
    np.savez_compressed(npz,names=np.array(['lowclimb_0000']),root_p=trace[:,:3].numpy()[:,None],
                        root_quat=trace[:,3:7].numpy()[:,None],q=trace[:,13:42].numpy()[:,None],
                        alive=np.ones((len(trace),1),bool),f=np.arange(len(trace))[:,None],
                        ref_root_p=trace[:,:3].numpy(),ref_root_R=quat_to_mat(trace[:,3:7]).numpy(),ref_q=trace[:,13:42].numpy(),
                        clip_start=np.array([0]),clip_len=np.array([len(trace)]),goal_points=target.numpy(),goal_root=root0.numpy(),
                        measured_contact_points=points.numpy(),measured_contact_active=observed.numpy(),frame_annotations=np.array(annotations),
                        **{k:np.asarray(v) for k,v in rec['metadata'].items() if isinstance(v,(str,int,float,bool))})
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for view,azimuth in [('main',135),('side',225)]:
        subprocess.run([sys.executable,str(ROOT/'tools/render.py'),str(npz),str(args.output/f'{view}.mp4'),
                        '--task_manifest',str(args.task_manifest),'--seconds','20','--hide_reference','--fps','50',
                        '--width','1280','--height','720','--camera_azimuth',str(azimuth),'--camera_distance','3.0',
                        '--title',f'Original tracking | segment {args.segment} and continuation'],check=True,cwd=runtime_root())
    endpoint_time=(endpoint-begin)*.02
    start_time=(start-begin)*.02
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tracking 第{args.segment}段及后续</title>
<style>body{{background:#111925;color:#eee;font:16px/1.7 system-ui;max-width:1380px;margin:24px auto;padding:0 18px}}.videos{{display:flex;gap:16px;flex-wrap:wrap}}section{{flex:1;min-width:340px;background:#203047;padding:12px;border-radius:10px}}video{{width:100%}}a{{color:#9cf}}button,select{{font:inherit;padding:6px;margin:5px}}#position{{padding:10px;background:#203047}}</style>
<h1>原始 Tracking：第{args.segment}段落地及后续动作</h1>
<p>来自原始 tracking 策略的连续真实执行。保留第{args.segment}段开始前 {start-begin} 帧，经过该段终点后继续播放 {(end-endpoint)*.02:.2f} 秒，直到原录像结束。</p>
<p>第{args.segment}段终点位于视频 <b>{endpoint_time:.2f}s</b>。橙色球固定标记这一段的接触目标，绿色球为该段 root 目标；后续动作仍继续播放，不在终点停住。青色/紫色为记录的左右脚接触点，其变化不等于整只脚滑移。</p>
<p><button onclick="playFrom(0)">从头播放</button><button onclick="playFrom({start_time})">第{args.segment}段开始</button><button onclick="seek({endpoint_time})">停在第{args.segment}段终点</button><button onclick="playFrom({endpoint_time})">播放后续</button><button onclick="step(-1)">上一帧</button><button onclick="step(1)">下一帧</button><select onchange="speed=+this.value;videos.forEach(v=>v.playbackRate=speed)"><option value="0.5">0.5×</option><option value="0.25">0.25×</option><option value="1">1×</option></select></p>
<p id="position">第{args.segment}段终点：视频 {endpoint_time:.2f}s</p><div class="videos"><section><h2>主视角</h2><video controls playsinline preload="auto" src="main.mp4"></video></section><section><h2>另一侧视角</h2><video controls playsinline preload="auto" src="side.mp4"></video></section></div>
<p><a href="../segment12-late-training/">返回学习策略的第12段失败录像</a> · <a href="main.mp4">下载主视角</a> · <a href="report.json">帧范围与后续状态</a></p>
<script>const videos=[...document.querySelectorAll('video')];let speed=.5;videos.forEach(v=>v.playbackRate=speed);function seek(t){{videos.forEach(v=>{{v.pause();v.currentTime=Math.max(0,Math.min(t,v.duration-.02))}})}}function playFrom(t){{videos.forEach(v=>{{v.currentTime=t;v.playbackRate=speed;v.play()}})}}function step(d){{seek(Math.round(videos[0].currentTime*50)/50+d/50)}}videos[0].addEventListener('timeupdate',()=>{{let dt=videos[0].currentTime-{endpoint_time};document.getElementById('position').textContent=`当前视频 ${{videos[0].currentTime.toFixed(2)}}s；距第{args.segment}段终点 ${{dt>=0?'+':''}}${{dt.toFixed(2)}}s`}});</script></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
