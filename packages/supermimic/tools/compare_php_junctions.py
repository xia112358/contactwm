"""Diagnostic playback of existing PHP joins; never exports training slices.

Recover source phase by fitting the settled joint trajectory to the untouched
60 Hz database. The fitted phase is an estimate, not a recorded matcher index.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

from supermimic.paths import runtime_root, package_path, workspace_root

ROOT = runtime_root()
OUT = workspace_root() / 'tmp/php-junctions-20261030'


def interp(x, values):
    return np.stack([np.interp(x, np.arange(len(values)), v) for v in values.T], -1)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    original = ROOT / 'data/php_originals_v1'
    axes = runpy.run_path(str(original / 'upstream/motion_matching/motion_matching/metadata/constants.py'))['G1_JOINT_AXES']
    source_q = []
    for k in range(2):
        r = np.load(original / f'original_clips/climb_76/range_{k:03}.npz')['bone_rotations'][:, 2:]
        aa = Rotation.from_quat(r.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(-1, 29, 3)
        source_q.append((aa * axes).sum(-1))
    spec = json.loads((ROOT / 'data/php296_surface_v2/task.json').read_text())
    manifest = Path(spec['somaforge_root']) / 'runtime/current/motions/php_wbt_v1_20261016/motions-low-climb-76/manifest.json'
    entries = json.loads(manifest.read_text())['motion_files']
    tasks = [t for t in spec['tasks'] if t['group'] == 'lowclimb']
    cases = []
    for task, entry in zip(tasks, entries):
        assert digest(task['source_motion']) == task['source_motion_sha256']
        name = Path(entry['source_file']).stem.removesuffix('_motion')
        duration = float(re.search(r'walk([\d.]+)s', name)[1])
        trigger = round((2.4 + duration) * 50)
        data = np.load(task['source_motion'])
        qpos = data['joint_pos'].copy()
        # Current skill schedule: 1s stand + 1s walk + named walk + .4s stop.
        # Independently verify its phase prediction against the settled core.
        fit_frames = np.arange(trigger + 50, trigger + 130)
        best = None
        for mirror, reference in enumerate(source_q):
            def loss(offset):
                expected = interp(offset + (fit_frames - trigger) * 1.2, reference)
                return float(np.mean((qpos[fit_frames, 7:] - expected) ** 2))
            grid = np.arange(68, 98, .2)
            center = grid[np.argmin([loss(x) for x in grid])]
            fit = minimize_scalar(loss, bounds=(center - .21, center + .21), method='bounded')
            if best is None or fit.fun < best[0]:
                best = (fit.fun, mirror, float(fit.x))
        error, mirror, offset = best
        assert np.sqrt(error) < .003, (name, best)
        times = np.arange(-30, 151) / 50
        frames = trigger + np.arange(-30, 151)
        expected = interp(offset + times * 60, source_q[mirror])
        delta = np.rad2deg(qpos[frames, 7:] - expected)
        rms = np.sqrt(np.mean(delta ** 2, axis=1))
        samples = {str(t): float(rms[np.argmin(abs(times-t))]) for t in [0,.1,.2,.4,.6,1,2]}
        # Static alignment at original source frame 200, not per-frame alignment.
        anchor = trigger + (200 - offset) / 1.2
        anchor_pos = interp(np.array([anchor]), qpos[:, :3])[0]
        rots = Rotation.from_quat(qpos[:, [4,5,6,3]])
        anchor_rot = Slerp(np.arange(len(qpos)), rots)([anchor]).as_matrix()[0]
        yaw = np.arctan2(anchor_rot[1,0], anchor_rot[0,0])
        align = Rotation.from_euler('z', -yaw)
        origin = np.array([anchor_pos[0], anchor_pos[1], 0.])
        aligned = qpos[frames].copy()
        aligned[:, :3] = align.apply(aligned[:, :3] - origin)
        aligned[:, 3:7] = (align * rots[frames]).as_quat()[:, [3,0,1,2]]
        boxes = np.asarray(task['boxes']).copy()
        boxes[:, :3] = align.apply(boxes[:, :3] - origin)
        boxes[:, 3:7] = (align * Rotation.from_quat(boxes[:, [4,5,6,3]])).as_quat()[:, [3,0,1,2]]
        info = dict(name=task['name'], scenario=name, trigger_frame_estimate=trigger,
                    fitted_source_phase_at_trigger=offset, mirror=mirror,
                    settled_fit_rms_deg=float(np.rad2deg(np.sqrt(error))),
                    joint_rms_deg_at_seconds=samples, source=task['source_motion'],
                    source_sha256=task['source_motion_sha256'], source_frames=[int(frames[0]), int(frames[-1])+1])
        cases.append(dict(info=info, aligned=aligned, boxes=boxes, expected=expected,
                          rms=rms, qpos=qpos, trigger=trigger, offset=offset, mirror=mirror))
    return spec, cases, times


def render(spec, cases, times):
    os.environ.setdefault('MUJOCO_GL','egl')
    sys.path.insert(0, str(package_path('tools')))
    import render as renderer_tools
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    renderer_tools.MJCF = spec['visualization_mjcf']
    assert digest(renderer_tools.MJCF) == spec['visualization_mjcf_sha256']
    renderer_tools.MESH_DIR = str(Path(spec['robot_asset']['urdf_path']).parent)
    renderer_tools.TASK_SPEC = {'tasks':[{'boxes':c['boxes']} for c in cases]}
    model = mujoco.MjModel.from_xml_string(renderer_tools.two_robot_xml())
    model.stat.extent = 5
    model.vis.headlight.ambient[:] = .55
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 450, 800)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.8, 125, -18
    camera.lookat[:] = [-.25,0,.85]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    pipe = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900','-r','50','-i','-',
                             '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'comparison.mp4')],stdin=subprocess.PIPE)
    try:
        for frame,time in enumerate(times):
            canvas = Image.new('RGB',(1600,900))
            for panel,c in enumerate(cases):
                for i in range(max(len(x['boxes']) for x in cases)):
                    gid=model.geom(f'task_box_{i}').id
                    model.geom_sameframe[gid]=mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                    model.geom_pos[gid]=[0,0,-1000]
                    if i<len(c['boxes']):
                        b=c['boxes'][i];model.geom_pos[gid]=b[:3];model.geom_quat[gid]=b[3:7];model.geom_size[gid]=b[7:]/2
                data.qpos[:36] = c['aligned'][frame]
                data.qpos[36:72] = c['aligned'][frame]
                data.qpos[43:72] = c['expected'][frame]
                # Ghost compares joint deformation only, sharing the actual root.
                for gid in range(model.ngeom):
                    if model.body(int(model.geom_bodyid[gid])).name.startswith('ref_'):
                        model.geom_rgba[gid,3] = .23 if time>=0 else 0
                mujoco.mj_kinematics(model,data)
                renderer.update_scene(data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                lines=[c['info']['name']+' | '+c['info']['scenario'],
                       f'trigger {time:+.2f}s | source phase {c["offset"]+60*time:.1f} | mirror {c["mirror"]}',
                       f'Joint RMS vs source: {c["rms"][frame]:.2f} deg' if time>=0 else 'Approach context (before skill trigger)']
                for j,line in enumerate(lines):
                    draw.rectangle((7,7+j*24,793,30+j*24),fill='white');draw.text((12,9+j*24),line,font=font,fill='black')
                draw.rectangle((0,420,800,450),fill='white')
                draw.text((10,425),'Reference playback | blue: source joints, same root | NO physics',font=font,fill='black')
                canvas.paste(image,((panel%2)*800,(panel//2)*450))
            if frame in [30,40,60,130]:canvas.save(OUT/f'frame_{frame:03}.jpg')
            pipe.stdin.write(np.asarray(canvas).tobytes())
        pipe.stdin.close()
        assert pipe.wait()==0
    finally:
        renderer.close()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,duration','-of','json',str(OUT/'comparison.mp4')]))
    assert int(probe['streams'][0]['nb_read_frames'])==len(times)
    return probe


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec,all_cases,times=prepare()
    # Same source/similar phase, earlier source entry, and mirror representative.
    chosen=[all_cases[i] for i in [0,18,26,20]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for c in chosen:ax.plot(times[times>=0],c['rms'][times>=0],label=c['info']['name'])
    ax.set(xlabel='Seconds after scheduled skill trigger',ylabel='29-joint RMS vs fitted original (degrees)',ylim=(0,None))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'joint_difference.png',dpi=160);plt.close(fig)
    baseline=all_cases[0]
    pair=[]
    for c in chosen[1:]:
        if c['mirror']!=baseline['mirror']:continue
        record=dict(name=c['info']['name'])
        record['trigger_aligned_rms_deg']={str(t):float(np.rad2deg(np.sqrt(np.mean(
            (c['qpos'][c['trigger']+round(t*50),7:]-baseline['qpos'][baseline['trigger']+round(t*50),7:])**2))))
            for t in [0,.2,.4,.6,1,2]}
        for start,end in [(100,130),(130,160),(160,220)]:
            phase=np.arange(start,end,.6)
            a=interp(baseline['trigger']+(phase-baseline['offset'])/1.2,baseline['qpos'][:,7:])
            b=interp(c['trigger']+(phase-c['offset'])/1.2,c['qpos'][:,7:])
            record[f'phase_{start}_{end}_rms_deg']=float(np.rad2deg(np.sqrt(np.mean((a-b)**2))))
        pair.append(record)
    report=dict(diagnostic_only=True,new_training_slices=False,physical_contact_timing_available=False,
                alignment='static xy/yaw at fitted original frame 200; no mirroring of visual examples',
                phase_estimation='schedule-derived trigger; original 60Hz phase least-squares fitted at +1 to +2.6s, not an exact logged matcher index',
                ghost='Original joint angles at fitted source phase, anchored to composed root; excludes root differences',
                selected=[c['info'] for c in chosen],all40=[c['info'] for c in all_cases],same_source_phase_pairwise=pair)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(selected=report['selected'],pairs=pair),indent=2),flush=True)
    report['video_probe']=render(spec,chosen,times)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    rows=''.join('<tr><td>'+c['info']['scenario']+'</td><td>'+str(c['mirror'])+'</td><td>'+f'{c["offset"]:.2f}'+'</td>'+''.join(f'<td>{c["info"]["joint_rms_deg_at_seconds"][str(t)]:.2f}°</td>' for t in [0,.2,.4,1,2])+'</tr>' for c in chosen)
    pair_rows=''.join(f'<tr><td>{p["name"]} 对 0000</td><td>{p["trigger_aligned_rms_deg"]["0"]:.2f}°</td><td>{p["trigger_aligned_rms_deg"]["0.4"]:.2f}°</td><td>{p["trigger_aligned_rms_deg"]["2"]:.2f}°</td><td>{p["phase_130_160_rms_deg"]:.3f}°</td><td>{p["phase_160_220_rms_deg"]:.3f}°</td></tr>' for p in pair)
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PHP 拼接衔接差异</title><style>body{background:#111827;color:#e5e7eb;font:16px/1.7 system-ui;max-width:1600px;margin:auto;padding:24px}video,img{width:100%}button,select{font:inherit;padding:8px;margin:4px}a{color:#7dd3fc}td,th{padding:8px;border-bottom:1px solid #475569}table{border-collapse:collapse}small{color:#bdc7d6}</style>
<h1>PHP 低速攀爬：四个拼接入口的实际参考轨迹</h1><p>四画面同步：直行 / 左转后进入 / 提前进入原片段 / 镜像入口。画面按技能触发时刻同步（0 秒）；统一在原始第 200 帧处的根部 XY 与朝向，保留相对地形。不是策略执行，也没有新增训练数据。</p>
<video id="v" controls playsinline loop src="comparison.mp4"></video><p><button onclick="v.currentTime=0;v.play()">从准备段播放</button><button onclick="v.pause();v.currentTime=.6">技能触发</button><button onclick="v.pause();v.currentTime=Math.max(0,v.currentTime-.02)">前一帧</button><button onclick="v.pause();v.currentTime+=.02">后一帧</button><select onchange="v.playbackRate=+this.value"><option value=".25" selected>0.25×</option><option value=".5">0.5×</option><option value="1">1×</option></select><input id="seek" type="range" min="0" max="180" step="1" value="0" oninput="v.pause();v.currentTime=this.value/50" style="width:40%"></p>
<p>实体为已有组合参考动作。蓝色半透明模型使用对应原始帧的关节角，根部跟随实体，只展示关节形变，不能据此判断根部轨迹误差。镜像样例与镜像原片段比较，避免把左右互换算成新动作。</p>
<h2>相对各自原始片段的 29 关节 RMS 差异</h2><p>入口帧由场景时长推算；原始相位在触发后 1–2.6 秒拟合，数字是估计相位，不是保存的 matcher 日志。插值和导出误差也包含在残差内。</p>
<div style="overflow:auto"><table><tr><th>场景</th><th>镜像</th><th>触发处拟合相位</th><th>0s</th><th>0.2s</th><th>0.4s</th><th>1s</th><th>2s</th></tr>'''+rows+'''</table></div><img src="joint_difference.png"><p><b>接触说明：</b>这些文件没有原生解算接触力和接触时刻；这里不把脚靠近地面当成实际接触。原始脚接触提示也不能证明拼接后的接触顺序。</p><p><a href="report.json">全部 40 条扫描结果、来源哈希及相位对齐统计</a> · <a href="comparison.mp4">下载四画面对比视频</a></p><script>v.playbackRate=.25;v.ontimeupdate=()=>seek.value=Math.round(v.currentTime*50);</script></html>''')
    page=(OUT/'index.html').read_text()
    pair_html='<h2>组合样例彼此有多不同？</h2><p>以下均为非镜像，和左上角 0000 比。按触发时刻对齐混合了相位差；按原始帧对齐后，后半段几乎重合。上表的约 20° 是相对原始片段的形变，不是样例之间的差异。</p><div style="overflow:auto"><table><tr><th>样例对</th><th>触发 0s</th><th>触发 +0.4s</th><th>触发 +2s</th><th>原片段帧 130–160</th><th>原片段帧 160–220</th></tr>'+pair_rows+'</table></div>'
    (OUT/'index.html').write_text(page.replace('<img src="joint_difference.png">','<img src="joint_difference.png">'+pair_html))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
