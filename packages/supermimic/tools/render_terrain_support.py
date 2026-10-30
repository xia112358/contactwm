"""Two-view playback of actual solver collision model/qpos; no reference ghost."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
vendor = Path(__file__).resolve().parents[1] / "docker/10_nvidia.json"
if vendor.exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("record")
p.add_argument("out")
p.add_argument("--task_ids", nargs='+', type=int, default=[13, 30, 53])
p.add_argument("--azimuth_offsets", nargs=2, type=float, default=[90, 140])
args = p.parse_args()
d = np.load(args.record + ".npz")
spec = json.loads(Path(str(d["task_manifest"])).read_text())
model = mujoco.MjModel.from_binary_path(args.record + ".mjb")
model.stat.extent = 5
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480
model.vis.headlight.ambient[:] = .6
model.vis.headlight.diffuse[:] = .6
for geom in range(model.ngeom):
    name = model.geom(geom).name
    model.geom_rgba[geom] = [.77, .8, .84, 1]
    if "Terrain_" in name:
        model.geom_rgba[geom] = [.33, .43, .57, 1]
        if "/box_0_" in name:
            model.geom_rgba[geom] = [.82, .84, .87, 1]
    elif "ankle" in name:
        model.geom_rgba[geom] = [.95, .47, .08, 1]
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)
font = ImageFont.load_default(size=19)
camera = mujoco.MjvCamera()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", "1280x480", "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                       "yuv420p", "-crf", "19", args.out], stdin=subprocess.PIPE)
for task_id in args.task_ids:
    world = int(np.flatnonzero(d["task_ids"] == task_id)[0])
    task = spec["tasks"][task_id]
    box = np.array(task["boxes"][0])
    for name in ("geom_pos", "geom_size", "geom_quat"):
        getattr(model, name)[:] = d[name][world % d[name].shape[0]]
    model.geom_sameframe[:] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    yaw = np.degrees(2*np.arctan2(box[6], box[3]))
    for t in range(0, len(d["qpos"]), 2):
        data.qpos[:] = d["qpos"][t, world]
        mujoco.mj_forward(model, data)
        image = Image.new("RGB", (1280, 480), "white")
        for side, angle in enumerate(args.azimuth_offsets):
            camera.lookat[:] = [box[0], box[1], .75]
            camera.distance = 3.25
            camera.elevation = -12 if side == 0 else -25
            camera.azimuth = yaw + angle
            renderer.update_scene(data, camera=camera)
            image.paste(Image.fromarray(renderer.render()), (640*side, 0))
        draw = ImageDraw.Draw(image)
        footer = (f'Foot normal force: {d["forces"][t,world]:.0f} N | ' if 'forces' in d else 'Recorded pose replay | ')
        labels = [(12, 10, f'{task["name"]}  t={t*.02+float(d.get("time_offset", .02)):.2f}s | actual collision shapes | 0.5x speed'),
                  (12, 450, footer+'orange: ankle/foot colliders | NO reference ghost'),
                  (12, 38, 'SIDE VIEW'), (652, 38, 'PERSPECTIVE')]
        for x, y, text in labels:
            bounds = draw.textbbox((x,y),text,font=font)
            draw.rectangle((bounds[0]-3,bounds[1]-2,bounds[2]+3,bounds[3]+2),fill='white')
            draw.text((x,y),text,font=font,fill='black')
        raw = np.asarray(image).tobytes()
        ff.stdin.write(raw)
        ff.stdin.write(raw)
ff.stdin.close()
if ff.wait():
    raise RuntimeError("ffmpeg encoding failed")
renderer.close()
print("wrote", args.out)
