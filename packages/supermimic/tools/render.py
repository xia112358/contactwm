"""Render recorded evaluation trajectories (supermimic/evaluate.py --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
"""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
# use the NVIDIA EGL vendor (GPU) rather than Mesa's CPU rasterizer when the image does not register it
_NV_EGL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker", "10_nvidia.json")
if os.path.exists(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", os.path.abspath(_NV_EGL))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = str(runtime_path("data", "g1_description", "g1_29dof.xml"))
MESH_DIR = str(runtime_path("data", "g1_description"))
TASK_SPEC = None


def two_robot_xml(goal_count=0, root_goal=False, measured_count=0) -> str:
    xml = open(MJCF).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + MESH_DIR + '/"', xml)
    body = xml[xml.index('<body name="pelvis"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.25"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    <geom name="contact_goal_marker" type="sphere" size="0.025" pos="0 0 -1000" rgba="1 0.42 0.05 0.9" contype="0" conaffinity="0"/>
    """
    for index in range(goal_count):
        scene += (f'<geom name="segment_goal_{index}" type="sphere" size="0.025" '
                  'pos="0 0 -1000" rgba="1 0.42 0.05 0.9" contype="0" conaffinity="0"/>')
    if root_goal:
        scene += ('<geom name="segment_root_goal" type="sphere" size="0.045" '
                  'pos="0 0 -1000" rgba="0.1 0.8 0.3 0.65" contype="0" conaffinity="0"/>')
    for index in range(measured_count):
        color = '0.05 0.8 1 1' if index == 0 else '0.9 0.15 0.85 1'
        scene += (f'<geom name="measured_contact_{index}" type="sphere" size="0.014" '
                  f'pos="0 0 -1000" rgba="{color}" contype="0" conaffinity="0"/>')
    if TASK_SPEC:
        for index in range(max(len(task["boxes"]) for task in TASK_SPEC["tasks"])):
            scene += (f'<geom name="task_box_{index}" type="box" size="1 1 1" '
                      'pos="0 0 -1000" rgba="0.48 0.55 0.62 1" contype="0" conaffinity="0"/>\n')
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", '<mujoco', 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", '<visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def main():
    global MJCF, MESH_DIR, TASK_SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--task_manifest", type=str, default=None)
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its global xy instead of anchored to the robot")
    ap.add_argument("--hide_reference", action="store_true", help="show only the simulated robot")
    ap.add_argument("--camera_azimuth", type=float, default=135.)
    ap.add_argument("--camera_elevation", type=float, default=-12.)
    ap.add_argument("--camera_distance", type=float, default=3.6)
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    if "task_schema" in d and not a.task_manifest:
        ap.error("This trajectory requires its --task_manifest and canonical robot")
    if a.task_manifest:
        import hashlib
        import json
        from pathlib import Path
        import sys
        task_path = Path(a.task_manifest).resolve()
        from supermimic.task_manifest import load_task_manifest
        TASK_SPEC = load_task_manifest(task_path)
        a.global_ghost = True
        if str(d.get("task_manifest_sha256", "")) != hashlib.sha256(task_path.read_bytes()).hexdigest():
            raise ValueError("Trajectory and task manifest disagree")
        pass
        from somaforge_core.robot_assets import decode_robot_asset_json
        decode_robot_asset_json(d["robot_asset_json"], context=a.traj)
        MJCF = TASK_SPEC["visualization_mjcf"]
        if hashlib.sha256(Path(MJCF).read_bytes()).hexdigest() != TASK_SPEC["visualization_mjcf_sha256"]:
            raise ValueError("Auxiliary visualization model changed")
        MESH_DIR = str(Path(TASK_SPEC["robot_asset"]["urdf_path"]).parent)
    names = list(d["names"])
    measured_count = d['measured_contact_points'].shape[1] if 'measured_contact_points' in d else 0
    model = mujoco.MjModel.from_xml_string(two_robot_xml(len(d.get('goal_points', [])), 'goal_root' in d, measured_count))
    for index, point in enumerate(d.get('goal_points', [])):
        model.geom_pos[model.geom(f'segment_goal_{index}').id] = point
    if 'goal_root' in d:
        model.geom_pos[model.geom('segment_root_goal').id] = d['goal_root']
    if TASK_SPEC:
        # Hidden box slots start far below the floor; exclude them from camera
        # clipping scale so nearby robot geometry remains visible.
        model.stat.extent = 5.0
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    data = mujoco.MjData(model)
    if a.hide_reference:
        for index in range(model.ngeom):
            if model.body(int(model.geom_bodyid[index])).name.startswith("ref_"):
                model.geom_rgba[index, 3] = 0.0
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = a.camera_distance, a.camera_azimuth, a.camera_elevation
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    clips = a.clips or names
    if a.max_clips is not None:
        if a.max_clips < 1:
            ap.error("--max_clips must be positive")
        clips = [clips[index] for index in np.linspace(0, len(clips) - 1, min(a.max_clips, len(clips)), dtype=int)]
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = int(round(50 / a.fps))
    for clip in clips:
        c = names.index(clip)
        if TASK_SPEC:
            task = next(task for task in TASK_SPEC["tasks"] if task["name"] == clip)
            for index in range(max(len(item["boxes"]) for item in TASK_SPEC["tasks"])):
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"task_box_{index}")
                # Slots compile with identity rotations. Runtime terrain poses
                # must bypass the compiler's cached BODYROT/IDENTITY shortcut.
                model.geom_sameframe[geom] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                model.geom_pos[geom] = [0, 0, -1000]
                if index < len(task["boxes"]):
                    box = np.asarray(task["boxes"][index])
                    model.geom_pos[geom] = box[:3]
                    model.geom_quat[geom] = box[3:7]
                    model.geom_size[geom] = box[7:] * 0.5
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = min(T, t0 + int(a.seconds * 50))
        fell_at = None
        for t in range(t0, t1, stride):
            if not alive[t] and fell_at is None:
                fell_at = t
            tt = t if alive[t] else (fell_at if fell_at is not None else t)
            f = int(d["f"][tt, c])
            data.qpos[0:3] = d["root_p"][tt, c]
            data.qpos[3:7] = d["root_quat"][tt, c]
            data.qpos[7:36] = d["q"][tt, c]
            data.qpos[36:39] = d["ref_root_p"] if d["ref_root_p"].ndim==1 else d["ref_root_p"][f]
            if not a.global_ghost:  # anchor the reference to the robot's xy, as the tracking reward does
                data.qpos[36:38] = d["root_p"][tt, c][:2]
            data.qpos[39:43] = _mat_to_quat(d["ref_root_R"] if d["ref_root_R"].ndim==2 else d["ref_root_R"][f])
            data.qpos[43:72] = d["ref_q"] if d["ref_q"].ndim==1 else d["ref_q"][f]
            if "goal_point" in d:
                marker=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"contact_goal_marker")
                model.geom_pos[marker]=d["goal_point"][tt,c]
            if 'goal_points_by_frame' in d:
                for index, point in enumerate(d['goal_points_by_frame'][tt]):
                    marker=model.geom(f'segment_goal_{index}').id
                    model.geom_sameframe[marker]=mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                    model.geom_pos[marker]=point
            if 'goal_root_by_frame' in d:
                marker=model.geom('segment_root_goal').id
                model.geom_sameframe[marker]=mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                model.geom_pos[marker]=d['goal_root_by_frame'][tt]
            for index in range(measured_count):
                marker = model.geom(f'measured_contact_{index}').id
                model.geom_sameframe[marker] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
                model.geom_pos[marker] = (d['measured_contact_points'][tt,index]
                    if d['measured_contact_active'][tt,index] else [0,0,-1000])
            mujoco.mj_kinematics(model, data)
            if TASK_SPEC and t == t0:
                for index, box in enumerate(task['boxes']):
                    geom = model.geom(f'task_box_{index}').id
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, np.asarray(box[3:7], dtype=np.float64))
                    if not (np.allclose(data.geom_xpos[geom], box[:3], atol=1e-6)
                            and np.allclose(data.geom_xmat[geom], rotation, atol=1e-6)
                            and np.allclose(model.geom_size[geom], np.asarray(box[7:])*.5, atol=1e-6)):
                        raise ValueError('Rendered box transform disagrees with task collision geometry')
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            label = f"{a.title}  {clip}  t={(f - int(d['clip_start'][c])) * 0.02:5.1f}s"
            if fell_at is not None:
                ended = f >= int(d["clip_start"][c] + d["clip_len"][c] - 2)
                label += "  [clip ended]" if ended else "  [terminated]"
            legend = ("simulated G1 policy | corrected terrain collisions | reference hidden" if a.hide_reference
                      else "solid: simulated G1 policy (Isaac Lab + MuJoCo-Warp)   blue ghost: reference")
            annotations = [((16, 12), label), ((16, a.height - 38), legend)]
            if 'frame_annotations' in d:
                annotations += [((16, 46 + 30 * index), line)
                                for index, line in enumerate(str(d['frame_annotations'][tt]).split('\n'))]
            for xy, txt in annotations:
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError('Video encoder failed')
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()
