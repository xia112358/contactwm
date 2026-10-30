"""Direct paired drop test of actual low-step support surfaces (no policy)."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import sys
pass
from isaaclab.app import add_launcher_args, launch_simulation
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task_manifest", default="data/lowstep_boxes/task.json")
p.add_argument("--output", default=str(runtime_path("tmp", "lowstep_support.json")))
p.add_argument("--inspect_only", action="store_true")
p.add_argument("--steps", type=int, default=75)
p.add_argument("--record", default=None, help="save actual solver model, per-world geometry and simulated qpos")
add_launcher_args(p)
args = p.parse_args()
import torch
from supermimic.lowstep import install
spec = install(args.task_manifest)
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP
from supermimic.robot import default_joint_pos_list
from supermimic.env import TrackingSim, make_sim

dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(spec["motion_dir"], kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
cfg = make_sim(dev, args)
with launch_simulation(cfg, args), torch.no_grad():
    n = 2 * len(spec["tasks"])
    env = TrackingSim(cfg, n, motion, mdp, eval_mode=True)
    if args.inspect_only:
        solver = env.sim.backend_manager._solver
        arrays = {name: getattr(solver.mjw_model, name).numpy()
                  for name in ("geom_pos", "geom_size", "geom_quat", "geom_dataid")}
        geoms = [(i, solver.mj_model.geom(i).name) for i in range(solver.mj_model.ngeom)
                 if "Terrain_" in solver.mj_model.geom(i).name]
        import re
        import numpy as np
        report = {"array_shapes": {k:list(v.shape) for k,v in arrays.items()},
                  "terrain_geoms": geoms, "tasks": []}
        slots = {int(re.search(r"/box_(\d+)", name).group(1)): i for i,name in geoms}
        for task_id, task in enumerate(spec["tasks"]):
            world = int((env.task_ids == task_id).nonzero()[0].item())
            checks = []
            for j, box in enumerate(task["boxes"], 1):
                index = slots[j]
                pos = arrays["geom_pos"][world % arrays["geom_pos"].shape[0], index]
                size = arrays["geom_size"][world % arrays["geom_size"].shape[0], index]
                quat = arrays["geom_quat"][world % arrays["geom_quat"].shape[0], index]
                error = float(np.max(np.abs(pos - box[:3])))
                size_error = float(np.max(np.abs(size - np.array(box[7:])*.5)))
                orientation_error = float(1-abs(np.dot(quat,box[3:7])))
                checks.append({"position_error": error, "size_error": size_error,
                               "orientation_error": orientation_error,
                               "collision_type": int(solver.mj_model.geom_contype[index]),
                               "collision_affinity": int(solver.mj_model.geom_conaffinity[index]),
                               "actual_position": pos.tolist()})
            report["tasks"].append({"task":task["name"],"checks":checks})
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != "tasks"}),flush=True)
        print("saved",args.output,flush=True)
        sys.exit(0)
    ids = torch.arange(n, device=dev)
    boxes = torch.tensor([t["boxes"][0] for t in spec["tasks"]], device=dev)
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 6] = 1
    pose[::2, 3:6] = boxes[:, 4:7]
    pose[::2, 6] = boxes[:, 3]
    pose[1::2, 3:7] = pose[::2, 3:7]
    pose[::2, :2] = boxes[:, :2]
    pose[::2, 2] = boxes[:, 2] + boxes[:, 9] * .5 + .85
    pose[1::2, 2] = .85
    q = env.default_q.expand(n, -1).clone()
    env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    env.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
    env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
    env.robot.set_joint_position_target_index(target=q)
    env.scene.reset(ids)
    env.refresh()
    root_heights, forces, root_positions, recorded_qpos = [], [], [], []
    if args.record:
        import mujoco
        solver = env.sim.backend_manager._solver
        mujoco.mj_saveModel(solver.mj_model, args.record + ".mjb", None)
    for step in range(args.steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(.02)
        position = env.state()[0].clone()
        root_positions.append(position)
        root_heights.append(position[:, 2])
        force = torch.stack([sum(sensor.data.net_normal_forces_w.torch.sum(1) for sensor in group).norm(dim=-1)
                             for group in env.contact_sensors], -1).sum(-1)
        forces.append(force)
        if args.record:
            recorded_qpos.append(solver.mjw_data.qpos.numpy().copy())
    heights = torch.stack(root_heights)
    forces = torch.stack(forces)
    # Early settling window avoids later balance falls influencing support diagnosis.
    settled = heights[10:25].mean(0)
    delta = settled[::2] - settled[1::2]
    positions = torch.stack(root_positions)
    tops = boxes[:, 2] + boxes[:, 9] * .5
    initial_on_box = heights[10:25, ::2].mean(0)
    # Report any later pelvis descent >15cm separately from initial settling.
    dropped = (heights[25:, ::2] < initial_on_box[None] - .15).any(0)
    result = {"mean_support_height_difference": delta.mean().item(),
              "expected_step_top": (boxes[:, 2] + boxes[:, 9] * .5).mean().item(),
              "simulation_seconds": args.steps*.02,
              "n_pelvis_dropped_over_15cm_after_settling": int(dropped.sum()),
              "per_task": [{"task": t["name"], "root_z_over_box": settled[2*i].item(),
                            "root_z_over_floor": settled[2*i+1].item(),
                            "difference": delta[i].item(),
                            "peak_force_over_box": forces[:, 2*i].max().item(),
                            "peak_force_over_floor": forces[:, 2*i+1].max().item(),
                            "final_root_xyz_over_box": positions[-1,2*i].tolist(),
                            "final_root_z_over_floor": heights[-1,2*i+1].item(),
                            "minimum_root_z_after_settling": heights[25:,2*i].min().item(),
                            "pelvis_dropped_over_15cm_after_settling": bool(dropped[i]),
                            "final_foot_force_over_box": forces[-1,2*i].item()}
                           for i, t in enumerate(spec["tasks"])],
              "root_positions": positions.cpu().tolist()}
    solver = env.sim.backend_manager._solver
    result["actual_colliders"] = [{"index": i, "name": solver.mj_model.geom(i).name,
                                    "type": int(solver.mj_model.geom_type[i]),
                                    "contype": int(solver.mj_model.geom_contype[i]),
                                    "conaffinity": int(solver.mj_model.geom_conaffinity[i])}
                                   for i in range(solver.mj_model.ngeom)]
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.record:
        import numpy as np
        geometry = {name: getattr(solver.mjw_model, name).numpy()
                    for name in ("geom_pos", "geom_size", "geom_quat")}
        np.savez_compressed(args.record + ".npz", qpos=np.stack(recorded_qpos),
                            forces=forces.cpu().numpy(), task_ids=env.task_ids.cpu().numpy(),
                            **geometry, task_manifest=str(Path(args.task_manifest).resolve()),
                            dt=.02, simulation="IsaacLab Newton/MuJoCo-Warp; actual solver state")
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_task","root_positions","actual_colliders")}), flush=True)
    print("saved", args.output, flush=True)
