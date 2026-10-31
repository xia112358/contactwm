"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from supermimic.paths import runtime_root
REPO = str(runtime_root())
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--result_base", type=str, default=None, help="separate output prefix for a checkpoint evaluation")
parser.add_argument("--eval_seed", type=int, default=None, help="seed RNGs without enabling evaluation noise")
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--iteration_interval", type=int, default=None,
                    help="evaluate checkpoints at multiples of this training-update interval")
parser.add_argument("--stop_after", type=int, default=None,
                    help="stop watching after evaluating this training iteration")
parser.add_argument("--task_manifest", type=str, default=None)
parser.add_argument("--contact_model", choices=("binary2", "force6", "original", "height_only", "force_only"), default=None)
parser.add_argument("--terrain_transfer", action="store_true", help="explicitly test policy weights on corrected terrain; preserve robot, motions and observation schema")
parser.add_argument("--collision_profile",choices=("legacy_v1","surface_v2"),default=None)
parser.add_argument("--contact_response", choices=("legacy", "firm_v2", "manifold_v1", "manifold_v2", "manifold_v3", "newton_contacts_v1", "newton_contacts_v2", "manifold_soft_v1", "manifold_balanced_v1"), default=None)
add_launcher_args(parser)
args = parser.parse_args()
if args.iteration_interval is not None and args.iteration_interval < 1:
    parser.error("--iteration_interval must be positive")
if args.stop_after is not None and args.stop_after < 1:
    parser.error("--stop_after must be positive")
if args.terrain_transfer and (not args.ckpt or not args.record or not args.task_manifest):
    parser.error("--terrain_transfer requires --ckpt, --record and an explicit --task_manifest")

from supermimic.evaluation_queue import pending_checkpoints, wait_for_checkpoint

if args.watch and not args.ckpt:
    print("[eval] waiting for an eligible checkpoint before initializing CUDA/simulation", flush=True)
    if not wait_for_checkpoint(args.run_dir, iteration_interval=args.iteration_interval,
                               every=args.every, stop_after=args.stop_after):
        print("[eval] stopped/completed run; no simulator initialized", flush=True)
        sys.exit(0)

task_metadata = {}
run_args_path = os.path.join(args.run_dir, "args.json")
from somaforge_core.published_data import relocate_published_values
run_args = relocate_published_values(json.load(open(run_args_path))) if os.path.isfile(run_args_path) else {}
args.contact_model = args.contact_model or run_args.get("contact_model", "binary2")
if not args.task_manifest and os.path.isfile(run_args_path):
    args.task_manifest = run_args.get("task_manifest")
if args.task_manifest:
    from supermimic.lowstep import checkpoint_metadata, install
    task_spec = install(args.task_manifest, contact_model=args.contact_model,
                        policy_force_observation=run_args.get("policy_force_observation", "measured"),
                        perception=run_args.get("perception", "heightmap"), perception_buffer_steps=10,
                        depth_layout=run_args.get("depth_layout", "full"),
                        depth_encoder=run_args.get("depth_encoder", "cnn"),
                        collision_profile=args.collision_profile or run_args.get('collision_profile','legacy_v1'),
                        contact_response=args.contact_response or run_args.get('contact_response','legacy'))
    args.motion_dir = task_spec["motion_dir"]
    task_metadata = checkpoint_metadata(task_spec)

import numpy as np
import torch

if args.eval_seed is not None:
    import random
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, OBS_DIM, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    if task_metadata:
        from supermimic.lowstep import validate_checkpoint
        if args.terrain_transfer:
            identity = {k: v for k, v in task_metadata.items() if k != "task_manifest_sha256"}
            validate_checkpoint(ck, identity)
            source = json.load(open(ck["args"]["task_manifest"]))
            keys = ("name", "csv_sha256", "source_motion_sha256")
            if [[task[k] for k in keys] for task in source["tasks"]] != [[task[k] for k in keys] for task in task_spec["tasks"]]:
                raise ValueError("Terrain transfer cannot change reference motions")
        else:
            validate_checkpoint(ck, task_metadata)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        if run_args.get("perception", "heightmap").startswith("panorama_"):
            env.update_perception()
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    reference_steps = motion.clip_len - 1
    length_ratio = steps / reference_steps
    res = {
        "terrain_transfer": args.terrain_transfer,
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
        "mean_rollout_seconds": (steps.mean() * 0.02).item(),
        "mean_reference_seconds": (reference_steps.float().mean() * 0.02).item(),
        "mean_length_ratio": length_ratio.mean().item(),
        "length_ratio_framew": (steps.sum() / reference_steps.sum()).item(),
        "per_clip_lengths": {motion.names[i]: {
            "steps": int(steps[i].item()), "seconds": steps[i].item() * 0.02,
            "reference_steps": int(reference_steps[i].item()),
            "reference_seconds": reference_steps[i].item() * 0.02,
            "length_ratio": length_ratio[i].item(),
            "failed": bool(env.failed_eval[i].item()), "norm_return": norm[i].item(),
        } for i in range(C)},
    }
    if args.task_manifest and any("group" in task for task in task_spec["tasks"]):
        groups = sorted({task.get("group", "ungrouped") for task in task_spec["tasks"]})
        res["groups"] = {}
        for group in groups:
            ids = torch.tensor([i for i, task in enumerate(task_spec["tasks"])
                                if task.get("group", "ungrouped") == group], device=dev)
            res["groups"][group] = {
                "n_clips": len(ids), "n_failed": int(env.failed_eval[ids].sum().item()),
                "norm_return": norm[ids].mean().item(),
                "mean_rollout_seconds": (steps[ids].mean() * 0.02).item(),
                "mean_reference_seconds": (reference_steps[ids].float().mean() * 0.02).item(),
                "mean_length_ratio": length_ratio[ids].mean().item(),
            }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            terrain_transfer=np.asarray(args.terrain_transfer),
                            source_checkpoint=np.asarray(args.ckpt or ""),
                            **{k: np.asarray(v) for k, v in task_metadata.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        res = evaluate(env, ck, args.record)
        result_base = os.path.splitext(args.record)[0] if args.record else os.path.splitext(args.ckpt)[0]
        result_base = args.result_base or result_base
        res["eval_seed"] = args.eval_seed
        json.dump(res, open(result_base + "_eval.json", "w"), indent=2)
        import csv
        with open(result_base + "_lengths.csv", "w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["clip", *next(iter(res["per_clip_lengths"].values()))])
            writer.writeheader()
            for name, values in res["per_clip_lengths"].items():
                writer.writerow({"clip": name, **values})
        print(json.dumps({k: v for k, v in res.items() if k not in ("per_clip", "per_clip_lengths")}, indent=1))
        sys.exit(0)
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        if args.watch and os.path.isfile(os.path.join(args.run_dir, "invalid_transition.pt")):
            print("[eval] training saved an invalid transition; watcher exiting", flush=True)
            break
        todo = pending_checkpoints(args.run_dir, done, iteration_interval=args.iteration_interval,
                                   every=args.every, stop_after=args.stop_after)
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(env, ck)
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        if args.stop_after is not None and any(it >= args.stop_after for it in done):
            break
        if os.path.isfile(os.path.join(args.run_dir, "STOPPED.json")):
            print("[eval] training run marked stopped; watcher exiting", flush=True)
            break
        time.sleep(60)
