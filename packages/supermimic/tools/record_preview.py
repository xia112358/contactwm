"""Record a short checkpoint rollout for visualization, without computing evaluation scores."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
pass
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--clips", nargs="+", default=[
    "dance2_subject1", "jumps1_subject1", "fallAndGetUp1_subject1", "sprint1_subject4",
])
add_launcher_args(parser)
args = parser.parse_args()
if args.seconds <= 0:
    parser.error("--seconds must be positive")

import numpy as np
import torch
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import mat_to_quat

device = "cuda:0"
torch.backends.cuda.matmul.allow_tf32 = True
checkpoint = torch.load(args.ckpt, map_location=device)
trained = checkpoint["args"]
networks.AMP_DTYPE = torch.bfloat16 if trained.get("amp", "none") == "bf16" else None
kin = G1Kinematics(device)
files = [str(runtime_path('data/lafan1_g1') / f"{name}.csv") for name in args.clips]
motion = MotionLib(files, kin, device)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
history_length, goal_frames = trained["hist"], trained["goal_frames"]
policy = Policy(OBS_DIM, 73 * goal_frames, hist=history_length).to(device)
policy.load_state_dict(checkpoint["pi"])
policy.eval()
trajectory = {key: [] for key in ["root_p", "root_quat", "q", "f", "alive"]}
sim_config = make_sim(device, args)
with torch.no_grad(), launch_simulation(sim_config, args):
    env = TrackingSim(sim_config, len(files), motion, mdp, eval_mode=True)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(len(files), device=device), motion.clip_start.clone())
    history = None
    for step in range(round(args.seconds * 50)):
        state = env.state()
        observation = mdp.obs(state, env.a_prev, env.xi, env.f)
        history = (observation.unsqueeze(1).repeat(1, history_length, 1) if history is None
                   else torch.cat((history[:, 1:], observation.unsqueeze(1)), 1))
        action = policy(history, motion.goal(env.f, goal_frames))
        if not torch.isfinite(action).all():
            raise RuntimeError("Non-finite policy action")
        values = [state[0], mat_to_quat(state[1]), state[4], env.f, ~env.done_eval]
        for key, value in zip(trajectory, values):
            trajectory[key].append(value.cpu().clone())
        env.step(action)
        env.f = torch.minimum(env.f, motion.clip_end - 1)
    print("Preview terminated clips:", [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()])
output = Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output, names=np.array(motion.names),
    **{key: torch.stack(values).numpy() for key, values in trajectory.items()},
    ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(),
    ref_q=motion.q.cpu().numpy(), clip_start=motion.clip_start.cpu().numpy(),
    clip_len=motion.clip_len.cpu().numpy())
print(f"Saved {args.seconds}s per clip from checkpoint {checkpoint['iteration']} to {output}")
