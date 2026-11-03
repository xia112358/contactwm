"""Check healthy CUDA-graph stepping and intentionally injected NaN capture."""
from supermimic.paths import runtime_path
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_manifest", default="data/lowclimb_boxes/task.json")
add_launcher_args(parser)
args = parser.parse_args()

import torch
import warp as wp
from supermimic.lowstep import install
spec = install(args.task_manifest, contact_model="force6")
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.physics_failure import PhysicsFailureRecorder
from supermimic.robot import default_joint_pos_list

device = "cuda:0"
kin = G1Kinematics(device)
motion = MotionLib.from_dir(spec["motion_dir"], kin, device)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
cfg = make_sim(device, args)
with launch_simulation(cfg, args), torch.no_grad():
    env = TrackingSim(cfg, len(spec["tasks"]), motion, mdp, eval_mode=True)
    recorder = PhysicsFailureRecorder(env.sim.backend_manager)
    action = torch.zeros(env.N, 29, device=device)
    for _ in range(3):
        recorder.begin_control_step()
        out = env.step(action)
        assert torch.isfinite(out["s1"]).all()
        assert not recorder.snapshot()["first_failure_detected"]
    print("PASS healthy physical steps and CUDA graph replay with recorder", flush=True)
    state = env.sim.backend_manager.backend.state_0
    # Deliberate diagnostic injection into environment 0's root coordinate.
    wp.to_torch(state.joint_q)[0] = float("nan")
    recorder.begin_control_step()
    env.step(action)
    captured = recorder.snapshot()
    assert captured["first_failure_detected"], captured
    assert captured["world"] == 0
    assert captured["substep"] == 1
    assert captured["phase"] == "position_collision_constraints"
    assert len(captured["pre_substeps_qpos_qvel_warmstart_ctrl"]) == 1
    target = runtime_path('validation/nan/capture_injection_test.pt')
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"intentional_injection": True, "capture": captured}, target)
    print("PASS injected NaN captured at first substep/position phase with warm-start and contact evidence", flush=True)
