"""Check canonical FK, network dimensions, and optional matched-terrain GPU simulation."""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
pass
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_manifest", required=True)
parser.add_argument("--contact_model", choices=("binary2", "force6"), default="binary2")
parser.add_argument("--sim", action="store_true")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--report", type=Path, default=None, help="save per-environment rollout diagnostics")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
from supermimic.lowstep import install
spec = install(args.task_manifest, contact_model=args.contact_model)
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP, OBS_DIM, INV_DIM, CRITIC_EXTRA, XI_DIM, normalize_xi
from supermimic.networks import WorldModel
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat
from supermimic.env import TrackingSim, make_sim

kin = G1Kinematics("cpu")
errors = []
for task in spec["tasks"]:
    with np.load(task["source_motion"], allow_pickle=False) as data:
        indices = [0, len(data["joint_pos"]) // 2, len(data["joint_pos"]) - 1]
        qpos = torch.tensor(data["joint_pos"][indices])
        names = data["body_names"].tolist()
        root_index = names.index("pelvis")
        state = (qpos[:, :3], quat_to_mat(qpos[:, 3:7]),
                 torch.tensor(data["body_lin_vel_w"][indices, root_index]),
                 torch.tensor(data["body_ang_vel_w"][indices, root_index]),
                 qpos[:, 7:], torch.tensor(data["joint_vel"][indices, 6:]))
        predicted, rotation = kin.forward(*state)[:2]
        body_indices = [names.index(name) for name in TRACKED_BODIES]
        reference = torch.tensor(data["body_pos_w"][indices][:, body_indices])
        reference_rotation = quat_to_mat(torch.tensor(data["body_quat_w"][indices][:, body_indices]))
        error = (predicted - reference).abs().max().item()
        rotation_error = (rotation - reference_rotation).abs().max().item()
        if error > 2e-5 or rotation_error > 2e-5:
            raise AssertionError((task["name"], error, rotation_error))
        errors.append((error, rotation_error))
print("PASS canonical FK vs published Newton arrays:", np.max(errors, axis=0).tolist(), flush=True)

motion = MotionLib.from_dir(spec["motion_dir"], kin, "cpu")
assert motion.num_frames == sum(task["frames"] for task in spec["tasks"])
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
frames = motion.clip_start.clone()
state = motion.ref_state(frames)
xi = torch.zeros(len(spec["tasks"]), XI_DIM)
xi[:, 0] = 1
xi[:, 33] = torch.arange(len(spec["tasks"]))
actions = torch.zeros(len(spec["tasks"]), 29)
obs = mdp.obs(state, actions, xi, frames)
critic = mdp.critic_features(state, frames, normalize_xi(xi))
assert obs.shape[-1] == OBS_DIM
assert critic.shape[-1] == INV_DIM + CRITIC_EXTRA
wm = WorldModel(hidden=(64, 64))
wm_inputs = wm.inputs(state, actions, normalize_xi(xi))
assert wm_inputs.shape[-1] == wm.in_dim
position = state[0].detach().clone().requires_grad_(True)
gradient_state = (position, *state[1:])
mdp.obs(gradient_state, actions, xi, frames).sum().backward()
assert position.grad is not None and torch.isfinite(position.grad).all()
assert position.grad.abs().sum() > 0
print(f"PASS network dimensions and terrain gradients: obs={OBS_DIM}, WM={wm.in_dim}, xi={XI_DIM}", flush=True)
from supermimic.mdp import flatten_state
window = {"s": flatten_state(state).unsqueeze(1), "a": actions.unsqueeze(1),
          "xi": xi.unsqueeze(1), "mask": torch.ones(len(xi), 1),
          "contact_next": torch.full((len(xi), 1, XI_DIM-34), .5 if args.contact_model == "force6" else 0.)}
contact_loss = wm.contact_loss(window)
contact_loss.backward()
assert torch.isfinite(contact_loss)
assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
           for parameter in wm.contact_head.parameters())
assert sum(parameter.grad.abs().sum() for parameter in wm.contact_head.parameters()) > 0
next_xi = wm.next_context(state, actions, xi, normalize_xi(xi))
assert torch.isfinite(next_xi).all()
if args.contact_model == "binary2":
    assert ((next_xi[:, 34:] > 0) & (next_xi[:, 34:] < 1)).all()
assert torch.equal(next_xi[:, :34], xi[:, :34])
from supermimic.lowstep import checkpoint_metadata, validate_checkpoint
metadata = checkpoint_metadata(spec)
validate_checkpoint(metadata, metadata)
try:
    validate_checkpoint({**metadata, "observation_schema": "old_geometry"}, metadata)
except ValueError:
    pass
else:
    raise AssertionError("Old observation schema accepted")
print("PASS trainable contact head, next contact context and checkpoint schema rejection", flush=True)

if args.sim:
    device = "cuda:0"
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(spec["motion_dir"], kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    config = make_sim(device, args)
    with launch_simulation(config, args), torch.no_grad():
        env = TrackingSim(config, args.num_envs, motion, mdp, eval_mode=True)
        ids = torch.arange(args.num_envs, device=device)
        env.reset(ids, motion.clip_start[env.task_ids] + 50)
        env.refresh()
        state = env.state()
        assert not env.robot.is_fixed_base, "Canonical robot must have a floating root"
        initial_root = state[0].clone()
        first_term_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
        first_term_root = initial_root.clone()
        max_speed_before_term = torch.zeros(args.num_envs, device=device)
        initial_frames = env.f.clone()
        contact_seen = torch.zeros(6 if args.contact_model == "force6" else 2, dtype=torch.bool, device=device)
        peak_forces = torch.zeros_like(contact_seen, dtype=torch.float32)
        body_ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
        actual = env.robot.data.body_link_pose_w.torch[:, body_ids, :3] - env.scene.env_origins[:, None]
        expected = kin.forward(*state)[0]
        error = (actual - expected).abs().max().item()
        assert error < 2e-5, error
        print(f"PASS canonical FK vs live simulator: max component error {error:.3g} m", flush=True)
        for step in range(args.steps):
            frame = torch.minimum(env.f + 1, motion.clip_end[env.task_ids])
            action = (motion.q[frame] - env.default_q) / env.alpha
            output = env.step(action)
            if args.report:
                next_state = env.state()
                active = first_term_step < 0
                max_speed_before_term = torch.maximum(
                    max_speed_before_term, next_state[2].norm(dim=-1) * active)
                newly_terminated = active & output["term"]
                first_term_step[newly_terminated] = step + 1
                first_term_root[newly_terminated] = next_state[0][newly_terminated]
            assert output["contact_next"].shape == (args.num_envs, XI_DIM-34)
            assert torch.isfinite(output["contact_next"]).all()
            if args.contact_model == "force6":
                forces = output["contact_next"].reshape(args.num_envs, 6, 3) * env.force_scale
                magnitude = forces.norm(dim=-1).amax(0)
                peak_forces = torch.maximum(peak_forces, magnitude)
                contact_seen |= magnitude > 1.
            else:
                assert ((output["contact_next"] == 0) | (output["contact_next"] == 1)).all()
                contact_seen |= output["contact_next"].bool().any(0)
            assert all(torch.isfinite(value).all() for value in env.state())
            env.f = torch.minimum(env.f, motion.clip_end[env.task_ids] - 1)
        displacement = (env.state()[0] - initial_root).abs().max().item()
        assert displacement > 1e-5, "Root pose did not move under physics"
        assert contact_seen[:2].all(), "Both feet must register physical contact during the rollout"
        if args.contact_model == "force6":
            assert contact_seen.all(), "All six body parts must produce measured contact in this coverage check"
            print("Measured force peaks N:", dict(zip(env.contact_parts, peak_forces.tolist())), flush=True)
        print("PASS physical left/right foot contacts observed", flush=True)
        print(f"PASS floating root moves under physics: max displacement component {displacement:.3g} m", flush=True)
        print(f"PASS {args.steps} finite physical steps on paired terrains; "
              f"open-loop terminations={int(env.failed_eval.sum())}/{args.num_envs} (not a trained-policy score)", flush=True)
        if args.report:
            final_root = env.state()[0]
            args.report.parent.mkdir(parents=True, exist_ok=True)
            solver = env.sim.backend_manager._solver
            floor_ids = [i for i in range(solver.mj_model.ngeom)
                         if "/Terrain_" in solver.mj_model.geom(i).name
                         and "/box_0_" in solver.mj_model.geom(i).name]
            assert len(floor_ids) == 1, floor_ids
            floor_id = floor_ids[0]
            floor_arrays = {name: getattr(solver.mjw_model, name).numpy()
                            for name in ("geom_pos", "geom_size", "geom_quat")}
            terrain_ids = [i for i in range(solver.mj_model.ngeom)
                           if "/Terrain_" in solver.mj_model.geom(i).name]
            sizes = torch.as_tensor(solver.mjw_model.geom_size.numpy()).expand(args.num_envs, -1, -1)[:, terrain_ids]
            bounds = torch.as_tensor(solver.mjw_model.geom_aabb.numpy()).expand(args.num_envs, -1, -1, -1)[:, terrain_ids, 1]
            bounds_error = (bounds - sizes).abs().amax(dim=(-1, -2))
            report = {"task_manifest": str(Path(args.task_manifest).resolve()),
                      "contact_model": args.contact_model, "steps": args.steps, "dt": 0.02,
                      "control": "open-loop reference joint targets; simulation continues after termination",
                      "floor_geom_name": solver.mj_model.geom(floor_id).name,
                      "terrain_bounds_max_halfsize_error_m": bounds_error.tolist(),
                      "per_environment": [
                          {"environment": i, "task": spec["tasks"][int(env.task_ids[i])]["name"],
                           "initial_frame": int(initial_frames[i]),
                           "initial_root": initial_root[i].tolist(),
                           "first_term_step": int(first_term_step[i]),
                           "first_term_root": first_term_root[i].tolist(),
                           "max_speed_before_term_m_s": float(max_speed_before_term[i]),
                           "floor_geom": {key: value[i % value.shape[0], floor_id].tolist()
                                          for key, value in floor_arrays.items()},
                           "final_root": final_root[i].tolist()}
                          for i in range(args.num_envs)]}
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print("Saved per-environment diagnostics:", args.report, flush=True)
