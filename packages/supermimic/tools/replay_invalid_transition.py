"""Replay a saved invalid physical transition without loading or training networks.

Restores measured link state, friction, COM perturbation, offsets and action.
The original snapshot does not contain solver warm-start/contact-cache state.
"""
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
parser.add_argument("--snapshot", type=Path, default=runtime_path('logs/climb_low_force6_main/invalid_transition.pt'))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bad_env", type=int, default=652)
parser.add_argument("--num_envs", type=int, default=40)
parser.add_argument("--restore_all", action="store_true")
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--substeps", type=int, default=4)
parser.add_argument("--solver", choices=("newton", "cg"), default="newton")
parser.add_argument("--zero_com", action="store_true")
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--policy_steps", type=int, default=0, help="after replay, stress physics with the frozen saved policy")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch
import warp as wp
from newton import ModelFlags
from supermimic.lowstep import install

saved = torch.load(args.snapshot, map_location="cpu", weights_only=False)
transition = saved["transition"]
spec = install(saved["args"]["task_manifest"], contact_model="force6")
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP, unflatten_state
from supermimic.robot import default_joint_pos_list
from supermimic.env import TrackingSim, make_sim

device = "cuda:0"
torch.manual_seed(0)
kin = G1Kinematics(device)
motion = MotionLib.from_dir(spec["motion_dir"], kin, device)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
cfg = make_sim(device, args)
cfg.physics.use_cuda_graph = False  # Allow reads between physical substeps.
cfg.physics.num_substeps = args.substeps
cfg.physics.solver_cfg.solver = args.solver
args.output.parent.mkdir(parents=True, exist_ok=True)
report = {"snapshot": str(args.snapshot.resolve()), "configuration": vars(args).copy(),
          "warmstart_restored": False, "events": []}
report["configuration"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["configuration"].items()}


def stats(array):
    values = np.asarray(array)
    finite = np.isfinite(values)
    return {"shape": list(values.shape), "nonfinite": int((~finite).sum()),
            "absmax_finite": float(np.abs(values[finite]).max()) if finite.any() else None}


def clean(value):
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


with launch_simulation(cfg, args), torch.no_grad():
    env = TrackingSim(cfg, args.num_envs, motion, mdp, eval_mode=True)
    manager = env.sim.backend_manager
    solver = manager._solver
    data = solver.mjw_data
    task_id = int(transition["xi"][args.bad_env, 33])
    if args.restore_all:
        assert args.num_envs == len(transition["s"])
        ids = torch.arange(args.num_envs, device=device)
        source_ids = ids.cpu()
        inspected = args.bad_env
        assert torch.equal(env.task_ids.cpu(), transition["xi"][:, 33].long())
    else:
        inspected = int((env.task_ids == task_id).nonzero()[0])
        ids = torch.tensor([inspected], device=device)
        source_ids = torch.tensor([args.bad_env])
    print(f"Inspecting environment {inspected}, task {task_id}; restore_all={args.restore_all}", flush=True)
    default_com = env.robot.data.body_com_pose_b.torch[ids, env.torso_id, :3].clone()
    report["task"] = spec["tasks"][task_id]["name"]
    report["inspected_world"] = inspected
    original_step = manager._step_solver
    counter = {"repeat": 0, "substep": 0}

    def capture(label):
        event = {**counter, "label": label}
        for name in ("qpos", "qvel", "qacc", "qacc_smooth", "qacc_warmstart", "qfrc_actuator",
                     "qfrc_smooth", "qfrc_constraint", "qLD", "qLDiagInv", "nefc", "solver_niter"):
            value = getattr(data, name).numpy()[inspected]
            event[name] = stats(value)
            if name in ("qpos", "qvel", "qacc"):
                event[name]["values"] = value.tolist()
        active_rows = int(data.nefc.numpy()[inspected])
        for name in ("D", "aref", "force", "pos"):
            value = getattr(data.efc, name).numpy()[inspected]
            event["efc_" + name] = stats(value[:active_rows])
        count = int(data.nacon.numpy()[0])
        worlds = data.contact.worldid.numpy()[:count]
        indices = np.flatnonzero(worlds == inspected)
        fields = {name: getattr(data.contact, name).numpy() for name in
                  ("dist", "includemargin", "frame", "geom", "type", "efc_address")}
        event["contacts"] = []
        for index in indices:
            contact = {name: value[index].tolist() for name, value in fields.items()}
            contact["geom_names"] = [solver.mj_model.geom(int(i)).name for i in fields["geom"][index]]
            event["contacts"].append(contact)
        report["events"].append(clean(event))
        print(label, "repeat", counter["repeat"], "substep", counter["substep"],
              "qacc", event["qacc"]["nonfinite"], event["qacc"]["absmax_finite"],
              "contacts", len(indices), "nefc", int(data.nefc.numpy()[inspected]), flush=True)

    def traced_step(*values):
        counter["substep"] += 1
        original_step(*values)
        capture("after_solver_substep")

    manager._step_solver = traced_step
    for repeat in range(args.repeats):
        counter.update(repeat=repeat, substep=0)
        xi = transition["xi"][source_ids].to(device).clone()
        env.xi[ids] = xi
        mu = wp.to_torch(manager.get_model().shape_material_mu)
        shape_body = wp.to_torch(manager.get_model().shape_body)
        mu[shape_body < 0] = 0.3  # Training uses 0.3 for terrain, max with robot friction.
        env._apply_friction()
        com_offset = torch.zeros_like(xi[:, 30:33]) if args.zero_com else xi[:, 30:33]
        env.robot.set_coms_index(coms=(default_com + com_offset).unsqueeze(1),
                                body_ids=[env.torso_id], env_ids=ids)
        raw = transition["s"][source_ids].to(device)
        state = unflatten_state(raw)
        pose = torch.cat((state[0] + env.scene.env_origins[ids], raw[:, 4:7], raw[:, 3:4]), -1)
        env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat(state[2:4], -1), env_ids=ids)
        env.robot.write_joint_position_to_sim_index(position=state[4], env_ids=ids)
        env.robot.write_joint_velocity_to_sim_index(velocity=state[5], env_ids=ids)
        action = transition["a"][source_ids].to(device) * args.action_scale
        target = env.default_q + xi[:, 1:30] + env.alpha * action
        env.robot.set_joint_position_target_index(target=target, env_ids=ids)
        env.scene.reset(ids)
        env.refresh()
        env.f[ids] = transition["f"][source_ids].to(device)
        env.ep_step[ids] = transition["ep_step"][source_ids].to(device)
        restored = env.state()
        errors = [float((a[ids] - b).abs().max()) for a, b in zip(restored, state)]
        report.setdefault("restore_errors", []).append(errors)
        print("Restoration errors:", errors, flush=True)
        assert max(errors) < 1e-4
        capture("restored")
        for _ in range(args.steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(0.02)
            if not all(torch.isfinite(x[inspected]).all() for x in env.state()):
                print("REPRODUCED nonfinite simulator state", flush=True)
                break
        args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    report["reproduced"] = any(e["qpos"]["nonfinite"] or e["qvel"]["nonfinite"]
                               for e in report["events"] if e["label"] == "after_solver_substep")
    if args.policy_steps:
        import importlib
        from supermimic import networks
        from supermimic.mdp import OBS_DIM, flatten_state
        from supermimic.networks import Policy

        networks.AMP_DTYPE = torch.bfloat16 if saved["args"]["amp"] == "bf16" else None
        torch.backends.cuda.matmul.allow_tf32 = True
        hist_len, goals = saved["args"]["hist"], saved["args"]["goal_frames"]
        policy = Policy(OBS_DIM, 73 * goals, hist=hist_len).to(device)
        policy.load_state_dict(saved["pi"])
        policy.eval()
        history = torch.zeros(args.num_envs, hist_len, OBS_DIM, device=device)
        env.eval_mode = False
        env.reset_noise = True
        previous_substep = {}
        active_transition = {}

        class LocatedNonfinite(RuntimeError):
            pass

        def guard(original, label):
            def wrapped(*values, **kwargs):
                global inspected
                result = original(*values, **kwargs)
                bad = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
                for name in ("qpos", "qvel", "qacc", "qacc_smooth"):
                    v = wp.to_torch(getattr(data, name))
                    bad |= ~torch.isfinite(v).all(-1)
                if bool(bad.any()):
                    inspected = int(bad.nonzero()[0])
                    report["stress_failure"] = {"policy_step": counter["policy_step"],
                                                "phase": label, "world": inspected,
                                                "task": int(env.task_ids[inspected])}
                    capture("FIRST_NONFINITE_" + label)
                    fault = {"failure": report["stress_failure"],
                             "transition_before": {k: v[inspected].cpu() for k, v in active_transition.items()},
                             "substep_before": {k: v[inspected].cpu() for k, v in previous_substep.items()},
                             "xi": env.xi[inspected].cpu(),
                             "joint_target": env.robot.actuators.target_command.position.torch[inspected].cpu()
                             if hasattr(env.robot.actuators.target_command, "position") else None}
                    torch.save(fault, args.output.with_suffix(".fault.pt"))
                    raise LocatedNonfinite(label)
                return result
            return wrapped

        forward = importlib.import_module("mujoco_warp._src.forward")
        for name in ("fwd_position", "fwd_velocity", "fwd_actuation", "fwd_acceleration", "implicit"):
            setattr(forward, name, guard(getattr(forward, name), name))
        forward.solver.solve = guard(forward.solver.solve, "constraint_solve")

        def monitored_substep(*values):
            counter["substep"] += 1
            previous_substep.clear()
            for name in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
                previous_substep[name] = wp.to_torch(getattr(data, name)).clone()
            previous_substep["newton_joint_q"] = wp.to_torch(values[0].joint_q).reshape(args.num_envs, -1).clone()
            previous_substep["newton_joint_qd"] = wp.to_torch(values[0].joint_qd).reshape(args.num_envs, -1).clone()
            return original_step(*values)

        manager._step_solver = monitored_substep
        env.a_prev[ids] = transition["a"][source_ids].to(device)
        env.f[ids] += args.steps
        env.ep_step[ids] += args.steps
        try:
            for step in range(args.policy_steps):
                counter.update(policy_step=step, substep=0)
                state = env.state()
                obs = mdp.add_noise(mdp.obs(state, env.a_prev, env.xi, env.f))
                history = torch.where((env.ep_step == 0)[:, None, None], obs[:, None].expand_as(history),
                                      torch.cat((history[:, 1:], obs[:, None]), 1))
                if step == 0:
                    history[:] = obs[:, None]
                action = policy(history, motion.goal(env.f, goals))
                action = (action + saved["args"]["action_noise"] * torch.randn_like(action)).clamp(-20, 20)
                active_transition = {"s": flatten_state(state), "a": action.clone(), "f": env.f.clone()}
                env.step(action)
                if step % 50 == 0:
                    print("Frozen-policy stress step", step, "physics finite", flush=True)
                report["stress_steps_completed"] = step + 1
        except LocatedNonfinite as error:
            print("LOCATED nonfinite phase:", error, report["stress_failure"], flush=True)
    args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print("Report:", args.output, "reproduced:", report["reproduced"], flush=True)
