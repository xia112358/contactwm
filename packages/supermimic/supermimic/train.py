"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from supermimic.paths import runtime_root
REPO = str(runtime_root())

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task_manifest", type=str, default=None, help="verified terrain-conditioned task experiment")
parser.add_argument("--contact_model", choices=("binary2", "force6", "original", "height_only", "force_only"), default="binary2")
parser.add_argument("--wm_contact_architecture",choices=("independent","shared"),default="independent")
parser.add_argument("--collision_profile",choices=("legacy_v1","surface_v2"),default="surface_v2")
parser.add_argument("--contact_response", choices=("legacy", "firm_v2", "manifold_v1", "manifold_v2", "manifold_v3", "newton_contacts_v1", "newton_contacts_v2", "newton_contacts_v3", "manifold_soft_v1", "manifold_balanced_v1"), default="manifold_v3")
parser.add_argument("--policy_force_observation", choices=("measured", "none"), default="measured")
parser.add_argument("--perception", choices=("heightmap", "panorama_asymmetric", "panorama_rolling", "panorama_oracle"), default="heightmap")
parser.add_argument("--depth_encoder", choices=("cnn", "mlp"), default="cnn")
parser.add_argument("--depth_layout", choices=("full", "lower_band"), default="full")
parser.add_argument("--reset_sampler", choices=("return_bins", "completion_recent30_failure_window"), default="return_bins")
parser.add_argument("--probes_per_motion", type=int, default=3)
parser.add_argument("--probe_mode", choices=("all_long", "long_short"), default="all_long")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--reference_manifest", type=str, help="Replacement motion references in the same physical task slots")
parser.add_argument("--check_numerics", action="store_true", help="stop and save the first invalid or extreme physical transition")
parser.add_argument("--capture_physics_failure", action="store_true", help="record first failing physics substep, phase and solver warm starts")
parser.add_argument("--no_reset_noise", action="store_true", help="initialize exactly from reference poses without perturbing the robot into nearby terrain")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--headless", action="store_true", help="Disable visualizers even if one was requested")
parser.add_argument("--startup_check", action="store_true", help="Initialize physics and run one policy step without optimization or checkpoint writes")
add_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    args.visualizer = None

task_metadata = {}
reference_provenance = None
reference_changed = False
if args.task_manifest:
    args.check_numerics = True
    from supermimic.lowstep import checkpoint_metadata, install
    args.task_manifest = os.path.abspath(args.task_manifest)
    task_spec = install(args.task_manifest, contact_model=args.contact_model,
                        reset_sampler=args.reset_sampler, probes_per_motion=args.probes_per_motion,
                        probe_mode=args.probe_mode, policy_force_observation=args.policy_force_observation,
                        perception=args.perception, perception_buffer_steps=args.buffer_steps, depth_layout=args.depth_layout, depth_encoder=args.depth_encoder,
                        collision_profile=args.collision_profile, contact_response=args.contact_response,wm_contact_architecture=args.wm_contact_architecture)
    args.motion_dir = task_spec["motion_dir"]
    task_metadata = checkpoint_metadata(task_spec)
    if args.reference_manifest:
        from supermimic.task_reference import replace_motion_references
        reference_provenance = replace_motion_references(task_spec, args.reference_manifest)
        args.motion_dir = task_spec["motion_dir"]

import torch
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = args.perception.startswith("panorama_") and args.depth_encoder == "cnn"
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
if args.perception == "panorama_oracle":
    # Compile depth encoders independently of Warp's optical queries. Module.compile
    # preserves checkpoint keys and supports gradients through the camera image.
    torch._dynamo.config.recompile_limit = 32
    wm.depth_encoder.compile(dynamic=False)
    pi.depth_encoder.compile(dynamic=False)
    # Keep kinematics and neural dynamics fused without tracing the Warp renderer.
    mdp.critic_base_features = torch.compile(mdp.critic_base_features, dynamic=False)
    wm.step_from_inputs = torch.compile(wm.step_from_inputs, dynamic=False)
    pi.forward_features = torch.compile(pi.forward_features, dynamic=False)
    critic.compile(dynamic=False)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    if ck["args"].get("wm_contact_architecture", "independent") != args.wm_contact_architecture:
        raise ValueError("Resume must preserve WM contact architecture; use a fresh comparison")
    if ck["args"].get("reset_sampler", "return_bins") != args.reset_sampler:
        raise ValueError("Resume must preserve the reset sampler; use a fresh experiment")
    if ck["args"].get("probe_mode", "all_long") != args.probe_mode:
        raise ValueError("Resume must preserve probe completion semantics")
    if task_metadata:
        from supermimic.lowstep import validate_checkpoint
        validate_checkpoint(ck, task_metadata)
        if 'floor_transfer' in ck:
            task_metadata['floor_transfer_json'] = json.dumps(ck['floor_transfer'], sort_keys=True)
        if 'contact_pipeline_transfer' in ck:
            task_metadata['contact_pipeline_transfer_json'] = json.dumps(ck['contact_pipeline_transfer'], sort_keys=True)
        elif 'contact_pipeline_transfer_json' in ck:
            task_metadata['contact_pipeline_transfer_json'] = ck['contact_pipeline_transfer_json']
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
    reference_changed = ck.get("reference_provenance_json") != (json.dumps(reference_provenance, sort_keys=True) if reference_provenance is not None else None)

if reference_provenance is not None:
    task_metadata["reference_provenance_json"] = json.dumps(reference_provenance, sort_keys=True)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g, visual_features=None):
    a = pi(o_hist, g) if visual_features is None else pi.forward_features(o_hist, g, visual_features)
    if args.perception in ("panorama_rolling", "panorama_oracle") or (args.task_manifest and args.contact_model in ("force6","force_only") and args.perception=="heightmap"):
        S1, xi1 = wm.joint_step(S, a, xi, xi_n)
    else:
        xi1 = wm.next_context(S, a, xi, xi_n) if args.task_manifest and args.contact_model != "original" else xi
        S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi1, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K), xi1, normalize_xi(xi1)


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


wm_rollout = torch.compile(_wm_rollout)
wm_err = torch.compile(_wm_err)
imag_step = torch.compile(_imag_step)
reward_term = torch.compile(_reward_term)
critic_input_c = torch.compile(_critic_input)
if args.perception.startswith("panorama_"):
    # Warp optical queries and image replay have explicit lifetimes; use eager
    # autograd around the renderer; pure neural and kinematic paths compile separately.
    wm_rollout, wm_err, imag_step = _wm_rollout, _wm_err, _imag_step
    def critic_input_c(S, f, xi_n, g):
        chunk = 8192 if args.depth_encoder == "mlp" else 512
        return torch.cat([_critic_input(tuple(x[i:i+chunk] for x in S),f[i:i+chunk],xi_n[i:i+chunk],g[i:i+chunk])
                          for i in range(0,len(f),chunk)])


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    state_loss=loss.detach()/args.wm_window
    force_loss=loss.new_zeros(())
    if args.task_manifest and args.contact_model != "original":
        force_loss=wm.contact_loss(w)
        loss = loss + 0.1 * force_loss
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=args.check_numerics)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), state_loss, force_loss.detach()


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    visual_features = pi.perception_features(o_hist) if args.perception in ("panorama_asymmetric", "panorama_rolling") else None
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    contexts = [xi_n]
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g, xi, xi_n = imag_step(S, o_hist, a_prev, f, xi, xi_n, g, visual_features)
        contexts.append(xi_n)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0, error_if_nonfinite=args.check_numerics)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = torch.stack(contexts).detach().flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        if PROFILE:
            torch.cuda.synchronize(); tq_features = time.time()
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=args.check_numerics)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms" + (f" (features {1e3*(tq_features-tq2):.0f} ms)" if not args.no_critic else ""), flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma, reset_noise=not args.no_reset_noise)
    if args.startup_check:
        with torch.no_grad():
            obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            action = pi(obs[:, None].expand(-1, Hh, -1), motion.goal(env.f, K))
            transition = env.step(action)
            if not all(bool(torch.isfinite(transition[key]).all()) for key in ("s", "s1", "a")):
                raise FloatingPointError("Nonfinite tracking startup transition")
        report = dict(stage="tracking", num_envs=env.N, physics_steps=1,
                      optimizer_updates=0, checkpoint_written=False,
                      collision_profile=args.collision_profile, contact_response=args.contact_response,wm_contact_architecture=args.wm_contact_architecture)
        Path(args.log_dir, "startup-check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        raise SystemExit(0)
    sampler = getattr(env, "reset_sampler_controller", None)
    if args.resume and sampler is not None and not reference_changed:
        sampler.load_state_dict(ck["reset_sampler_state"])
    failure_recorder = None
    if args.capture_physics_failure:
        from supermimic.physics_failure import PhysicsFailureRecorder
        failure_recorder = PhysicsFailureRecorder(env.sim.backend_manager)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
    obs_hist = torch.zeros(env.N, Hh, OBS_DIM, device=dev)
    training_ids = env.training_env_ids if sampler is not None else slice(None)

    @torch.no_grad()
    def collect(n_steps: int, stats: dict):
        global obs_hist, sim_steps
        for _ in range(n_steps):
            if args.perception.startswith("panorama_"):
                env.update_perception()
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
            g = motion.goal(env.f, K)
            pi.obs_norm.update(o[training_ids])
            pi.goal_norm.update(g[training_ids])
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
            if failure_recorder is not None:
                failure_recorder.begin_control_step()
            out = env.step(a)
            if args.check_numerics:
                # Check before replay insertion and normalizer updates. Finite but
                # explosive simulator values can poison both just like NaNs.
                checked = {key: out[key] for key in ("s", "s1", "a", "r", "xi", "contact_next") if key in out}
                bad = {key: (~torch.isfinite(value) | (value.abs() > 1e6)).any()
                       for key, value in checked.items()}
                if failure_recorder is not None:
                    bad["physics_stage"] = failure_recorder.has_failure()
                if bool(torch.stack(list(bad.values())).any()):
                    path = os.path.join(args.log_dir, "invalid_transition.pt")
                    diagnostic = failure_recorder.snapshot() if failure_recorder is not None else None
                    torch.save({"transition": out, "sim_steps": sim_steps,
                                "physics_failure": diagnostic,
                                "args": vars(args), "task_metadata": task_metadata,
                                "pi": pi.state_dict(), "wm": wm.state_dict(),
                                "critic": critic.state_dict()}, path)
                    details = {key: {"invalid": bool(bad[key]),
                                     "absmax": float(value.abs().max())}
                               for key, value in checked.items()}
                    if diagnostic and diagnostic.get("first_failure_detected"):
                        details["physics_stage"] = {key: diagnostic[key] for key in
                                                    ("world", "substep", "phase")}
                    raise FloatingPointError(f"Invalid simulator transition before replay/normalization: {details}; saved {path}")
            if sampler is not None:
                from supermimic.retry_sampler import training_transition
                out = training_transition(out, training_ids)
            buffer.add(out)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out and out["ep_len"].numel():
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
    # Replay and observation history are fresh on resume too. Fill enough
    # transitions before sampling multi-step windows or policy histories.
    collect(max(args.warmup_steps, args.wm_window + Hh), stats)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    write_log_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_f = open(log_path, "a")
    if write_log_header:
        log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n")
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "wm_state": [], "wm_force": []}
    for it in range(start_it, args.iterations):
        lr = lr_at(it)
        for o in (opt_wm, opt_pi, opt_v):
            for gparam in o.param_groups:
                gparam["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        prof = PROFILE and (it < 5 or it % 20 == 0)
        if prof:
            torch.cuda.synchronize(); tp0 = time.time()
        collect(n, stats)
        if prof:
            torch.cuda.synchronize(); tp1 = time.time()
        for _ in range(args.wm_batches):
            l, e8, state_loss, force_loss = wm_update()
        if prof:
            torch.cuda.synchronize(); tp2 = time.time()
            if it < args.wm_only_iters:
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms", flush=True)
        agg["wm"].append(l), agg["wm8"].append(e8)
        agg["wm_state"].append(state_loss);agg["wm_force"].append(force_loss)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv = policy_update()
            if prof:
                torch.cuda.synchronize(); tp3 = time.time()
                print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                   mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            with open(os.path.join(args.log_dir,"wm_diagnostics.jsonl"),"a") as diagnostic:
                diagnostic.write(json.dumps(dict(iteration=it+1,architecture=args.wm_contact_architecture,
                    state_loss=mean(agg["wm_state"]),force_loss=mean(agg["wm_force"]),
                    force_loss_weight=.1,wm_window=args.wm_window,iterations_per_second=ips))+"\n")
            if args.perception.startswith("panorama_"):
                perception_path = os.path.join(args.log_dir, "perception_stats.json")
                perception_stats = (dict(schema=task_metadata["observation_schema"],resolution=json.loads(task_metadata["perception_contract"])["resolution"],
                                         actor_force_input=False,actor_oracle_depth=True,wm_oracle_depth=True,
                                         depth_encoder=args.depth_encoder,depth_layout=args.depth_layout,
                                         image_bank_gib=0,history_map_gib=0)
                                    if args.perception == "panorama_oracle" else env.historical_sensor.stats)
                with open(perception_path + ".tmp", "w") as stream:
                    json.dump({"iteration": it+1, **perception_stats,
                               "cuda_allocated_gib": torch.cuda.memory_allocated()/2**30,
                               "cuda_reserved_gib": torch.cuda.memory_reserved()/2**30,
                               "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}, stream)
                os.replace(perception_path + ".tmp", perception_path)
            if sampler is not None:
                sampler_path = os.path.join(args.log_dir, "sampler_stats.json")
                with open(sampler_path + ".tmp", "w") as stream:
                    json.dump({"iteration": it+1, **sampler.summary(motion.names)}, stream)
                os.replace(sampler_path + ".tmp", sampler_path)
            print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                  f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
            ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                  "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                  "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            ck.update(task_metadata)
            if sampler is not None:
                ck["reset_sampler_state"] = sampler.state_dict()
                ck["training_data_audit"] = {"training_envs": args.num_envs, "physical_envs": env.N,
                                             "replay_envs": buffer.N, "probe_data_used": False}
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
            small.update(task_metadata)
            path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
            torch.save(small, path + ".tmp")
            os.replace(path + ".tmp", path)
