"""Frozen-teacher native contact stress audit; never a training or dataset run.

Every original terrain stays paired with its motion. Reference-frame restarts
are cold-start stress tests, not evidence that a policy completes those motions.
"""
from __future__ import annotations
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pass


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1325)
    p.add_argument('--restart-steps', type=int, default=160)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--headless', action='store_true')
    add_launcher_args(p)
    a = p.parse_args()
    if a.headless:
        a.visualizer = None
    import numpy as np
    import torch
    import warp as wp
    from supermimic.lowstep import install, checkpoint_metadata, validate_checkpoint
    from supermimic.contact_response import contract
    task = runtime_path('data/php296_surface_v2/task.json')
    checkpoint = runtime_path('logs/php296_surface_v2_startprobe_20261029_v1/tracking/ckpt_010000.pt')
    spec = install(task, contact_model='force6', collision_profile='surface_v2', contact_response='manifold_v3')
    from supermimic.kinematics import G1Kinematics
    from supermimic.motion import MotionLib
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.networks import Policy
    from supermimic.robot import default_joint_pos_list
    from supermimic.env import TrackingSim, make_sim
    from supermimic.contact_native import NativeContacts
    from supermimic.contact_solver_audit import ContactSolverAudit
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda:0'
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    import supermimic.networks as networks
    networks.AMP_DTYPE = torch.bfloat16 if ck['args'].get('amp', 'none') == 'bf16' else None
    target_metadata = checkpoint_metadata(spec)
    source_metadata = dict(target_metadata)
    source_metadata.pop('contact_response_contract', None)
    # Deliberate diagnostic transfer: still enforce the original scene, asset,
    # motion and observation contract; never label the teacher as new physics.
    validate_checkpoint(ck, source_metadata)
    hist, goal_frames = ck['args'].get('hist', 10), ck['args'].get('goal_frames', 10)
    policy = Policy(OBS_DIM, 73*goal_frames, hist=hist).to(device).eval().requires_grad_(False)
    policy.load_state_dict(ck['pi'])
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(spec['motion_dir'], kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    a.output.mkdir(parents=True, exist_ok=False)
    selected = json.loads((runtime_path('data/php_low_speed_surface_v2/task.json')).read_text())
    low_ids = {t['source_task_index'] for t in selected['tasks']}
    report = dict(schema='native_contact_solver_sweep_v1', status='starting',
                  production_compatible=False, training=False, seed=a.seed,
                  source_checkpoint=str(checkpoint), source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source_response=contract('legacy'), target_response=contract('manifold_v3'),
                  tasks=len(spec['tasks']), low_speed_tasks=len(low_ids), phases=[], comparisons=[],
                  limits='Frozen old policy transferred to new physics; cold reference starts are stress tests, not motion success evaluations.')
    def save():
        temp = a.output/'progress.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(a.output/'report.json')
    save()
    cfg = make_sim(device, a)
    try:
        with launch_simulation(cfg, a), torch.no_grad():
            env = TrackingSim(cfg, len(spec['tasks']), motion, mdp, eval_mode=True)
            ids = torch.arange(env.N, device=device)
            assert len(env.task_ids.unique()) == len(spec['tasks'])
            reader = NativeContacts(env)
            reader.export_collision_model(a.output/'solver-collision.mjb')
            audit = ContactSolverAudit(env.sim.backend_manager)
            report['status'] = 'running'
            report['body_metadata'] = reader.body_contact_metadata
            task_ids = env.task_ids.cpu().tolist()
            phases = [('from_start', 0., a.steps, 0.)]
            if a.restart_steps:
                phases += [(f'cold_{fraction:.2f}', fraction, a.restart_steps, .03) for fraction in (.25, .5, .75)]
            started = time.monotonic()
            total_steps = 0
            for name, fraction, limit, noise in phases:
                starts = motion.clip_start[env.task_ids] + ((motion.clip_len[env.task_ids]-2)*fraction).long()
                env.reset(ids, starts)
                env.refresh()
                h = None
                first_finished = np.zeros(env.N, bool)
                phase = dict(name=name, start_fraction=fraction, action_noise_std=noise,
                             control_steps=0, resets=0, first_episodes=[], sampled_max_foot_penetration_mm=0.,
                             sampled_max_body_penetration_mm=0., sampled_max_body_force_n=0.)
                report['phases'].append(phase)
                # Read after reset must not pass stale contacts to a controller.
                reset_evidence = reader.read_device()
                assert not bool(reset_evidence['known'].any())
                assert not bool(reset_evidence['active'].any())
                for step in range(limit):
                    state = env.state()
                    obs = mdp.obs(state, env.a_prev, env.xi, env.f)
                    h = obs[:, None].expand(-1, hist, -1).clone() if h is None else torch.cat((h[:, 1:], obs[:, None]), 1)
                    action = policy(h, motion.goal(env.f, goal_frames))
                    if noise:
                        action = (action + noise*torch.randn_like(action)).clamp(-20, 20)
                    if not bool(torch.isfinite(action).all()):
                        raise ValueError('Nonfinite policy action before solver')
                    pre_state = {k: wp.to_torch(getattr(reader.solver.mjw_data, k)).clone()
                                 for k in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl')}
                    pre_frames = env.f.clone()
                    out = env.step(action)
                    audit.check()
                    for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart'):
                        if not bool(torch.isfinite(wp.to_torch(getattr(reader.solver.mjw_data, field))).all()):
                            raise ValueError(f'Nonfinite native {field} after integration')
                    total_steps += 1
                    phase['control_steps'] = step+1
                    if step % 10 == 0:
                        fast = reader.read_device()
                        if not bool(fast['known'].all()):
                            raise AssertionError('All worlds must be known after a solved step')
                    if step % 100 == 0 or step == limit-1:
                        full = reader.read()
                        fast = reader.read_device()
                        errors = {}
                        for key in ('known', 'active', 'surface_local', 'point', 'force', 'load'):
                            x, y = full[key], fast[key]
                            if key in ('known', 'active', 'surface_local'):
                                torch.testing.assert_close(x, y, rtol=0, atol=0)
                                errors[key] = 0.
                            else:
                                torch.testing.assert_close(x.double(), y.double(), rtol=1e-4, atol=5e-4, equal_nan=True)
                                errors[key] = float((x.double()-y.double()).abs().max())
                        report['comparisons'].append(dict(phase=name, step=step, errors=errors))
                        names = reader.body_contact_metadata['body_contact_names']
                        feet = [i for i, label in enumerate(names) if 'ankle_roll_link' in label]
                        phase['sampled_max_foot_penetration_mm'] = max(phase['sampled_max_foot_penetration_mm'], float(full['body_contact_penetration'][:, feet].max())*1000)
                        phase['sampled_max_body_penetration_mm'] = max(phase['sampled_max_body_penetration_mm'], float(full['body_contact_penetration'].max())*1000)
                        phase['sampled_max_body_force_n'] = max(phase['sampled_max_body_force_n'], float(torch.linalg.vector_norm(full['body_contact_force'], dim=-1).max()))
                        # Exercise scope changes after the device selector exists.
                        subset = torch.arange(0, env.N, 7, device=device)
                        reader.set_observed_worlds(subset)
                        scoped = reader.read_device()
                        torch.testing.assert_close(scoped['active'][subset], fast['active'][subset])
                        expected = torch.zeros(env.N, device=device, dtype=torch.bool); expected[subset] = True
                        torch.testing.assert_close(scoped['known'][:, 0], expected)
                        assert not bool(scoped['active'][~expected].any())
                        reader.set_observed_worlds(ids)
                    ended = (out['done'] | (env.f >= motion.clip_end[env.task_ids]-1)).nonzero().flatten()
                    for world in ended.cpu().tolist():
                        if not first_finished[world]:
                            task_id = task_ids[world]
                            phase['first_episodes'].append(dict(task=task_id, name=spec['tasks'][task_id]['name'],
                                group=spec['tasks'][task_id]['group'], low_speed=task_id in low_ids,
                                steps=step+1, tracking_terminated=bool(out['term'][world])))
                            first_finished[world] = True
                    if len(ended):
                        env.reset(ended, starts[ended])
                        env.refresh()
                        new_obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                        h[ended] = new_obs[ended, None].expand(-1, hist, -1)
                        phase['resets'] += len(ended)
                    env.f = torch.minimum(env.f, motion.clip_end[env.task_ids]-1)
                    if step % 25 == 0 or step == limit-1:
                        report.update(total_control_steps=total_steps, world_control_steps=total_steps*env.N,
                                      elapsed_seconds=time.monotonic()-started, audit=audit.report())
                        if report['audit']['native_solves'] < total_steps*contract('manifold_v3')['substeps']:
                            raise RuntimeError('Substep audit did not execute for every native solve')
                        save()
                        print(json.dumps(dict(phase=name, step=step+1, total=total_steps,
                                              first_episodes=int(first_finished.sum()), audit=report['audit'])), flush=True)
            report['status'] = 'passed'
            report['audit'] = audit.report()
            save()
    except Exception:
        report['status'] = 'failed'
        report['exception'] = traceback.format_exc()
        if 'audit' in locals():
            report['audit'] = audit.report()
        if 'reader' in locals():
            d = reader.solver.mjw_data
            torch.save(dict(state={k:wp.to_torch(getattr(d, k)).cpu() for k in ('qpos', 'qvel', 'qacc', 'ctrl', 'qacc_warmstart')},
                            pre_state={k:v.cpu() for k,v in pre_state.items()} if 'pre_state' in locals() else None,
                            pre_frames=pre_frames.cpu() if 'pre_frames' in locals() else None,
                            action=action.cpu() if 'action' in locals() else None,
                            task_ids=env.task_ids.cpu(), frames=env.f.cpu()), a.output/'failure-state.pt')
            count = min(int(wp.to_torch(d.nacon)[0]), len(d.contact.dist))
            raw = {k: getattr(d.contact, k).numpy()[:count] for k in
                   ('worldid', 'geom', 'dist', 'includemargin', 'type', 'frame', 'pos', 'efc_address', 'dim')}
            np.savez_compressed(a.output/'failure-contacts.npz', **raw)
        save()
        raise


if __name__ == '__main__':
    main()
