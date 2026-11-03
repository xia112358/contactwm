"""Replay every extracted slice with its original teacher and genuine preceding history."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--task_manifest', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contact_response', choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_v3','newton_contacts_v1','newton_contacts_v2','newton_contacts_v3','manifold_soft_v1','manifold_balanced_v1'), default='legacy')
    p.add_argument('--grace_steps', type=int, default=10)
    p.add_argument('--headless', action='store_true')
    add_launcher_args(p); a = p.parse_args()
    if a.headless:
        a.visualizer = None
    import torch
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    from supermimic.lowstep import install, checkpoint_metadata, validate_checkpoint
    spec = install(a.task_manifest, contact_model='force6', collision_profile='surface_v2',contact_response=a.contact_response)
    from supermimic.env import TrackingSim, make_sim
    from supermimic.motion import MotionLib
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM, unflatten_state, flatten_state
    from supermimic.robot import default_joint_pos_list
    from supermimic.rotations import mat_to_quat, rot_angle_sq
    from supermimic.contact_native import NativeContacts
    from supermimic.contact_control import goal_contact_mask
    import supermimic.networks as networks
    device = 'cuda:0'
    saved = torch.load(a.library/'rollout.pt', map_location=device, weights_only=False)
    slices = torch.load(a.library/'cases.pt', map_location=device, weights_only=False)
    data, meta = saved['data'], saved['metadata']
    validate_checkpoint(meta, checkpoint_metadata(spec))
    path = Path(meta['source_checkpoint'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['source_checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed')
    ck = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint(ck, checkpoint_metadata(spec))
    networks.AMP_DTYPE = torch.bfloat16 if ck['args'].get('amp') == 'bf16' else None
    hist, frames = ck['args']['hist'], ck['args']['goal_frames']
    policy = networks.Policy(OBS_DIM, 73*frames, hist=hist).to(device).eval().requires_grad_(False)
    policy.load_state_dict(ck['pi']); del ck
    kin = G1Kinematics(device); motion = MotionLib.from_dir(spec['motion_dir'], kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    counts = [max(1, int((slices['motion_ids'] == i).sum())) for i in range(len(spec['tasks']))]
    a.num_envs = sum(counts); a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(dict(library=str(a.library.resolve()),
        task_manifest=a.task_manifest, checkpoint=str(path), num_envs=a.num_envs,
        contact_tolerance_m=.05, root_tolerance_m=.25, angle_tolerance_rad=.4, grace_steps=a.grace_steps,
        execution='one independent world per slice; original state, reference clock and preceding history; no optimization'), indent=2))
    cfg = make_sim(device, a); started = time.time()
    with launch_simulation(cfg, a), torch.no_grad():
        env = TrackingSim(cfg, a.num_envs, motion, mdp, eval_mode=True, task_world_counts=counts)
        all_ids = torch.arange(env.N, device=device)
        env.reset(all_ids, motion.clip_start[env.task_ids]); env.refresh()
        ids = torch.empty_like(slices['motion_ids'])
        for task in slices['motion_ids'].unique().tolist():
            case_ids = (slices['motion_ids'] == task).nonzero().flatten()
            worlds = (env.task_ids == task).nonzero().flatten()
            if len(worlds) != len(case_ids):
                raise ValueError('Incorrect per-terrain case allocation')
            ids[case_ids] = worlds
        starts, stops, goals = (slices[k] for k in ('starts', 'stops', 'goals'))
        env.reset(ids, data['f'][starts].long())
        pos, rot, vel, omega, q, qd = unflatten_state(data['s'][starts]); quat = mat_to_quat(rot)
        env.robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((pos+env.scene.env_origins[ids], quat[:,1:], quat[:,:1]), -1), env_ids=ids)
        env.robot.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((vel, omega), -1), env_ids=ids)
        env.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        env.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
        env.robot.set_joint_position_target_index(target=q, env_ids=ids)
        env.a_prev[ids] = data['a_prev'][starts]; env.xi[ids] = data['xi'][starts]
        env.ep_step[ids] = data['ep_step'][starts].to(env.ep_step.dtype)
        env.refresh()
        initial = flatten_state(env.state())[ids]
        # Quaternion signs can differ while representing the same orientation.
        state_error = (initial[:,7:]-data['s'][starts,7:]).abs().amax(-1)
        root_reset_error = (initial[:,:3]-data['s'][starts,:3]).norm(dim=-1)
        if float(state_error.max()) > 1e-3 or float(root_reset_error.max()) > 1e-4:
            raise ValueError('Recorded physical state was not restored')
        obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        history = obs[:,None].expand(-1,hist,-1).clone()
        hi = slices['history_indices']
        history[ids] = mdp.obs(unflatten_state(data['s'][hi].flatten(0,1)),
            data['a_prev'][hi].flatten(0,1), data['xi'][hi].flatten(0,1),
            data['f'][hi].flatten()).reshape(len(ids),hist,-1)
        # Capture uses primary-face contact centroids. Validate that same
        # measurement contract, not a nearest individual constraint point.
        reader = NativeContacts(env, direct=False)
        caps = stops-starts+a.grace_steps; new = slices['target_masks'] & ~slices['keep_masks']
        reached = torch.zeros_like(new); passed = torch.zeros(len(ids), dtype=torch.bool, device=device)
        failed = passed.clone(); endpoint_root = torch.zeros(len(ids), device=device)
        endpoint_angle = endpoint_root.clone(); restart_action_error = endpoint_root.clone()
        reference = unflatten_state(data['s1'][goals])
        restart_observation_error = 0.
        for step in range(int(caps.max())):
            live = (step < caps) & ~passed & ~failed
            current_state = env.state()
            if step == 0:
                # state() refreshes force sensors, which have no solve yet
                # after reset. The first policy input needs recorded evidence.
                env.xi[ids] = data['xi'][starts]
            obs = mdp.obs(current_state, env.a_prev, env.xi, env.f)
            if step == 0:
                recorded_obs = mdp.obs(unflatten_state(data['s'][starts]),data['a_prev'][starts],
                                       data['xi'][starts],data['f'][starts])
                restart_observation_error = float((obs[ids]-recorded_obs).abs().max())
                if restart_observation_error > 1e-3:
                    raise ValueError('Restart observation differs from recorded policy input')
            history = torch.cat((history[:,1:],obs[:,None]),1)
            action = policy(history, motion.goal(env.f,frames))
            if step == 0:
                restart_action_error = (action[ids]-data['a'][starts]).abs().amax(-1)
            out = env.step(action)
            evidence = {k:v[ids] for k,v in reader.read_device().items()}
            observed = evidence['active'] & evidence['known']
            reached |= goal_contact_mask(evidence, data['point'][goals], data['surface_local'][goals], tolerance=.05) & new & live[:,None]
            state = unflatten_state(out['s1'][ids])
            root = (state[0]-reference[0]).norm(dim=-1)
            angle = rot_angle_sq(state[1],reference[1]).sqrt()
            failed |= live & (out['term'][ids] | ~torch.isfinite(out['s1'][ids]).all(-1))
            arrived = ((~new)|reached).all(-1)
            kept = ((~slices['keep_masks'])|observed).all(-1)
            completed = live & arrived & (root < .25) & (angle < .4) & (new.any(-1)|kept) & ~failed
            passed |= completed
            ending = live & ((step+1 == caps) | completed | failed)
            endpoint_root[ending] = root[ending]; endpoint_angle[ending] = angle[ending]
            env.f = torch.minimum(env.f, motion.clip_end[env.task_ids]-1)
            if step % 50 == 0:
                print(json.dumps(dict(step=step, max_steps=int(caps.max()), completed=int((caps<=step+1).sum()),
                    reached=int(passed.sum()), elapsed_s=time.time()-started)), flush=True)
        valid = passed & ~failed
        rows = [dict(case=i+1, motion=int(slices['motion_ids'][i]), frames=int(stops[i]-starts[i]), budget_frames=int(caps[i]),
                     native_restart_passed=bool(valid[i]), tracking_terminated=bool(failed[i]),
                     goal_reached=bool(passed[i]), endpoint_root_error_m=float(endpoint_root[i]),
                     endpoint_angle_error_rad=float(endpoint_angle[i]),
                     restart_action_max_error=float(restart_action_error[i])) for i in range(len(ids))]
        report = dict(total=len(ids), passed=int(valid.sum()), failed=int((~valid).sum()),
            max_reset_state_error=float(state_error.max()), max_reset_root_error=float(root_reset_error.max()),
            max_restart_observation_error=restart_observation_error,
            contact_measurement='same primary-face centroid and local surface identity as capture',
            elapsed_s=time.time()-started, cases=rows,
            caveat='State/history restart validation; solver warm-start internal state is reset, not bitwise replay.')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2), flush=True)


if __name__ == '__main__':
    main()
