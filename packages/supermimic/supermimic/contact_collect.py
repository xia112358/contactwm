"""Record real tracking transitions and solver contacts, without a PHP policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys




def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task_manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num_envs', type=int, default=40)
    p.add_argument('--motion', type=int, default=5)
    p.add_argument('--all_motions',action='store_true',help='Record each paired motion separately in one native run')
    p.add_argument('--selection_manifest',type=Path,help='Select original task indices without changing the checkpoint scene contract')
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--capture_substeps', action='store_true', help='Save every native solver contact solve for stable-region supervision')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--noise', type=float, default=0.)
    p.add_argument('--collision_profile',choices=('legacy_v1','surface_v2'),default='surface_v2')
    p.add_argument('--contact_response', choices=('legacy','firm_v2','manifold_v1','manifold_v2','manifold_v3','newton_contacts_v1','newton_contacts_v2','newton_contacts_v3','manifold_soft_v1','manifold_balanced_v1'), default='legacy')
    p.add_argument('--allow_collision_transfer',action='store_true',help='Use legacy policy only as an unvalidated initialization in surface-v2 physics')
    p.add_argument('--headless', action='store_true')
    add_launcher_args(p)
    a = p.parse_args()
    if a.capture_substeps and not a.all_motions:
        p.error('--capture_substeps currently requires --all_motions; no partial capture is written')
    if a.headless:
        a.visualizer = None
    import torch
    import numpy as np
    from supermimic.lowstep import install, checkpoint_metadata, validate_checkpoint
    spec = install(a.task_manifest, contact_model='force6',collision_profile=a.collision_profile,contact_response=a.contact_response)
    from supermimic.kinematics import G1Kinematics
    from supermimic.motion import MotionLib
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.networks import Policy
    from supermimic.robot import default_joint_pos_list
    from supermimic.env import TrackingSim, make_sim
    from supermimic.contact_native import NativeContacts
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda:0'
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    import supermimic.networks as networks
    networks.AMP_DTYPE = torch.bfloat16 if ck['args'].get('amp','none')=='bf16' else None
    metadata = checkpoint_metadata(spec)
    transfer=validate_checkpoint(ck,metadata,allow_collision_transfer=a.allow_collision_transfer)
    if a.selection_manifest:
        if not a.all_motions:
            raise ValueError('A selection manifest requires --all_motions')
        import hashlib
        selected=json.loads(a.selection_manifest.read_text())
        if selected['selection_source_manifest_sha256']!=hashlib.sha256(Path(a.task_manifest).read_bytes()).hexdigest():
            raise ValueError('Selection is not derived from the checkpoint task manifest')
        a.selected_motion_ids=[]
        for task in selected['tasks']:
            index=task['source_task_index'];original=spec['tasks'][index]
            for key in ('name','source_motion_id','csv_sha256','terrain_usd_sha256'):
                if task[key]!=original[key]:raise ValueError(f'Selection source mismatch: {key}')
            a.selected_motion_ids.append(index)
        if not a.selected_motion_ids or len(set(a.selected_motion_ids))!=len(a.selected_motion_ids):
            raise ValueError('Selection must contain distinct original tasks')
    hist = ck['args'].get('hist',10)
    frames = ck['args'].get('goal_frames',10)
    policy = Policy(OBS_DIM,73*frames,hist=hist).to(device).eval().requires_grad_(False)
    policy.load_state_dict(ck['pi'])
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(spec['motion_dir'],kin,device)
    mdp = MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=device))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    cfg = make_sim(device,a)
    recorded = []
    with launch_simulation(cfg,a), torch.no_grad():
        env = TrackingSim(cfg,a.num_envs,motion,mdp,eval_mode=True)
        ids = torch.arange(env.N,device=device)
        env.reset(ids,motion.clip_start[env.task_ids])
        env.refresh()
        reader = NativeContacts(env)
        metadata.update(reader.body_contact_metadata)
        reader.export_collision_model(a.output/'solver-collision.mjb')
        if a.all_motions:
            from supermimic.contact_library import collect_motion_library
            collect_motion_library(a,env,motion,mdp,policy,reader,hist,frames,metadata,ck,transfer)
            return
        target = (env.task_ids==a.motion).nonzero().flatten()
        if len(target)!=1:
            raise ValueError('First diagnostic records one environment for the selected motion')
        i = int(target[0])
        reader.set_observed_worlds(target)
        h = None
        for step in range(min(a.steps,int(motion.clip_len[a.motion])-1)):
            S = env.state()
            obs = mdp.obs(S,env.a_prev,env.xi,env.f)
            h = obs[:,None].expand(-1,hist,-1).clone() if h is None else torch.cat((h[:,1:],obs[:,None]),1)
            action = policy(h,motion.goal(env.f,frames))
            if a.noise:
                action = (action+a.noise*torch.randn_like(action)).clamp(-20,20)
            out = env.step(action)
            evidence = reader.read()
            row = {key:out[key][i].cpu().clone() for key in ('s','s1','a','a_prev','xi','f','term','done','ep_step','r')}
            row.update({key:value[i].cpu().clone() for key,value in evidence.items()})
            recorded.append(row)
            if step%50==0:
                print(json.dumps(dict(step=step,active=row['active'].tolist(),terminated=bool(row['term']))),flush=True)
            if step%100==0:
                raw=reader.last_evidence.export()
                raw.update({key:reader.last_snapshot[key] for key in ('eligible','groups','robot_side')})
                np.savez_compressed(a.output/f'raw-{step:04d}.npz',**raw)
            if row['done']:
                break
            env.f = torch.minimum(env.f,motion.clip_end[env.task_ids]-1)
        data = {key:torch.stack([row[key] for row in recorded]) for key in recorded[0]}
        data['next_xi'] = torch.cat((data['xi'][1:],env.xi[i:i+1].cpu()),0)
        report = dict(schema='supermimic_native_contact_rollout_v1',motion=a.motion,seed=a.seed,
                      source_checkpoint=str(Path(a.checkpoint).resolve()),noise=a.noise,
                      frames=len(recorded),php_policy_used=False,
                      amp=ck['args'].get('amp','none'),
                      state_quaternion='wxyz',dt=.02,
                      contact_timing='last solver solve paired with post-integration articulation; not strictly same-state',
                      initialization_transfer=transfer,**metadata)
        torch.save(dict(data=data,metadata=report),a.output/'rollout.pt')
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
