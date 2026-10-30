"""Collect one frozen PHP trajectory's real Newton transitions for WM diagnostics.

AppLauncher-first entrypoint. No policy/WM training and no contact relabelling.
"""
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class Options:
    checkpoint:str
    output_dir:Path
    max_steps:int=1200
    num_envs:int=8
    first_motion:int=0
    matched_motion:int=-1
    probe_step:int=-1
    probe_amplitude:float=.02
    action_noise_std:float=0.
    noise_seed:int=17
    save_terrain_atlas:bool=False
    contact_policy:Path|None=None
    planned_contact_points:Path|None=None
    contact_goal_offset:tuple[float,float,float]=(0.,0.,0.)
    support_contract:Literal['fixed_points','sustained_contact']='fixed_points'
    task_initial_recording:Path|None=None
    task_warmup_steps:int=5
    manifest:str='runtime/current/motions/php_stage1_20261025_v1/prepared/train.json'
    teacher_spec:str='runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json'


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args,sync_launcher_headless_config
    launcher=parse_isaaclab_launcher_args(__doc__)
    import tyro
    o=tyro.cli(Options)
    if o.planned_contact_points is not None and o.contact_policy is None:
        raise ValueError('Planned contact execution requires --contact-policy')
    if o.action_noise_std<0 or (o.probe_step>=0 and o.action_noise_std):raise ValueError('Invalid exploration noise')
    import copy,json,numpy as np,torch
    from holosoma.php_contact.distill import Options as TrainOptions,build_config
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_contacts import measured_contacts
    from holosoma.contact_wm.validation import contact_transitions
    cfg=build_config(TrainOptions(manifest=o.manifest,teacher_spec=o.teacher_spec,num_envs=o.num_envs,
        segment_mode='continuous',pure_distill_iters=200,regularizer_mse_coef=.1))
    cfg=cfg.get_eval_config(EvaluationConfig(num_envs=o.num_envs));term=cfg.command.setup_terms['motion_command']
    motion=replace(term.params['motion_config'],start_at_timestep_zero_prob=1.,fixed_motion_assignment_in_eval=True,
        noise_to_initial_pose=replace(term.params['motion_config'].noise_to_initial_pose,overall_noise_scale=0.))
    cfg=replace(cfg,command=replace(cfg.command,setup_terms={'motion_command':replace(term,
        params={**term.params,'motion_config':motion,'segment_start_probability':1.})}),
        randomization=RandomizationManagerCfg(),observation=replace(cfg.observation,groups={
        k:replace(v,enable_noise=False) for k,v in cfg.observation.groups.items()}))
    cfg=sync_launcher_headless_config(cfg,launcher)
    if o.contact_policy is not None:
        from holosoma.config_types.termination import TerminationTermCfg
        terms=dict(cfg.termination.terms)
        terms['bad_tracking']=TerminationTermCfg(func='holosoma.contact_wm.runtime:execution_failure')
        cfg=replace(cfg,termination=replace(cfg.termination,terms=terms))
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/'raw').mkdir()
    class DiagnosticContext(TrainingContext):
        def __exit__(self,exc_type,exc_val,exc_tb):
            if exc_type is not None:
                import traceback
                message=''.join(traceback.format_exception(exc_type,exc_val,exc_tb))
                (out/'error.txt').write_text(message);print(message,flush=True)
            return super().__exit__(exc_type,exc_val,exc_tb)
    with DiagnosticContext(cfg,launcher_args=launcher):
        env=get_class(cfg.env_class)(get_tyro_env_config(cfg),device=launcher.device);env.set_is_evaluating()
        algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(out/'inference'),device=launcher.device)
        algo.setup();algo.load_for_inference(o.checkpoint);algo._eval_mode()
        saved=torch.load(o.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        boundary=copy.copy(algo);boundary.current_learning_iteration=int(saved['iter']);boundary._update_php_student_tracking_boundary()
        cmd=env.command_manager.get_state('motion_command');rows=[[] for _ in range(o.num_envs)];pending={};finished=np.zeros(o.num_envs,bool);capture_step=0
        if o.first_motion<0 or o.first_motion+o.num_envs>cmd.motion.num_motions:raise ValueError('Motion range outside manifest')
        probe_joints=(15,16,18,3)
        if o.matched_motion>=0:
            if o.matched_motion>=cmd.motion.num_motions:raise ValueError('Invalid matched motion')
        if o.probe_step>=0 and (o.num_envs!=9 or o.matched_motion<0):raise ValueError('Paired probe requires 9 matched environments')
        # reset_all reinitializes the command buffers before normal state reset.
        # Assign at that boundary so matched runs actually reset to the same motion.
        original_init_buffers=cmd.init_buffers
        def init_assigned_buffers():
            original_init_buffers()
            assignment=torch.arange(o.first_motion,o.first_motion+o.num_envs,device=env.device)
            if o.matched_motion>=0:assignment.fill_(o.matched_motion)
            cmd.env_motion_ids.copy_(assignment)
            cmd.motion_ids.copy_(assignment)
        cmd.init_buffers=init_assigned_buffers
        def state():return torch.cat((env.simulator.robot_root_states[:],env.simulator.dof_pos,env.simulator.dof_vel),-1).detach().cpu().numpy().copy()
        native=env._update_log_dict
        def capture():
            nonlocal capture_step
            native()
            if not pending or finished.all():return
            data=measured_contacts(env);reader=env._wbt_measured_contact_reader
            evidence=reader.read_evidence()
            snap=reader.export_snapshot(data['snapshot'])
            selected=reader.contact_selector(data['snapshot']);reader.contact_selector.validate(selected)
            count=int(snap['count']);groups=selected['groups'][:count].cpu().numpy();eligible=selected['eligible'][:count].cpu().numpy()
            np.savez_compressed(out/'raw'/f'{capture_step:05d}.npz',**evidence.export())
            next_states=state()
            for i in range(o.num_envs):
                if finished[i]:continue
                points=np.zeros((6,3),np.float32)
                for part in range(6):
                    use=eligible&(groups==i*6+part)
                    if use.any():points[part]=snap['position_w'][use].mean(0)
                row={k:v[i].copy() for k,v in pending.items()}
                row.update(next_state=next_states[i],active=data['active'][i].cpu().numpy().copy(),
                    known=data['known'][i].cpu().numpy().copy(),force_w=data['force_w'][i].cpu().numpy().copy(),
                    solver_point_w=points,terminated=np.asarray(bool(env.termination_manager.terminated[i])),
                    timeout=np.asarray(bool(env.termination_manager.time_outs[i])),reference_frame=np.asarray(int(cmd.time_steps[i])),
                    raw_step=np.asarray(capture_step))
                rows[i].append(row);finished[i]=bool(env.reset_buf[i])
            capture_step+=1
        env._update_log_dict=capture
        obs=env.reset_all(step_after_reset=False)
        mids=cmd.motion_ids.cpu().tolist()
        expected=[o.matched_motion]*o.num_envs if o.matched_motion>=0 else list(range(o.first_motion,o.first_motion+o.num_envs))
        if mids!=expected:raise RuntimeError(f'Motion assignment reset unexpectedly: {mids} != {expected}')
        yy,xx=torch.meshgrid(torch.linspace(-.8,.8,11,device=env.device),torch.linspace(-.5,1.5,11,device=env.device),indexing='ij')
        local=torch.stack((xx.flatten(),yy.flatten(),torch.zeros(121,device=env.device)),-1)
        from holosoma.php_contact.terms import rotate,yaw
        query=env.terrain_manager.get_state('locomotion_terrain').query_terrain_heights
        executor=None;policy_env=None
        if o.contact_policy is not None:
            from holosoma.contact_wm.runtime import ContactExecutor
            policy_meta=torch.load(o.contact_policy,map_location='cpu',weights_only=False)
            policy_env=mids.index(policy_meta['motion_index'])
            executor=ContactExecutor(o.contact_policy,env.device,cmd._reference_env_origins[policy_env],o.contact_goal_offset,
                planned_contact_points=o.planned_contact_points)
        replay_initial=None
        if o.task_initial_recording is not None:
            if executor is None or o.num_envs!=1 or o.task_warmup_steps<1:raise ValueError('Explicit task initialization requires one policy environment and positive warmup')
            replay_initial=np.load(o.task_initial_recording/'transitions.npz')
            replay_meta=json.loads((o.task_initial_recording/'report.json').read_text())
            if replay_meta['motion_index']!=executor.motion_index or replay_meta['joint_names']!=list(env.dof_names):raise ValueError('Initial-state recording mismatch')
            replay_start=executor.start_step-o.task_warmup_steps
            if replay_start<0:raise ValueError('Warmup begins before recording')
            initial=torch.as_tensor(replay_initial['state'][replay_start:replay_start+1],device=env.device)
            ids=torch.arange(1,device=env.device)
            env.simulator.robot_root_states[:]=initial[:,:13]
            env.simulator.set_actor_root_state_tensor_robots(ids,env.simulator.robot_root_states)
            env.simulator.set_dof_state_tensor_robots(ids,torch.stack((initial[:,13:42],initial[:,42:71]),-1))
            cmd.time_steps[:]=cmd.motion.motion_start_idx[cmd.motion_ids]+replay_start
            cmd._set_segments(ids,cmd.segment_offsets[cmd.motion_ids]+cmd.contact_data['segment_id'][cmd.time_steps])
            cmd._update_relative_body_targets()
            env.observation_manager.reset()
            env._refresh_sim_tensors();env._pre_compute_observations_callback();env._compute_observations();env._post_compute_observations_callback();env._clip_observations()
            obs=env.obs_buf_dict
            executor.start_step=o.task_warmup_steps
            np.savez_compressed(out/'requested-initial-state.npz',state=initial.cpu().numpy())
        perception=None
        if executor is not None and executor.perception:
            if replay_initial is None:
                raise ValueError('Panorama task evaluation requires --task-initial-recording for causal pre-task sensor history')
            from holosoma.contact_wm.panorama_sensor import NativePanorama
            # Explicit task reset uses the recorded world frame directly.
            perception=NativePanorama(env,replay_initial,list(env.dof_names),
                torch.zeros(o.num_envs,3,device=env.device),start_frame=replay_start)
            (out/'perception.json').write_text(json.dumps(perception.contract,indent=2))
        if o.save_terrain_atlas:
            minimum=env.simulator.robot_root_states[:,:2]-6.
            ay,ax=torch.meshgrid(torch.arange(241,device=env.device),torch.arange(241,device=env.device),indexing='ij')
            xy=torch.stack((ax,ay),-1)[None]*.05+minimum[:,None,None]
            atlas=query(xy.reshape(-1,2)).reshape(o.num_envs,241,241)
            np.savez_compressed(out/'terrain_atlas.npz',height=atlas.cpu().numpy(),minimum=minimum.cpu().numpy(),resolution=.05)
        noise_rng=torch.Generator(device=env.device).manual_seed(o.noise_seed)
        with torch.inference_mode():
            for step in range(o.max_steps):
                if perception is not None:
                    perception.update(step,torch.as_tensor(state(),device=env.device))
                value=torch.cat([obs[k] for k in algo.actor_obs_keys],-1)
                action=algo.actor.act_inference({'actor_obs':algo._normalize_actor_obs(value,update=False)})
                if o.action_noise_std and executor is None:
                    action=(action+o.action_noise_std*torch.randn(action.shape,device=action.device,generator=noise_rng)).clamp(-10,10)
                if replay_initial is not None and step<o.task_warmup_steps:
                    action[:]=torch.as_tensor(replay_initial['action'][replay_start+step],device=env.device)
                if executor is not None and step>=executor.start_step and not finished[policy_env]:
                    i=policy_env
                    support=torch.as_tensor(rows[i][-1]['solver_point_w'],device=env.device)[None]
                    actual_state=torch.as_tensor(state()[i:i+1],device=env.device)
                    if step==executor.start_step:
                        evidence=measured_contacts(env)
                        np.savez_compressed(out/'task-initial-state.npz',state=actual_state.cpu().numpy(),observation=value[i:i+1].cpu().numpy(),
                            previous_force_w=evidence['force_w'][i:i+1].cpu().numpy(),
                            active=evidence['active'][i:i+1].cpu().numpy(),known=evidence['known'][i:i+1].cpu().numpy())
                    depth_args={} if perception is None else dict(
                        panorama=perception.image[i:i+1],view_pose=perception.view[i:i+1])
                    action[i:i+1]=executor.act(value[i:i+1],actual_state,step,support,**depth_args)
                    if o.action_noise_std:
                        action[i:i+1]=(action[i:i+1]+o.action_noise_std*torch.randn(action[i:i+1].shape,device=action.device,generator=noise_rng)).clamp(-10,10)
                if o.probe_step>=0:
                    action=action[:1].expand(o.num_envs,-1).clone()
                    if step==o.probe_step:
                        same=state();same[:,:3]-=cmd._reference_env_origins.cpu().numpy()
                        mismatch=np.max(np.abs(same-same[:1]),axis=1)
                        (out/'paired-start.json').write_text(json.dumps(dict(max_state_mismatch=mismatch.tolist(),motion_ids=mids)))
                        if mismatch.max()>1e-4:
                            print(f'PAIRED_START_REJECTED {mismatch.tolist()}',flush=True)
                            raise RuntimeError('Paired native starts differ before perturbation')
                        for direction,joint in enumerate(probe_joints):
                            action[1+2*direction,joint]+=o.probe_amplitude
                            action[2+2*direction,joint]-=o.probe_amplitude
                        root_atlas=env.simulator.robot_root_states[:,:2]
                        yy_a,xx_a=torch.meshgrid(torch.arange(101,device=env.device),torch.arange(101,device=env.device),indexing='ij')
                        minimum=root_atlas-2.5
                        xy=torch.stack((xx_a,yy_a),-1)[None]*.05+minimum[:,None,None]
                        atlas=query(xy.reshape(-1,2)).reshape(o.num_envs,101,101)
                        np.savez_compressed(out/'terrain_atlas.npz',height=atlas.cpu().numpy(),minimum=minimum.cpu().numpy(),resolution=.05)
                root=env.simulator.robot_root_states
                world=rotate(local[None].expand(o.num_envs,-1,-1),yaw(root[:,3:7]))+root[:,None,:3]
                heights=(query(world[...,:2].reshape(-1,2)).reshape(o.num_envs,121)-root[:,2:3]).clamp(-2,2)
                force_before=measured_contacts(env)
                pending=dict(state=state(),action=action.cpu().numpy().copy(),observation=value.cpu().numpy().copy(),
                    heights121=heights.cpu().numpy().copy(),previous_force_w=force_before['force_w'].cpu().numpy().copy(),
                    previous_known=force_before['known'].cpu().numpy().copy(),
                    target_active=cmd.goal_active.cpu().numpy().copy(),target_position_w=(cmd.goal_position_w+cmd._reference_env_origins[:,None]).cpu().numpy().copy())
                obs,_,_,_=env.step({'actions':action})
                if finished.all():break
        summaries=[]
        for i,episode in enumerate(rows):
            if not episode:raise RuntimeError('No native transitions captured')
            directory=out/(f'probe_{i:02d}' if o.probe_step>=0 else f'motion_{mids[i]:03d}');directory.mkdir()
            arrays={k:np.stack([r[k] for r in episode]) for k in episode[0]}
            arrays['transition_valid']=~(arrays['terminated']|arrays['timeout'])
            np.savez_compressed(directory/'transitions.npz',**arrays)
            events=contact_transitions(arrays['active'],arrays['known'],minimum_hold=10,minimum_history=10)
            report=dict(schema='contact_wm_native_transitions_v1',frames=len(episode),motion_index=mids[i],checkpoint=o.checkpoint,
                control_dt=float(env.dt),joint_names=list(env.dof_names),default_joint_pos=env.default_dof_pos[0].cpu().tolist(),
                native_backend=env._wbt_measured_contact_reader.backend,raw_directory=str(out/'raw'),
                pose_contact_strict_same_state=False,contact_sampling='last native solve; poses post integration',
                heights121_source='initialized terrain oracle; diagnostic only, not deployable perception',
                state_layout='root xyz,xyzw,linear_velocity_w,angular_velocity_w,joint_position,joint_velocity',
                reference_role='fixed_q intention only',terminated=bool(arrays['terminated'][-1]),
                completed=bool(arrays['timeout'][-1] and not arrays['terminated'][-1]),
                transitions=[dict(arrival=e['arrival'],establish=e['establish'].tolist(),maintain=e['maintain'].tolist(),starts=e['starts']) for e in events],
                force_and_slip_success_not_yet_validated=True,robot_asset=saved['robot_asset'],
                reference_origin_w=cmd._reference_env_origins[i].cpu().tolist(),probe_step=o.probe_step,
                probe_amplitude=o.probe_amplitude,probe_joints=list(probe_joints),probe_env=i,
                action_noise_std=o.action_noise_std,noise_seed=o.noise_seed,
                probe_actions='baseline environment shared action sequence; one signed pulse' if o.probe_step>=0 else None)
            if executor is not None and i==policy_env:
                report['contact_policy']=str(o.contact_policy)
                report['execution_start']=executor.start_step
                report['execution_horizon']=executor.horizon
                report['execution_goal']={k:v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in vars(executor.goal).items()}
                report['contact_goal_offset']=list(o.contact_goal_offset)
                report['support_contract']=o.support_contract
                report['task_initial_recording']=str(o.task_initial_recording) if o.task_initial_recording else None
                report['task_warmup_steps']=o.task_warmup_steps if replay_initial is not None else None
                report['failure_contract']='nonfinite root, root up dot world up < .2, or root world z < .15; no reference tracking termination'
            (directory/'report.json').write_text(json.dumps(report,indent=2));summaries.append({k:report[k] for k in ('motion_index','frames','completed','terminated','transitions')})
        (out/'summary.json').write_text(json.dumps(summaries,indent=2))
        print('CONTACT_WM_COLLECTION '+json.dumps(summaries),flush=True)

if __name__=='__main__':main()
