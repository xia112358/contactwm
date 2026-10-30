"""Persistent native collection with SuperMimic-frequency WM/policy alternation."""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Literal


def wm_update_count(mode, batches):
    """Number of dynamics-only updates before each policy update."""
    if batches < 1:raise ValueError('WM update ratio must be positive')
    if mode == 'frozen':return 0
    if mode == 'alternating':return batches
    raise ValueError('Invalid WM schedule')


def restored_wm_updates(checkpoint, default_batches):
    """Legacy counts use the saved ratio, never the new experiment's ratio."""
    return int(checkpoint.get('wm_updates',int(checkpoint['iteration'])*
        checkpoint.get('options',{}).get('wm_batches',default_batches)))

@dataclass(frozen=True)
class Options:
    checkpoint: str
    wm_checkpoint: Path
    recording: Path
    atlas: Path
    output_dir: Path
    num_envs: int = 4096
    panorama: bool = False
    planned_contact_points: Path | None = None
    reference_contact_frame: int = 286
    contact_deadline_delay: int = 5
    iterations: int = 20000
    buffer_steps: int = 256
    warmup_steps: int = 64
    wm_batches: int = 10
    wm_schedule: Literal['alternating','frozen'] = 'alternating'
    freeze_policy: bool = False
    export_native_replay: bool = False
    spatial_contact_slots: int = 0
    policy_learning_rate: float | None = None
    wm_dynamics_lr: float | None = None
    fixed_probe_every: int = 0
    sampling_seed: int | None = None
    wm_batch: int = 2048
    wm_window: int = 8
    wm_holdout_envs: int = 64
    wm_motion_envs: int = 128
    wm_audit_every: int = 10
    wm_audit_batch: int = 128
    wm_event_fraction: float = .5
    wm_motion_fraction: float = .25
    wm_activation_lr: float = 2e-4
    wm_max_root_error: float = .03
    wm_max_witness_error: float = .03
    wm_max_joint_error: float = .10
    wm_max_rotation_error: float = .15
    wm_max_force_error: float = 100.
    wm_max_contact_error: float = .10
    target_update_scale: float = 1.
    target_tangent_scale: float = .05
    policy_groups: int = 455
    compile_kernels: bool = True
    benchmark_iterations: int = 0
    validation_iterations: int = 8
    max_steady_iteration_seconds: float = 3.
    save_every: int = 100
    noise: float = .1
    resume: Path | None = None
    initialize_wm: Path | None = None
    manifest: str = 'runtime/current/motions/php_stage1_20261025_v1/prepared/train.json'
    teacher_spec: str = 'runtime/current/motions/php_stage1_20261025_v1/prepared/teachers.json'


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args(__doc__)
    import tyro
    o = tyro.cli(Options)
    if o.wm_holdout_envs<2 or o.wm_motion_envs<1 or o.num_envs<=9+o.wm_holdout_envs+o.wm_motion_envs:
        raise ValueError('Need 9 native evaluation envs, WM holdout envs and training envs')
    if o.wm_audit_every<1 or o.wm_audit_batch<1 or not 0<=o.wm_event_fraction<=1 or o.target_update_scale<=0:
        raise ValueError('Invalid WM audit/sampling or policy update options')
    if not 0<o.wm_motion_fraction<1 or o.wm_activation_lr<=0:raise ValueError('Invalid native Motion mixture or activation-head rate')
    if o.resume is not None and o.initialize_wm is not None:
        raise ValueError('Use either full resume or WM-only initialization')
    import json, time, signal, traceback
    import numpy as np
    import torch
    import math
    if o.wm_batches<1 or o.fixed_probe_every<0:
        raise ValueError('Invalid update schedule or probe frequency')
    if o.policy_learning_rate is not None and (not math.isfinite(o.policy_learning_rate) or o.policy_learning_rate<=0):
        raise ValueError('Policy learning rate must be finite and positive')
    if o.wm_dynamics_lr is not None and (not math.isfinite(o.wm_dynamics_lr) or o.wm_dynamics_lr<=0):
        raise ValueError('WM dynamics learning rate must be finite and positive')
    if o.panorama and o.fixed_probe_every:
        raise ValueError('Fixed reference probe currently requires height observations')
    from holosoma.contact_wm.train_policy import Options as LearnerOptions, build_training
    from holosoma.contact_wm.core import ContactGoal, fit_loss
    from holosoma.contact_wm.runtime import goal_observation
    from holosoma.contact_wm.target_response import target_grid
    from holosoma.contact_wm.stream_replay import Replay
    from holosoma.php_contact.distill import Options as TrainOptions, build_config
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.config_types.termination import TerminationTermCfg
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_contacts import measured_contacts
    from holosoma.php_contact.terms import rotate, yaw
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    def status(**fields):
        payload=dict(num_envs=o.num_envs,wm_schedule=o.wm_schedule,
            wm_updates_per_iteration=fields.pop('wm_updates_this_iteration',0),
            policy_updates_per_iteration=int(fields.get('policy_updated',False)),policy_updates_per_iteration_max=int(not o.freeze_policy),policy_batch=o.policy_groups*9,
            native_steps_per_iteration=[2,3],perception='historical_panorama_depth_128x64' if o.panorama else 'oracle_height',updated_at=time.time(),**fields)
        p=out/'status.tmp';p.write_text(json.dumps(payload,indent=2));p.replace(out/'status.json')
    status(status='starting',phase='initialization',iteration=0)
    (out/'config.json').write_text(json.dumps(asdict(o),default=str,indent=2))
    cfg=build_config(TrainOptions(manifest=o.manifest,teacher_spec=o.teacher_spec,num_envs=o.num_envs,
        segment_mode='continuous',pure_distill_iters=200,regularizer_mse_coef=.1))
    cfg=cfg.get_eval_config(EvaluationConfig(num_envs=o.num_envs))
    term=cfg.command.setup_terms['motion_command']
    motion=replace(term.params['motion_config'],start_at_timestep_zero_prob=1.,fixed_motion_assignment_in_eval=True,
        noise_to_initial_pose=replace(term.params['motion_config'].noise_to_initial_pose,overall_noise_scale=0.))
    cfg=replace(cfg,command=replace(cfg.command,setup_terms={'motion_command':replace(term,
        params={**term.params,'motion_config':motion,'segment_start_probability':1.})}),
        randomization=RandomizationManagerCfg(),observation=replace(cfg.observation,groups={
        k:replace(v,enable_noise=False) for k,v in cfg.observation.groups.items()}),
        termination=replace(cfg.termination,terms={**cfg.termination.terms,
        'bad_tracking':TerminationTermCfg(func='holosoma.contact_wm.runtime:execution_failure')}))
    cfg=sync_launcher_headless_config(cfg,launcher)
    class Context(TrainingContext):
        def __exit__(self,kind,error,tb):
            if kind:
                message=''.join(traceback.format_exception(kind,error,tb))
                (out/'error.txt').write_text(message);print(message,flush=True)
                status(status='failed',phase='exception',error=str(error))
            return super().__exit__(kind,error,tb)
    with Context(cfg,launcher_args=launcher):
        env=get_class(cfg.env_class)(get_tyro_env_config(cfg),device=launcher.device);env.set_is_evaluating()
        device=env.device;n=env.num_envs
        if n!=o.num_envs:raise RuntimeError(f'Actual environment count {n} differs from requested {o.num_envs}')
        l=build_training(LearnerOptions(recording=o.recording,atlas=o.atlas,wm_checkpoint=o.wm_checkpoint,
            output_dir=out/'policy',device=str(device),source_wm=True,multi_target=True,panorama=o.panorama,
            learning_rate=1e-6,target_tangent_scale=o.target_tangent_scale,goal_offset=(0.,0.,0.) if o.planned_contact_points is not None else (.05,0.,0.),
            planned_contact_points=o.planned_contact_points,arrival=o.reference_contact_frame,
            arrival_delay=o.contact_deadline_delay,
            target_learning_rate_scale=o.target_update_scale))
        if l['horizon']!=35:raise ValueError('This stream task requires a 35-step contact horizon; adjust deadline delay to the measured reference event')
        if Path(l['meta']['checkpoint']).resolve()!=Path(o.checkpoint).resolve():raise ValueError('Prior checkpoint mismatch')
        from somaforge_core.collision_profiles import require_profile, execution_profile
        require_profile(l['meta']['robot_asset'],execution_profile(env),context='Stream reference versus actual physics')
        require_profile(l['provenance']['robot_asset'],execution_profile(env),context='Stream policy versus actual physics')
        policy,wm,opt=l['policy'],l['wm'],l['optimizer']
        activation=list(wm.activation_head.parameters())
        dynamics=[p for name,p in wm.named_parameters() if not name.startswith('activation_head.')]
        wmopt=torch.optim.Adam([{'params':dynamics,'lr':2e-5},{'params':activation,'lr':o.wm_activation_lr}],fused=True)
        (out/'transfer.json').write_text(json.dumps(wm.transfer_report,indent=2))
        if o.initialize_wm is not None:
            adapted=torch.load(o.initialize_wm,map_location=device,weights_only=False)
            if adapted['schema']!='contact_wm_stream_v1':raise ValueError('Invalid WM initialization checkpoint')
            if adapted['robot_asset']!=l['provenance']['robot_asset']:raise ValueError('WM robot asset mismatch')
            wm.load_state_dict(adapted['wm'],strict=True)
            if hasattr(wm,'world_field'):
                wmopt.add_param_group({'params':wm.world_field.encoder.parameters(),'lr':1e-3})
                wmopt.add_param_group({'params':wm.world_field.feedback.parameters(),'lr':2e-5})
            (out/'wm-initialization.json').write_text(json.dumps(dict(
                source=str(o.initialize_wm.resolve()),source_iteration=int(adapted['iteration']),
                policy='original Stage 1 height-conditioned prior; fresh policy optimizer',
                wm='adapted dynamics weights; fresh WM optimizer'),indent=2))
            del adapted
        resumed_iteration=0
        if o.resume is not None:
            resumed=torch.load(o.resume,map_location=device,weights_only=False)
            if resumed['schema']!='contact_wm_stream_v1':raise ValueError('Invalid stream checkpoint')
            from somaforge_core.robot_assets import validate_g1_asset_metadata
            validate_g1_asset_metadata(resumed.get('robot_asset'),context='Contact WM stream checkpoint')
            require_profile(resumed.get('robot_asset'),execution_profile(env),context='Stream resume versus actual physics')
            policy.load_state_dict(resumed['policy']);wm.load_state_dict(resumed['wm'])
            opt.load_state_dict(resumed['optimizer'])
            if hasattr(wm,'world_field'):
                wmopt.add_param_group({'params':wm.world_field.encoder.parameters(),'lr':1e-3})
                wmopt.add_param_group({'params':wm.world_field.feedback.parameters(),'lr':2e-5})
            wm_state=resumed['wm_optimizer']
            if len(wm_state['param_groups'])==1:
                # Legacy optimizer used model order, with activation head last.
                group=wm_state['param_groups'][0];ids=group['params'];cut=len(dynamics)
                if len(ids)!=cut+len(activation):raise ValueError('Legacy WM optimizer parameter count differs')
                wm_state={**wm_state,'param_groups':[{**group,'params':ids[:cut]},
                    {**group,'params':ids[cut:],'lr':o.wm_activation_lr}]}
            wmopt.load_state_dict(wm_state)
            wmopt.param_groups[1]['lr']=o.wm_activation_lr
            resumed_iteration=int(resumed['iteration'])
            torch.set_rng_state(resumed['torch_rng_state'].cpu())
            torch.cuda.set_rng_state_all([r.cpu() for r in resumed['cuda_rng_state']])
        # Override AFTER restoring Adam moments; otherwise resume silently restores
        # the old rate and an apparent learning-rate experiment is ineffective.
        if o.policy_learning_rate is not None:
            for group in opt.param_groups:group['lr']=o.policy_learning_rate
        if o.wm_dynamics_lr is not None:wmopt.param_groups[0]['lr']=o.wm_dynamics_lr
        if (hasattr(wm,'world_field') and o.iterations>resumed_iteration and
                o.wm_schedule=='alternating' and o.spatial_contact_slots<1):
            raise ValueError('Spatial WM fitting requires actual labels: --spatial-contact-slots must be positive')
        wm_updates=restored_wm_updates(resumed,o.wm_batches) if o.resume is not None else 0
        run_wm_updates=0
        frozen_policy_state=None
        if o.freeze_policy:
            frozen_policy_state={k:v.detach().clone() for k,v in policy.state_dict().items()}
            policy.requires_grad_(False)
        from holosoma.contact_wm.batched_learner import BatchedLearner
        learner=BatchedLearner(l,o.atlas,compile_kernels=o.compile_kernels,wm_window=o.wm_window)
        cmd=env.command_manager.get_state('motion_command')
        original_init=cmd.init_buffers
        def init_buffers():
            original_init();cmd.env_motion_ids.fill_(l['meta']['motion_index']);cmd.motion_ids.fill_(l['meta']['motion_index'])
        cmd.init_buffers=init_buffers
        obs=env.reset_all(step_after_reset=False)
        origin=torch.tensor(l['meta']['reference_origin_w'],device=device)
        shifts=cmd._reference_env_origins-origin
        age=torch.zeros(n,device=device,dtype=torch.long)
        source=l['z'];start=261
        initial=torch.tensor(source['state'][start],device=device).expand(n,-1).clone();initial[:,:3]+=shifts
        replay_actions=torch.tensor(source['action'][start:start+5],device=device)
        motion_actions=torch.tensor(source['action'][start:start+40],device=device)
        if len(motion_actions)!=40:raise ValueError('Need a full native Motion action prefix for WM coverage')
        motion_env=torch.zeros(n,device=device,dtype=torch.bool)
        motion_env[9:9+o.wm_holdout_envs//2]=True
        motion_env[9+o.wm_holdout_envs:9+o.wm_holdout_envs+o.wm_motion_envs]=True
        goal=ContactGoal(**{k:v[:1].expand(n,*v.shape[1:]).clone() if isinstance(v,torch.Tensor) else v for k,v in vars(l['goal']).items()})
        goal.position_w=l['base_target'][:1].expand(n,-1,-1).clone()+shifts[:,None]
        support=goal.support_position_w.clone()+shifts[:,None]
        goal.support_position_w=support
        offsets=target_grid(l['target_span'],device)[torch.arange(n,device=device)%9]
        goal.position_w+=goal.establish[...,None]*offsets[:,None]
        (out/'contact-targets.json').write_text(json.dumps(dict(
            source='explicit_planned_world_points' if o.planned_contact_points is not None else 'reference_contact_with_sampled_targets',
            input_file=str(o.planned_contact_points.resolve()) if o.planned_contact_points is not None else None,
            frame='recording world frame; environment origins translated together',
            position_w=(goal.position_w[:9]-shifts[:9,None]).cpu().tolist(),
            establish=goal.establish[:9].cpu().tolist(),
            objective='absolute contact position during arrival/hold; no response loss'),indent=2))
        hint=l['hint'][:1].expand(n,-1)
        perception=None
        original_reset=env.reset_envs_idx
        def reset(ids,*args,**kwargs):
            original_reset(ids,*args,**kwargs)
            if not len(ids):return
            env.simulator.robot_root_states[ids]=initial[ids,:13]
            env.simulator.set_actor_root_state_tensor_robots(ids,env.simulator.robot_root_states)
            env.simulator.set_dof_state_tensor_robots(ids,torch.stack((initial[:,13:42],initial[:,42:71]),-1))
            # Simulator joint arrays use advanced indexing and are cached copies.
            # The normal post-reset refresh writes these arrays back into physics.
            env.simulator.dof_pos[ids]=initial[ids,13:42]
            env.simulator.dof_vel[ids]=initial[ids,42:71]
            cmd.time_steps[ids]=cmd.motion.motion_start_idx[cmd.motion_ids[ids]]+start
            cmd._set_segments(ids,cmd.segment_offsets[cmd.motion_ids[ids]]+cmd.contact_data['segment_id'][cmd.time_steps[ids]])
            cmd._update_relative_body_targets();age[ids]=0
            if perception is not None:perception.reset(ids,native_steps*env.dt)
        env.reset_envs_idx=reset
        reset(torch.arange(n,device=device))
        def observations():
            env._refresh_sim_tensors();env._pre_compute_observations_callback();env._compute_observations()
            env._post_compute_observations_callback();env._clip_observations()
            return env.obs_buf_dict
        obs=observations()
        def state():
            s=torch.cat((env.simulator.robot_root_states[:],env.simulator.dof_pos,env.simulator.dof_vel),-1).clone()
            s[:,:3]-=shifts;return s
        yy,xx=torch.meshgrid(torch.linspace(-.8,.8,11,device=device),torch.linspace(-.5,1.5,11,device=device),indexing='ij')
        local=torch.stack((xx.flatten(),yy.flatten(),torch.zeros(121,device=device)),-1)
        query=env.terrain_manager.get_state('locomotion_terrain').query_terrain_heights
        def height():
            root=env.simulator.robot_root_states[:]
            world=rotate(local[None].expand(n,-1,-1),yaw(root[:,3:7]))+root[:,None,:3]
            return (query(world[...,:2].reshape(-1,2)).reshape(n,121)-root[:,2:3]).clamp(-2,2)
        replay=Replay(o.buffer_steps,n,device)
        holdout_end=9+o.wm_holdout_envs
        heldout=Replay(o.buffer_steps,o.wm_holdout_envs,device)
        # Fixed native reference is never supplied to WM fitting. It catches
        # forgetting/narrow reset-distribution fits that live replicas can miss.
        reference=Replay(35,1,device)
        if len(source['state'])<301:raise ValueError('Need native frames 266..300 for fixed-reference validation')
        for t in range(266,301):
            row={k:torch.as_tensor(source[k][t:t+1],device=device) for k in
                ('state','next_state','action','active','known','force_w','previous_known','previous_force_w')}
            row.update(height=torch.as_tensor(source['heights121'][t:t+1],device=device),
                previous_active=torch.as_tensor(source['observation'][t:t+1,:6],device=device).bool(),
                valid=torch.as_tensor(source['transition_valid'][t:t+1],device=device).bool(),
                age=torch.tensor([t-start],device=device),ended=torch.as_tensor(source['terminated'][t:t+1]|source['timeout'][t:t+1],device=device))
            reference.add(row)
        from holosoma.contact_wm.dynamics_audit import DynamicsLimits,audit_dynamics,policy_gate
        limits=DynamicsLimits(o.wm_max_root_error,o.wm_max_witness_error,o.wm_max_joint_error,
            o.wm_max_rotation_error,o.wm_max_force_error,o.wm_max_contact_error,o.wm_max_contact_error)
        gate=dict(allowed=False,reasons=['WM not yet validated'])
        policy_updates=int(resumed.get('policy_updates',resumed_iteration)) if o.resume is not None else 0
        run_policy_updates=0
        def audit(iteration):
            nonlocal gate
            report=audit_dynamics(learner,heldout,o.wm_audit_batch)
            fixed_report=audit_dynamics(learner,reference,min(32,o.wm_audit_batch))
            gate=policy_gate(report,limits,reference=fixed_report)
            report['fixed_reference']=fixed_report
            report.update(iteration=iteration,native_steps=native_steps,gate=gate)
            path=out/'wm-audit.tmp';path.write_text(json.dumps(report,allow_nan=False));path.replace(out/'wm-audit-latest.json')
            with (out/'wm-audits.jsonl').open('a') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
        native_metrics={}
        import copy
        from holosoma.contact_wm.native_monitor import NativeMonitor
        eval_policy=copy.deepcopy(policy).requires_grad_(False)
        monitor=NativeMonitor(out,l['meta']['joint_names'],
            '/home/zhouxia/sites/acp-history-player-20261025/dist/robot.json',o.atlas)
        policy_version=resumed_iteration
        pending={};captured={};native_steps=0;fixed_pano=None
        if o.panorama:
            from holosoma.contact_wm.panorama_sensor import NativePanorama
            from holosoma.contact_wm.panorama_policy import PanoramaReplay,relative_view
            status(status='starting',phase='causal_history_initialization',iteration=0,actual_num_envs=n)
            perception=NativePanorama(env,source,l['meta']['joint_names'],shifts)
            image_bank=PanoramaReplay((o.buffer_steps+4)//5+2,n,device)
            actor_encode=torch.compile(policy.encode,fullgraph=True) if o.compile_kernels else policy.encode
            (out/'perception.json').write_text(json.dumps(perception.contract,indent=2))
        original_log=env._update_log_dict
        def capture():
            original_log()
            if not pending:return
            data=measured_contacts(env);reader=env._wbt_measured_contact_reader
            selected=reader.contact_selector(data['snapshot']);reader.contact_selector.validate(selected)
            count=int(data['snapshot']['count'])
            groups=selected['groups'][:count].clamp(0,n*6-1)
            eligible=selected['eligible'][:count]
            points=torch.zeros(n*6,3,device=device);counts=torch.zeros(n*6,device=device)
            points.index_add_(0,groups,torch.where(eligible[:,None],data['snapshot']['position_w'][:count],0.))
            counts.index_add_(0,groups,eligible.float())
            points=(points/counts.clamp_min(1)[:,None]).reshape(n,6,3)-shifts[:,None]
            # Keep raw solver point pairs for a sampled audit, never alter their labels.
            if native_steps%1000==0:
                np.savez_compressed(out/f'native-audit-{native_steps:07d}.npz',**reader.read_evidence().export())
            next_state=state()
            # This hook runs before reset. A finite terminal transition is valid
            # dynamics evidence; only continuing through the reset is forbidden.
            finite=torch.isfinite(next_state).all(-1)&torch.isfinite(pending['state']).all(-1)&torch.isfinite(pending['action']).all(-1)
            finite&=torch.where(data['known'][...,None],torch.isfinite(data['force_w']),True).all((1,2))
            captured.clear();captured.update(next_state=next_state,active=data['active'].clone(),known=data['known'].clone(),
                force_w=data['force_w'].clone(),valid=finite)
            if o.spatial_contact_slots:
                from holosoma.contact_wm.world_contact_field import pack_native_contacts
                captured.update(pack_native_contacts(data['snapshot'],selected,shifts,n,o.spatial_contact_slots))
            with torch.no_grad():
                witness_points=l['geometry'](captured['next_state'])
                error=(witness_points[:,2]+shifts-goal.position_w[:,2]).norm(dim=-1)
                policy_env=~motion_env
                hold=(age>=5+goal.arrival_step-1)&policy_env
                native_metrics.update(native_contact_fraction=float(data['active'][policy_env,2].float().mean()),
                    native_support_fraction=float((data['active'][policy_env,3]&data['active'][policy_env,4]).float().mean()),
                    native_motion_contact_fraction=float(data['active'][motion_env,2].float().mean()),
                    native_diagnostic_scope='policy environments only; Motion replay reported separately',
                    native_hold_samples=int(hold.sum()),native_hold_error_m=float(error[hold].mean()) if hold.any() else None)
            end=age>=39
            env.reset_buf|=end
            captured['ended']=env.reset_buf.bool().clone()
            captured['event']=((data['active']!=pending['previous_active'])&data['known']&pending['previous_known']).any(-1)|env.termination_manager.terminated.bool()
            monitor.add(captured['next_state'],pending['action'],points,data['active'],data['known'],
                data['force_w'],env.termination_manager.terminated)
        env._update_log_dict=capture
        stopped=False
        def stop(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        def collect(count):
            nonlocal obs,native_steps,fixed_pano
            with torch.no_grad():
                for _ in range(count):
                    if native_steps%40==0:
                        reset(torch.arange(9,device=device));obs=observations()
                        eval_policy.load_state_dict(policy.state_dict())
                        normalized_goal=copy.deepcopy(goal)
                        normalized_goal.position_w-=shifts[:,None]
                        normalized_goal.support_position_w-=shifts[:,None]
                        monitor.begin(policy_version,normalized_goal)
                    s=state();raw=torch.cat((obs['actor_obs'],obs['actor_proprio_history']),-1)
                    evidence=measured_contacts(env)
                    # Re-anchor sustained supports to actual robot-side witness at task entry.
                    ids=age==5
                    support[ids]=l['geometry'](s[ids])+shifts[ids,None]
                    world=s.clone();world[:,:3]+=shifts
                    value,extra=goal_observation(raw,world,goal,hint,0)
                    extra[:,-1]=(goal.arrival_step-(age-5)).clamp_min(0)*.02
                    if o.panorama:
                        if perception.update(native_steps,s):image_bank.add(perception.image)
                        features=actor_encode(perception.image)
                        action=policy.forward_encoded(value,extra,features,relative_view(s,perception.view))
                    else:action=policy(value,extra)
                    action=(action+o.noise*torch.randn_like(action)).clamp(-10,10)
                    if o.panorama:
                        action[:9]=eval_policy(value[:9],extra[:9],panorama=perception.image[:9],state=s[:9],view_pose=perception.view[:9])
                    else:action[:9]=eval_policy(value[:9],extra[:9])
                    # Native reference-action executions supply contact-mode
                    # coverage to WM; deployment and the nine policy tests do
                    # not use this training-only action replay.
                    action[motion_env]=motion_actions[age[motion_env]]
                    warm=age<5;action[warm]=replay_actions[age[warm]]
                    pending.clear();pending.update(state=s,action=action,observation=raw.clone(),height=height(),
                        previous_force_w=evidence['force_w'].clone(),previous_known=evidence['known'].clone(),
                        previous_active=evidence['active'].clone(),
                        motion_replay=motion_env.clone(),
                        support=support.clone()-shifts[:,None],age=age.clone())
                    if o.panorama:
                        pending.update(pano_frame=torch.full((n,),image_bank.frame,dtype=torch.long,device=device),
                            pano_view=perception.view.clone(),env_id=torch.arange(n,device=device))
                        if fixed_pano is None and int(age[0])==5:
                            fixed_pano={k:v[:1].clone() for k,v in pending.items()}
                            fixed_pano['panorama']=perception.image[:1].clone()
                    oldage=age.clone();obs,_,_,_=env.step({'actions':action})
                    if not captured:raise RuntimeError('Missing pre-reset native capture')
                    reset_ids=env.reset_buf.bool()
                    if reset_ids.any():
                        restored=state()[reset_ids,13:71]
                        expected=initial[reset_ids,13:71]
                        if not torch.allclose(restored,expected,atol=1e-5,rtol=0):
                            raise RuntimeError(f'Post-reset joint state overwritten: {float((restored-expected).abs().max())}')
                    row={**pending,**captured}
                    heldout.add({k:v[9:holdout_end] for k,v in row.items()})
                    row={**row,'valid':row['valid'].clone()}
                    row['valid'][:holdout_end]=False  # Evaluation and WM validation never enter training.
                    replay.add(row)
                    age.copy_(torch.where(env.reset_buf.bool(),0,oldage+1))
                    native_steps+=1
                    if native_steps%40==0:
                        report=monitor.finish()
                        native_metrics.update(native_successes=report['successes'],native_trials=9,
                            native_evaluation_iteration=report['iteration'],native_contact_error_m=report['mean_contact_error_m'])
        def update_wm():
            nm=max(1,min(o.wm_batch-1,int(o.wm_batch*o.wm_motion_fraction)))
            motion=replay.windows(nm,o.wm_window,event_fraction=o.wm_event_fraction,motion_replay=True)
            policy_data=replay.windows(o.wm_batch-nm,o.wm_window,event_fraction=o.wm_event_fraction,motion_replay=False)
            keys=('state','action','height','previous_force_w','next_state','active','force_w','known','valid')
            if hasattr(wm,'world_field'):
                keys+=('spatial_points_w','spatial_forces_w','spatial_mask')
            x={k:torch.cat((motion[k],policy_data[k]),1) for k in keys}
            if hasattr(wm,'world_field'):
                with torch.no_grad():
                    labels=wm.world_field.targets(x['spatial_points_w'].flatten(0,1),x['spatial_forces_w'].flatten(0,1),
                        x['spatial_mask'].flatten(0,1),x['known'].flatten(0,1),x['valid'].flatten(0,1))
                if not bool(labels[3].all()):
                    raise ValueError('Actual contact outside fixed world query patches; expand the external surface representation')
                for name,value in zip(('spatial_target_mass','spatial_target_force','spatial_observed'),labels[:3]):
                    x[name]=value.reshape(o.wm_window,o.wm_batch,*value.shape[1:])
            wmopt.zero_grad(set_to_none=True)
            loss=learner.wm_rollout(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True)
            wmopt.step()
            return loss.detach()
        def update_policy():
            from holosoma.contact_wm.core import frozen_parameters
            sample=replay.starts(o.policy_groups)
            if o.panorama:sample['panorama']=image_bank.gather(sample['pano_frame'],sample['env_id'])
            opt.zero_grad(set_to_none=True)
            with frozen_parameters(wm):
                values,valid=learner.policy_loss(sample)[:2]
                if not bool(valid):raise RuntimeError('Invalid imagined terrain or nonfinite policy loss')
                values[0].backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
            before=policy.net[0].weight[:,12:36].detach().clone();opt.step()
            with torch.no_grad():
                w=policy.net[0].weight[:,12:36];w.add_((o.target_update_scale-1)*(w-before))
            return dict(zip(learner.names,values.detach().cpu().tolist()),policy_grad_norm=float(grad_norm))
        fixed=dict(state=torch.tensor(source['state'][266:267],device=device),
            observation=torch.tensor(source['observation'][266:267],device=device),
            previous_known=torch.tensor(source['previous_known'][266:267],device=device),
            previous_force_w=torch.tensor(source['previous_force_w'][266:267],device=device),
            support=torch.tensor(source['solver_point_w'][265:266],device=device))
        probe=None
        if o.fixed_probe_every:
            frames=np.array([266,268,270,272,274,276])
            probe={k:torch.as_tensor(source[k][frames],device=device) for k in
                ('state','observation','previous_known','previous_force_w')}
            with torch.no_grad():anchor=l['geometry'](torch.as_tensor(source['state'][266:267],device=device))
            probe.update(support=anchor.expand(len(frames),-1,-1).clone(),age=torch.as_tensor(frames-261,device=device))
        def fixed_probe():
            # Eager, deterministic, identical states/goals across all evaluations.
            with torch.no_grad():values,valid=learner.policy_loss(probe,compiled=False)[:2]
            if not bool(valid):raise RuntimeError('Invalid fixed diagnostic probe')
            return dict(zip(learner.names,values.cpu().tolist()))
        def save(iteration):
            if o.panorama:
                if fixed_pano is None:raise RuntimeError('No causal sensor-conditioned task start available for checkpoint evaluation')
                from holosoma.contact_wm.target_response import response_metrics
                with torch.no_grad():
                    _,valid,endpoints=learner.policy_loss(fixed_pano,compiled=False)
                    target=learner.prepare(fixed_pano)['target'][:,2]
                    response=response_metrics(endpoints,target)
                (out/'policy'/f'response_{iteration:04d}.json').write_text(json.dumps(response,indent=2))
                torch.save(dict(schema='contact_wm_direct_policy_v1',policy=policy.state_dict(),optimizer=opt.state_dict(),
                    iteration=iteration,prior=l['provenance'],wm_checkpoint=str(o.wm_checkpoint),perception=policy.config,
                    goal={k:(l['base_target'][:1] if k=='position_w' else v[:1]).detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in vars(l['goal']).items()},
                    goal_hint=l['hint'][:1].cpu(),local_witness=l['local'].cpu(),start_step=266,motion_index=l['meta']['motion_index'],
                    horizon=35,reference_origin_w=l['meta']['reference_origin_w'],torch_rng_state=torch.get_rng_state(),
                    options=asdict(o),native_success_verified=False),out/'policy'/f'policy_{iteration:04d}.pt')
            else:
                l['set_start'](fixed,5)
                with torch.no_grad():
                    result=l['rollout']();l['export'](iteration,result)
                del result
            # No solver snapshot: resume must reset physics and warm up replay normally.
            torch.save(dict(schema='contact_wm_stream_v1',iteration=iteration,native_steps=native_steps,
                policy=policy.state_dict(),wm=wm.state_dict(),optimizer=opt.state_dict(),wm_optimizer=wmopt.state_dict(),
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                options=asdict(o),prior=l['provenance'],robot_asset=l['provenance']['robot_asset'],
                policy_updates=policy_updates,
                wm_updates=wm_updates,
                native_success_verified=False),out/f'checkpoint_{iteration:06d}.pt')
        status(status='running',phase='native_warmup',iteration=0,actual_num_envs=n)
        collect(o.warmup_steps);save(resumed_iteration)
        if o.freeze_policy:audit(resumed_iteration)
        status(status='running',phase='alternating',iteration=resumed_iteration,actual_num_envs=n)
        begin=time.time();iteration=resumed_iteration;steady_times=[]
        with (out/'metrics.jsonl').open('w') as log:
            while iteration<o.iterations and not stopped:
                # Synchronize only stage boundaries, so wall times include actual GPU work.
                torch.cuda.synchronize();tick=time.perf_counter()
                # Controlled comparisons: changing WM minibatch count must not
                # change collection noise or policy-start sampling randomness.
                if o.sampling_seed is not None:torch.manual_seed(o.sampling_seed+3*iteration)
                collect(2+iteration%2)
                torch.cuda.synchronize();collected=time.perf_counter()
                probing=probe is not None and (iteration-resumed_iteration)%o.fixed_probe_every==0
                before_wm=fixed_probe() if probing else None
                count=wm_update_count(o.wm_schedule,o.wm_batches)
                if o.sampling_seed is not None:torch.manual_seed(o.sampling_seed+3*iteration+1)
                losses=[update_wm() for _ in range(count)]
                wm_updates+=count;run_wm_updates+=count
                after_wm=fixed_probe() if probing else None
                torch.cuda.synchronize();modeled=time.perf_counter()
                if (iteration-resumed_iteration)%o.wm_audit_every==0:audit(iteration)
                torch.cuda.synchronize();audited=time.perf_counter()
                updated=gate['allowed'] and not o.freeze_policy
                if o.sampling_seed is not None:torch.manual_seed(o.sampling_seed+3*iteration+2)
                metrics=update_policy() if updated else {name:None for name in learner.names}
                if probing:
                    after_policy=fixed_probe()
                    probe_row=dict(iteration=iteration+1,before_wm=before_wm,after_wm=after_wm,after_policy=after_policy,
                        wm_delta=after_wm['target_geometry']-before_wm['target_geometry'],
                        policy_delta=after_policy['target_geometry']-after_wm['target_geometry'],
                        policy_updated=updated,wm_updates=count)
                    with (out/'fixed-probes.jsonl').open('a') as f:f.write(json.dumps(probe_row)+'\n')
                policy_updates+=int(updated)
                run_policy_updates+=int(updated)
                torch.cuda.synchronize();finished=time.perf_counter()
                iteration+=1
                if updated:policy_version=iteration
                compiling=iteration-resumed_iteration<=2 or (updated and run_policy_updates<=2)
                if not compiling:steady_times.append(finished-tick)
                row=dict(compile_warmup=compiling,iteration=iteration,step=iteration,native_steps=native_steps,native_transitions=native_steps*n,
                    wm_updates=wm_updates,run_wm_updates=run_wm_updates,wm_updates_this_iteration=count,
                    policy_learning_rates=[group['lr'] for group in opt.param_groups],
                    wm_learning_rates=[group['lr'] for group in wmopt.param_groups],
                    wm_loss=float(torch.stack(losses).mean()) if losses else None,collect_seconds=collected-tick,
                    policy_updated=updated,policy_updates=policy_updates,wm_gate_allowed=gate['allowed'],wm_gate_reasons=gate['reasons'],
                    wm_seconds=modeled-collected,wm_audit_seconds=audited-modeled,policy_seconds=finished-audited,iteration_seconds=finished-tick,
                    elapsed=time.time()-begin,**metrics,**native_metrics)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                status(status='running',phase='validation' if iteration-resumed_iteration<=o.validation_iterations else 'alternating',actual_num_envs=n,**row)
                if iteration-resumed_iteration==o.validation_iterations:
                    mean=sum(steady_times)/max(1,len(steady_times))
                    passed=bool(steady_times) and mean<=o.max_steady_iteration_seconds
                    (out/'throughput-validation.json').write_text(json.dumps(dict(passed=passed,
                        steady_iteration_seconds=steady_times,mean_seconds=mean,
                        limit_seconds=o.max_steady_iteration_seconds,policy_batch=o.policy_groups*9,num_envs=n,
                        includes_policy_update=run_policy_updates>0),indent=2))
                    if not passed:
                        save(iteration)
                        raise RuntimeError(f'Throughput gate failed: {mean:.3f} s/iteration; long training not started')
                    print(f'THROUGHPUT_VALIDATED {mean:.3f} s/iteration',flush=True)
                if iteration%o.save_every==0:save(iteration)
                if o.benchmark_iterations and iteration-resumed_iteration>=o.benchmark_iterations:break
        # Finish any older evaluation, then execute a whole episode with the
        # final frozen policy. Do not label an older monitor result as final.
        if not stopped:collect(40+(-native_steps)%40)
        audit(iteration)
        if frozen_policy_state is not None:
            if not all(torch.equal(v,policy.state_dict()[k]) for k,v in frozen_policy_state.items()):
                raise RuntimeError('Policy changed during dynamics-only calibration')
            (out/'policy-frozen-verification.json').write_text(json.dumps(dict(unchanged=True,policy_updates=run_policy_updates)))
        if o.export_native_replay:
            # Fresh, solver-labelled evidence only; no native solver snapshots.
            keys=('state','next_state','action','height','previous_force_w','force_w','previous_active',
                  'active','previous_known','known','valid','motion_replay','age','ended','event',
                  'spatial_points_w','spatial_forces_w','spatial_mask')
            def export_buffer(buffer):
                start=max(0,buffer.steps-buffer.capacity)
                ids=torch.arange(start,buffer.steps,device=device)%buffer.capacity
                return {k:buffer.data[k][ids].cpu() for k in keys if k in buffer.data}
            torch.save(dict(schema='native_world_contact_replay_v1',train=export_buffer(replay),
                heldout=export_buffer(heldout),robot_asset=l['provenance']['robot_asset'],
                semantics='actual allocated Newton constraints; shared upward primary faces; forces on robot',
                timing='last native solve contact rows paired with post-integration articulation state',
                coordinate_frame='fixed reference world; clone environment translations removed',
                options=asdict(o)),out/'native-replay.pt')
        save(iteration);status(status='stopped' if stopped else 'completed',phase='saved',iteration=iteration,actual_num_envs=n,
            policy_updates=policy_updates,run_policy_updates=run_policy_updates,
            wm_updates=wm_updates,run_wm_updates=run_wm_updates,
            policy_learning_rates=[group['lr'] for group in opt.param_groups],
            wm_learning_rates=[group['lr'] for group in wmopt.param_groups],
            wm_gate_allowed=gate['allowed'],wm_gate_reasons=gate['reasons'],**native_metrics)

if __name__=='__main__':main()
