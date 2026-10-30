"""First complete-trajectory attempt per task; frozen student, no optimizer steps."""
from dataclasses import dataclass, replace
import copy
import json
from pathlib import Path


@dataclass(frozen=True)
class Options:
    manifest: str
    teacher_spec: str
    checkpoint: Path
    output_dir: Path
    max_steps: int = 1500


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args,sync_launcher_headless_config
    launcher=parse_isaaclab_launcher_args(__doc__)
    import tyro
    options=tyro.cli(Options)
    import numpy as np
    import torch
    from holosoma.php_contact.distill import Options as TrainOptions,build_config
    from holosoma.php_contact.dataset import load_prepared
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_contacts import measured_contacts

    prepared=load_prepared(options.manifest)
    n=len(prepared['motion_files'])
    config=build_config(TrainOptions(manifest=options.manifest,teacher_spec=options.teacher_spec,
        num_envs=n,segment_mode='continuous',pure_distill_iters=200,regularizer_mse_coef=.1))
    config=config.get_eval_config(EvaluationConfig(num_envs=n))
    term=config.command.setup_terms['motion_command']
    motion=replace(term.params['motion_config'],start_at_timestep_zero_prob=1.,fixed_motion_assignment_in_eval=True,
        noise_to_initial_pose=replace(term.params['motion_config'].noise_to_initial_pose,overall_noise_scale=0.))
    config=replace(config,command=replace(config.command,setup_terms={'motion_command':replace(term,
        params={**term.params,'motion_config':motion,'segment_start_probability':1.})}),
        randomization=RandomizationManagerCfg(),observation=replace(config.observation,groups={
            name:replace(group,enable_noise=False) for name,group in config.observation.groups.items()}))
    config=sync_launcher_headless_config(config,launcher)
    output=options.output_dir.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'raw_boundaries').mkdir()
    outcomes={};boundaries=[]
    saved=torch.load(options.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    class EvaluationContext(TrainingContext):
        def __exit__(self,kind,error,tb):
            if error is not None:
                import traceback
                detail=''.join(traceback.format_exception(kind,error,tb))
                (output/'failure.json').write_text(json.dumps(dict(error=str(error),traceback=detail),indent=2)+'\n')
                print(detail,flush=True)
            return super().__exit__(kind,error,tb)
    with EvaluationContext(config,launcher_args=launcher):
        env=get_class(config.env_class)(get_tyro_env_config(config),device=launcher.device)
        env.set_is_evaluating()
        algo=get_class(config.algo._target_)(env=env,config=config.algo.config,
            log_dir=str(output/'inference'),device=launcher.device)
        algo.setup();algo.load_for_inference(str(options.checkpoint));algo._eval_mode()
        boundary=copy.copy(algo);boundary.current_learning_iteration=int(saved['iter'])
        boundary._update_php_student_tracking_boundary()
        cmd=env.command_manager.get_state('motion_command')
        native_metrics=env._update_log_dict
        def capture():
            eligible=cmd._trajectory_open & cmd._episode_from_head & (env.episode_length_buf>0)
            due=eligible & (cmd.time_steps>=cmd.segment_end-1) & (cmd._last_boundary!=cmd.segment_end)
            native_metrics()
            term=env.termination_manager
            ended=term.terminated|term.time_outs
            event_ids=[i for i in torch.where(due)[0].tolist() if i not in outcomes]
            if event_ids:
                actual=measured_contacts(env)
                snapshot=env._wbt_measured_contact_reader.export_snapshot(actual['snapshot'])
                count,capacity=int(snapshot['count']),len(snapshot['dist'])
                raw={k:v[:count] if getattr(v,'ndim',0)>0 and len(v)==capacity else v for k,v in snapshot.items()}
                path=output/'raw_boundaries'/f'{len(boundaries):06d}.npz'
                np.savez_compressed(path,**raw,
                    post_physics_root_state=env.simulator.robot_root_states[:].cpu().numpy(),
                    post_physics_joint_pos=env.simulator.dof_pos.cpu().numpy(),
                    reference_frames=cmd.time_steps.cpu().numpy(),episode_lengths=env.episode_length_buf.cpu().numpy())
                for i in event_ids:
                    boundaries.append(dict(env=i,motion=int(cmd.motion_ids[i]),segment=int(cmd.selected_segment[i]),
                        frame=int(cmd.time_steps[i]),goal_frame=int(cmd.segment_end[i]),
                        known=bool(actual['known'][i].all()),matched=bool((actual['active'][i]==cmd.goal_active[i]).all()),
                        root_error_m=float(cmd._last_boundary_root_error[i]),
                        limb_error_m=float(cmd._last_boundary_limb_error[i]),
                        episode_length=int(env.episode_length_buf[i]),
                        failed=bool(term.terminated[i]),raw_solver_snapshot=str(path)))
            for i in torch.where(ended)[0].tolist():
                if i in outcomes:continue
                mid=int(cmd.motion_ids[i]);first=int(cmd.motion.motion_start_idx[mid]);last=int(cmd.motion.motion_end_idx[mid])-1
                outcomes[i]=dict(env=i,motion=mid,motion_file=cmd._checkpoint_motion_keys()[mid],
                    frame=int(cmd.time_steps[i])-first,frame_count=last-first+1,
                    run_completed=bool(term.term_dones['motion_ends'][i] & ~term.terminated[i]),
                    failure_segment=int(cmd.selected_segment[i]-cmd.segment_offsets[mid]),
                    boundary_observations=int(cmd._trajectory_observed[i]),
                    total_segments=int(cmd.segment_counts[mid]),all_boundary_contacts=bool(cmd._trajectory_all_contacts[i]),
                    reasons=[name for name,mask in term.term_dones.items() if bool(mask[i])])
        env._update_log_dict=capture
        obs=env.reset_all(step_after_reset=False)
        assert cmd._episode_from_head.all() and torch.equal(cmd.motion_ids,torch.arange(n,device=env.device))
        with torch.inference_mode():
            for step in range(options.max_steps):
                value=torch.cat([obs[k] for k in algo.actor_obs_keys],dim=1)
                action=algo.actor.act_inference({'actor_obs':algo._normalize_actor_obs(value,update=False)})
                if not torch.isfinite(action).all():raise RuntimeError('Nonfinite continuous student actions')
                obs,_,_,_=env.step({'actions':action})
                if len(outcomes)==n:break
        if len(outcomes)!=n:raise RuntimeError(f'Evaluation incomplete: {len(outcomes)}/{n}')
        assert not algo.actor_optimizer.state and not algo.critic_optimizer.state
        rows=[outcomes[i] for i in range(n)]
        report=dict(schema='php_continuous_first_attempt_evaluation_v1',checkpoint=str(options.checkpoint),
            checkpoint_iteration=int(saved['iter']),tasks=n,control_steps=step+1,optimizer_steps=0,
            from_complete_trajectory_head=True,policy='deterministic mean; no observation/reset noise',
            run_completed=sum(r['run_completed'] for r in rows),
            all_boundary_contacts_completed=sum(r['run_completed'] and r['all_boundary_contacts']
                and r['boundary_observations']==r['total_segments'] for r in rows),
            head_time_progress_mean=sum((r['frame']+1)/(r['frame_count']-1) for r in rows)/n,
            boundary_observations=len(boundaries),boundary_contact_success_rate=sum(
                r['known'] and r['matched'] and not r['failed'] for r in boundaries)/max(len(boundaries),1),
            outcomes=rows,boundaries=boundaries,live_contact_backend=env._wbt_measured_contact_reader.backend,
            boundary_execution='same physical state; no intermediate reset or reference pose writes',
            raw_boundary_evidence=str(output/'raw_boundaries'))
        (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'binding.json').write_text(json.dumps(env._wbt_measured_contact_reader.binding,indent=2)+'\n')
        print('CONTINUOUS_EVALUATION',json.dumps({k:report[k] for k in (
            'tasks','run_completed','all_boundary_contacts_completed','boundary_contact_success_rate','optimizer_steps')}),flush=True)
        algo.writer.close()


if __name__=='__main__':main()
