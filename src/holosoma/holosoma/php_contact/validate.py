"""Bounded native inference diagnostic, never starts a PHP training run."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class Options:
    manifest: str
    output_dir: Path
    num_envs: int = 8
    steps: int = 48
    teacher_spec: str | None = None


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args(__doc__)
    import tyro
    options = tyro.cli(Options)
    if min(options.num_envs, options.steps) < 1:
        raise ValueError('Diagnostic sizes must be positive')
    import numpy as np
    import torch
    from holosoma.php_contact.train import Options as TrainOptions, build_config
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_contacts import measured_contacts
    from somaforge_core.contact_face_selection import FACE_POLICY
    from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART
    from somaforge_core.newton_contacts import PARTS, SCHEMA
    from somaforge_core.robot_assets import canonical_g1_asset_metadata

    if options.teacher_spec:
        from holosoma.php_contact.distill import Options as DistillOptions, build_config as distill_config
        config = distill_config(DistillOptions(manifest=options.manifest,
            teacher_spec=options.teacher_spec, num_envs=options.num_envs))
    else:
        config = build_config(TrainOptions(manifest=options.manifest, num_envs=options.num_envs, probes=False))
    source = config.command.setup_terms['motion_command']
    motion = replace(source.params['motion_config'], start_at_timestep_zero_prob=0. if options.teacher_spec else 1.,
        fixed_motion_assignment_in_eval=True, hold_at_motion_end_in_eval=True,
        noise_to_initial_pose=replace(source.params['motion_config'].noise_to_initial_pose, overall_noise_scale=0.))
    config = replace(config, command=replace(config.command, setup_terms={
        'motion_command': replace(source, params={**source.params, 'motion_config': motion})}))
    config = sync_launcher_headless_config(config, launcher)
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'raw_contacts').mkdir()
    class DiagnosticContext(TrainingContext):
        def __exit__(self, exc_type, exc_value, exc_tb):
            if exc_value is not None:
                import traceback
                detail = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
                (output/'failure.json').write_text(json.dumps({'status':'failed', 'traceback':detail}, indent=2)+'\n')
                print(detail, flush=True)
            return super().__exit__(exc_type, exc_value, exc_tb)

    with DiagnosticContext(config, launcher_args=launcher):
        env = get_class(config.env_class)(get_tyro_env_config(config), device=launcher.device)
        if not options.teacher_spec:
            env.set_is_evaluating()
        obs = env.reset_all(step_after_reset=False)
        algo = get_class(config.algo._target_)(env=env, config=config.algo.config,
            log_dir=str(output/'inference'), device=launcher.device)
        algo.setup()
        algo.attach_checkpoint_metadata(config, None)
        algo._eval_mode()
        cmd = env.command_manager.get_state('motion_command')
        rows = []
        native_reward = env._compute_reward

        def capture():
            native_reward()
            actual = measured_contacts(env)
            snapshot = env._wbt_measured_contact_reader.export_snapshot(actual['snapshot'])
            count, capacity = int(snapshot['count']), len(snapshot['dist'])
            raw = {key: value[:count] if getattr(value, 'ndim', 0) > 0 and len(value) == capacity else value
                   for key, value in snapshot.items()}
            np.savez_compressed(output/'raw_contacts'/f'frame_{len(rows):06d}.npz', **raw,
                post_physics_root_state=env.simulator.robot_root_states[:].cpu().numpy(),
                post_physics_joint_pos=env.simulator.dof_pos.cpu().numpy(),
                observed_active=actual['active'].cpu().numpy(), observed_known=actual['known'].cpu().numpy(),
                observed_force_w=actual['force_w'].cpu().numpy(),
                motion_ids=cmd.motion_ids.cpu().numpy(), reference_frame=cmd.time_steps.cpu().numpy(),
                command_target_frame=(cmd.segment_end if options.teacher_spec else
                    cmd.contact_data['target_frame'][cmd.time_steps]).cpu().numpy())
            rows.append(dict(active=int(actual['active'].sum()), known=int(actual['known'].sum()),
                             terminated=int(env.termination_manager.terminated.sum())))

        env._compute_reward = capture
        distill_batch = None
        with torch.inference_mode():
            for _ in range(options.steps):
                value = torch.cat([obs[key] for key in algo.actor_obs_keys], dim=1)
                normalized = algo._normalize_actor_obs(value, update=False)
                if options.teacher_spec:
                    # Preserve a real student-state/teacher-reference pair.
                    teacher_value = torch.cat([obs[key] for key in algo.teacher_obs_keys], dim=1)
                    valid = algo._compute_dagger_valid_mask().float()
                    distill_batch = (normalized.detach(), teacher_value.detach(), valid.detach())
                action = algo.actor.act_inference({'actor_obs': normalized})
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite PHP contact actor output')
                obs, reward, _, _ = env.step({'actions': action})
                if not torch.isfinite(reward).all() or any(not torch.isfinite(value).all() for value in obs.values()):
                    raise RuntimeError('Nonfinite PHP contact observations/rewards')
        distill_report = None
        if distill_batch is not None:
            actor_value, teacher_value, valid = [value.clone() for value in distill_batch]
            loss, mse, ld, lp, valid_rate = algo._compute_teacher_distill_loss(dict(
                actor_obs=actor_value, teacher_actor_obs=teacher_value, dagger_valid_mask=valid))
            loss.backward()
            gradients = [p.grad for p in algo.actor.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Invalid native DAgger student gradients')
            if any(p.grad is not None or p.requires_grad for p in algo.teacher_actor.parameters()):
                raise RuntimeError('Teacher must remain frozen during student distillation')
            distill_report = dict(loss=float(loss.detach()), mse=float(mse), valid_rate=float(valid_rate),
                lambda_d=float(ld), lambda_ppo=float(lp), finite_student_gradients=True,
                frozen_teacher=True, optimizer_steps=0)
            algo.actor_optimizer.zero_grad(set_to_none=True)
        (output/'binding.json').write_text(json.dumps(env._wbt_measured_contact_reader.binding, indent=2)+'\n')
        checkpoint_path = output/'inference'/'untrained_model.pt'
        algo.save(str(checkpoint_path))
        segment_report = None
        segment_csv = None
        statistics_checkpoint_verified = False
        if options.teacher_spec:
            from holosoma.php_contact.statistics import SegmentStatistics
            segment_report = cmd.segment_statistics.metrics(cumulative=True)
            segment_csv = cmd.segment_statistics.write_csv(output, 0)
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
            restored = SegmentStatistics(cmd.segment_statistics.keys, 'cpu')
            restored.load_state_dict(saved['env_state']['motion_sampler']['segment_head_statistics'])
            torch.testing.assert_close(restored.counts, cmd.segment_statistics.counts.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(restored.progress, cmd.segment_statistics.progress.cpu(), rtol=0, atol=0)
            statistics_checkpoint_verified = True
        algo.load_for_inference(str(checkpoint_path))
        # Kit may exit on shutdown, so reports must be persisted inside the context.
        report = dict(schema='somaforge_php_contact_runtime_check_v1', status='bounded_inference_passed',
            env_class=config.env_class, algorithm=config.algo._target_, manifest=options.manifest,
            num_envs=options.num_envs, control_steps=len(rows), training_updates=0,
            teacher_spec=options.teacher_spec, native_dagger_check=distill_report,
            segment_start_probability=cmd.start_probability if options.teacher_spec else None,
            segment_head_statistics=segment_report, segment_statistics_csv=str(segment_csv) if segment_csv else None,
            segment_statistics_checkpoint_verified=statistics_checkpoint_verified,
            complex_sampler_flags=(dict(reset_sampler=cmd._reset_sampler,
                failure_window=cmd._uses_failure_window_sampler,
                completion_learning=cmd._use_completion_learning_sampler,
                start_probe=cmd._use_start_probe_envs, group_probe=cmd._use_group_probe_envs,
                routed_schedule_active=algo.motion_schedule is not None) if options.teacher_spec else None),
            actor_inputs={key: int(obs[key].shape[1]) for key in algo.actor_obs_keys},
            critic_inputs={key: int(obs[key].shape[1]) for key in algo.critic_obs_keys},
            robot_asset=canonical_g1_asset_metadata(), parts=PARTS, face_selection=FACE_POLICY,
            body_names_by_part=CONTACT_BODY_NAMES_BY_PART,
            live_contact_semantics=SCHEMA, live_sampling='last_native_solve_before_policy_observation',
            activation='CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc',
            strict_pose_contact_same_state=False, reference_sampling='fixed_q_forward_no_integration',
            physical_trajectory_acceptance=False, dynamic_support_load_verified=False,
            checkpoint=str(checkpoint_path), checkpoint_inference_reload_verified=True,
            binding=str(output/'binding.json'), raw_contacts=str(output/'raw_contacts'), frames=rows)
        report['live_contact_backend'] = env._wbt_measured_contact_reader.backend
        (output/'runtime_report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key: report[key] for key in ('status', 'control_steps', 'training_updates', 'actor_inputs')}), flush=True)


if __name__ == '__main__':
    main()
