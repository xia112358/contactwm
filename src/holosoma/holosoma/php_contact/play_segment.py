"""Inference-only real-time playback of an action segment or complete trajectory."""
from dataclasses import dataclass, replace
import copy
import json
from pathlib import Path
import signal
import time

from holosoma.php_contact.segments import SegmentCommand


class PlaybackSegmentCommand(SegmentCommand):
    def sample_reset_phase(self, env_ids):
        import torch
        if self.motion.num_motions != 1:
            raise ValueError('Pinned playback requires exactly one prepared motion')
        local = int(self.cfg.params['playback_segment'])
        if not 0 <= local < int(self.segment_counts[0]):
            raise ValueError('Playback segment is outside the prepared action clock')
        selected = torch.full_like(env_ids, local)
        self._set_segments(env_ids, selected)
        first = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        length = self.motion.motion_end_idx[self.motion_ids[env_ids]]-first-1
        return (self.segment_start[env_ids]-first+.5)/length.clamp_min(1)


@dataclass(frozen=True)
class Options:
    checkpoint: Path
    manifest: Path
    output_dir: Path
    motion_id: int = 24
    segment_id: int = 11
    whole_trajectory: bool = False
    out_of_distribution: bool = False
    tracking_termination: bool = True
    steps: int = 0


def refresh_reset_reference(cmd):
    """Refresh native relative body targets at the unchanged reset clock."""
    clock = cmd.time_steps.clone()
    cmd._update_relative_body_targets()
    if not cmd.time_steps.equal(clock):
        raise RuntimeError('Reset reference refresh changed the pinned clock')


def playback_bounds(cmd, whole_trajectory):
    if whole_trajectory:
        mid = int(cmd.motion_ids[0])
        return int(cmd.motion.motion_start_idx[mid]), int(cmd.motion.motion_end_idx[mid])-1
    return int(cmd.segment_start[0]), int(cmd.segment_end[0])


def configure_tracking_termination(config, enabled):
    if enabled:
        return config
    return replace(config, termination=replace(config.termination,
        terms={name:term for name,term in config.termination.terms.items() if name != 'bad_tracking'}))


def ood_inference_config(saved, manifest, source):
    """Keep checkpoint policy/control semantics; replace only the evaluation task."""
    from holosoma.config_types.algo import DistillPPOConfig
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.utils.eval_utils import _experiment_config_from_serialized
    if source.get('reference_origin') != 'out_of_distribution_native_reference' or source['split'] != 'evaluation':
        raise ValueError('OOD playback requires an explicitly evaluation-only native reference')
    data = copy.deepcopy(saved['experiment_config'])
    if data['algo']['_target_'] != 'holosoma.php_contact.distill.SegmentDistillPPO':
        raise ValueError('Expected a contact-segment student checkpoint')
    data['algo']['config'] = DistillPPOConfig(**data['algo']['config'])
    cfg = _experiment_config_from_serialized(data).get_eval_config(EvaluationConfig(num_envs=1))
    original = cfg.command.setup_terms['motion_command']
    motion = replace(original.params['motion_config'], motion_manifest=str(manifest), motion_file='', motion_dir='',
        reset_sampler='uniform', use_start_probe_envs=False, use_group_probe_envs=False,
        use_completion_learning_sampler=False)
    params = {key:value for key,value in original.params.items() if key != 'teacher_training_manifest'}
    term = replace(original, params={**params, 'motion_config':motion, 'segment_mode':'continuous'})
    spacing = source['terrain_spacing']
    return replace(cfg, command=replace(cfg.command, setup_terms={'motion_command':term}),
        observation=replace(cfg.observation, groups={name:group for name,group in cfg.observation.groups.items()
                                                   if not name.startswith('expert_')}),
        algo=replace(cfg.algo, config=replace(cfg.algo.config,
            distill=replace(cfg.algo.config.distill, enable_kl=False, teachers=(), teacher_spec_file=None,
                            teacher_checkpoint_path=None, teacher_obs_keys=None))),
        terrain=replace(cfg.terrain, terrain_term=replace(cfg.terrain.terrain_term,
            motion_matched_manifest=str(manifest), terrain_length=spacing[0], terrain_width=spacing[1])))


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args(__doc__)
    import tyro
    options = tyro.cli(Options)
    import torch
    from holosoma.php_contact.dataset import load_prepared, digest
    from holosoma.php_contact.distill import Options as DistillOptions, build_config
    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_types.eval_callback import EvaluationConfig
    from holosoma.config_types.randomization import RandomizationManagerCfg
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_contacts import measured_contacts
    from somaforge_core.robot_assets import validate_g1_asset_metadata

    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = load_prepared(options.manifest)
    selected = [r for r in source['motion_files'] if r['motion_id'] == options.motion_id]
    if len(selected) != 1:
        raise ValueError('Motion ID must identify one prepared motion')
    row = selected[0]
    saved = torch.load(options.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    validate_g1_asset_metadata(saved['robot_asset'], context=str(options.checkpoint))
    saved_command = saved['experiment_config']['command']['setup_terms']['motion_command']['params']
    if options.out_of_distribution and not options.whole_trajectory:
        raise ValueError('OOD playback currently requires whole-trajectory execution')
    if not options.out_of_distribution and digest(saved_command['motion_config']['motion_manifest']) != digest(options.manifest):
        raise ValueError('Playback data differs from the student checkpoint task manifest')
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    playback_manifest = output/'play_manifest.json'
    playback_manifest.write_text(json.dumps(dict(source, motion_files=[row]), indent=2)+'\n')
    if options.out_of_distribution:
        config = ood_inference_config(saved, playback_manifest, source)
    else:
        spec = output/'teachers.json'
        spec.write_text(json.dumps(dict(teachers=[dict(checkpoint=source['teacher']['path'],
                                                      manifest=str(playback_manifest))]), indent=2)+'\n')
        config = build_config(DistillOptions(manifest=str(playback_manifest), teacher_spec=str(spec),
            num_envs=1, output=str(output/'inference'),
            segment_mode='continuous' if options.whole_trajectory else 'independent')).get_eval_config(EvaluationConfig(num_envs=1))
    original = config.command.setup_terms['motion_command']
    prefix = original.func if options.whole_trajectory else 'holosoma.php_contact.play_segment:PlaybackSegmentCommand'
    motion = replace(original.params['motion_config'], start_at_timestep_zero_prob=1. if options.whole_trajectory else 0.,
        freeze_at_timestep_zero_prob=0., fixed_motion_assignment_in_eval=options.whole_trajectory,
        noise_to_initial_pose=replace(original.params['motion_config'].noise_to_initial_pose,
                                     overall_noise_scale=0.))
    term = replace(original, func=prefix, params={**original.params, 'motion_config':motion,
                                                'playback_segment':options.segment_id})
    config = replace(config, command=replace(config.command,
        setup_terms={'motion_command':term},
        reset_terms={'motion_command':CommandTermCfg(func=prefix)},
        step_terms={'motion_command':CommandTermCfg(func=prefix)}),
        randomization=RandomizationManagerCfg(),
        observation=replace(config.observation, groups={name:replace(group, enable_noise=False)
                                                      for name,group in config.observation.groups.items()}))
    config = configure_tracking_termination(config, options.tracking_termination)
    # A single segment uses the pinned phase sampler. A whole trajectory uses
    # fixed-task evaluation from frame zero and keeps state across all actions.
    config = sync_launcher_headless_config(config, launcher)
    status = dict(state='initializing', checkpoint=str(options.checkpoint.resolve()),
        checkpoint_iteration=int(saved['iter']), policy='student deterministic mean',
        motion_id=options.motion_id, segment_id=None if options.whole_trajectory else options.segment_id,
        whole_trajectory=options.whole_trajectory, optimizer_steps=0,
        out_of_distribution=options.out_of_distribution,
        tracking_termination=options.tracking_termination,
        completion_interpretation='reference endpoint reached; physical/contact achievement reported separately',
        contact_goal_source='own native reference contacts' if options.out_of_distribution else 'matched teacher extraction',
        reference_pose_written_during_execution=False, episodes=0, steps=0)
    def persist():
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    persist()
    state = dict(stop=False, pause=False, restart=False, follow=options.whole_trajectory)
    def stop(*_): state['stop'] = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    with TrainingContext(config, launcher_args=launcher) as context:
        try:
            env = get_class(config.env_class)(get_tyro_env_config(config), device=launcher.device)
            if options.whole_trajectory:
                env.set_is_evaluating()
            cmd = env.command_manager.get_state('motion_command')
            obs = env.reset_all(step_after_reset=False)
            algo = get_class(config.algo._target_)(env=env, config=config.algo.config,
                log_dir=str(output/'inference'), device=launcher.device)
            algo.setup()
            algo.load_for_inference(str(options.checkpoint))
            algo._eval_mode()
            # PPO construction performs a bootstrap physics step. Discard that
            # state and its observations before the first visible execution.
            obs = env.reset_all(step_after_reset=False)
            # Match the checkpoint's training-time tracking boundary without
            # importing optimizer, progress, statistics or simulator state.
            boundary = copy.copy(algo)
            boundary.current_learning_iteration = int(saved['iter'])
            boundary._update_php_student_tracking_boundary()
            tracking = env.termination_manager._term_instances.get('bad_tracking')
            initial_before = bool(tracking(env)[0]) if tracking is not None else None
            refresh_reset_reference(cmd)
            initial_after = bool(tracking(env)[0]) if tracking is not None else None
            status['reset_tracking_invalid_before_reference_refresh'] = initial_before
            status['reset_tracking_invalid_after_reference_refresh'] = initial_after
            env._compute_observations()
            env._clip_observations()
            obs = env.obs_buf_dict
            first, last = playback_bounds(cmd,options.whole_trajectory)
            if int(cmd.time_steps[0]) != first:
                raise RuntimeError('Playback did not reset to the selected start frame')
            status.update(start_frame=first, end_frame=last, duration_s=(last-first)*env.dt,
                          target_contacts=cmd.goal_active[0].cpu().tolist(),
                          total_segments=int(cmd.segment_counts[0]))
            label = None
            if not config.training.headless:
                import omni.ui as ui
                import omni.kit.viewport.utility as viewport
                import omni.usd
                from omni.kit.viewport.window import get_viewport_window_instances
                env.simulator.sim.initialize_visualizers()
                windows = list(get_viewport_window_instances())
                if not windows:
                    raise RuntimeError('Kit did not create a viewport')
                viewport_window = windows[0]
                selection_guard = viewport.disable_selection(viewport_window)
                omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
                window = ui.Window('Student trajectory playback' if options.whole_trajectory else
                                   'Student segment playback', width=440, height=260)
                with window.frame:
                    with ui.VStack(spacing=6):
                        label = ui.Label('', word_wrap=True)
                        with ui.HStack():
                            ui.Button('Pause / Resume', clicked_fn=lambda:state.update(pause=not state['pause']))
                            ui.Button('Restart trajectory' if options.whole_trajectory else 'Restart segment',
                                      clicked_fn=lambda:state.update(restart=True))
                        if options.whole_trajectory:
                            ui.Button('Toggle follow camera',clicked_fn=lambda:state.update(follow=not state['follow']))
                        ui.Button('Close playback', clicked_fn=stop)
                status['viewport_count'] = len(windows)
            target = (cmd.motion.body_pos_w[first,0]+cmd.motion.body_pos_w[last,0])*.5
            target = (target+cmd._reference_env_origins[0]).cpu().tolist()
            target[2] += .15
            if state['follow']:
                target=cmd.robot_root_pos_w[0].detach().cpu().tolist()
                target[2]+=.15
            env.simulator.sim.set_camera_view((target[0]+2.8,target[1]-3.0,target[2]+1.6),target)
            outcome = {}
            if options.out_of_distribution:
                (output/'raw_first_attempt').mkdir()
                (output/'binding.json').write_text(json.dumps(env._wbt_measured_contact_reader.binding,indent=2)+'\n')
            native_metrics = env._update_log_dict
            def capture_terminal():
                boundary_due = (options.whole_trajectory and bool(cmd._trajectory_open[0])
                    and int(env.episode_length_buf[0])>0 and int(cmd.time_steps[0])>=int(cmd.segment_end[0])-1
                    and int(cmd._last_boundary[0])!=int(cmd.segment_end[0]))
                native_metrics()
                terminations = env.termination_manager
                ended = bool((terminations.terminated|terminations.time_outs)[0])
                if options.out_of_distribution and status['episodes']==0 and (boundary_due or ended):
                    import numpy as np
                    actual = measured_contacts(env)
                    raw = env._wbt_measured_contact_reader.export_snapshot(actual['snapshot'])
                    raw_path = output/'raw_first_attempt'/f"frame_{int(cmd.time_steps[0]):05d}.npz"
                    np.savez_compressed(raw_path, **raw,
                        post_physics_root_state=env.simulator.robot_root_states[:].cpu().numpy(),
                        post_physics_joint_pos=env.simulator.dof_pos.cpu().numpy(),
                        goal_active=cmd.goal_active.cpu().numpy(),goal_position_w=cmd.goal_position_w.cpu().numpy())
                    row = dict(frame=int(cmd.time_steps[0]), goal_frame=int(cmd.segment_end[0]),
                        segment=int(cmd.selected_segment[0]), boundary=boundary_due, failed=bool(terminations.terminated[0]),
                        known=bool(actual['known'][0].all()), matched=bool((actual['active'][0]==cmd.goal_active[0]).all()),
                        raw_solver_snapshot=str(raw_path),
                        contact_sampling='last native solve; root and joints are post integration')
                    with (output/'first_attempt_contacts.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                if ended:
                    actual = measured_contacts(env)
                    outcome.update(frame=int(cmd.time_steps[0]),
                        failed=bool(terminations.terminated[0]),
                        contact_known=bool(actual['known'][0].all()),
                        contact_pattern_matched=bool((actual['active'][0]==cmd.goal_active[0]).all()),
                        actual_contacts=actual['active'][0].cpu().tolist(),
                        whole_trajectory_completed=bool(options.whole_trajectory
                            and terminations.term_dones['motion_ends'][0] and not terminations.terminated[0]),
                        elapsed_reference_s=(int(cmd.time_steps[0])+1-first)*env.dt,
                        reference_time_progress=min(1.,(int(cmd.time_steps[0])+1-first)/max(1,last-first)),
                        boundaries_observed=int(cmd._trajectory_observed[0]) if options.whole_trajectory else 0,
                        boundary_statistics=cmd.trajectory_statistics.metrics() if options.whole_trajectory else {},
                        tracking_errors={name:float(value) for name,value in tracking.metrics.items()} if tracking is not None else {},
                        reasons=[name for name,value in terminations.term_dones.items() if bool(value[0])])
            env._update_log_dict = capture_terminal
            status['state'] = 'running';persist()
            print('SEGMENT_PLAYBACK_READY',json.dumps(status),flush=True)
            next_status = time.monotonic()+1.
            with torch.inference_mode():
                while context.simulation_app.is_running() and not state['stop']:
                    begin = time.monotonic()
                    if state['pause']:
                        context.simulation_app.update();time.sleep(.02);continue
                    if state['restart']:
                        obs = env.reset_all(step_after_reset=False)
                        refresh_reset_reference(cmd)
                        env._compute_observations();env._clip_observations();obs=env.obs_buf_dict
                        state['restart'] = False
                    frame = int(cmd.time_steps[0])
                    value = torch.cat([obs[key] for key in algo.actor_obs_keys],dim=1)
                    action = algo.actor.act_inference({'actor_obs':algo._normalize_actor_obs(value,update=False)})
                    if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite student action')
                    obs, _, dones, _ = env.step({'actions':action})
                    status['steps'] += 1
                    if bool(dones[0]):
                        if not outcome: raise RuntimeError('Missing pre-reset terminal observation')
                        status['episodes'] += 1
                        status['last_episode'] = dict(outcome)
                        with (output/'episodes.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(episode=status['episodes'],**outcome))+'\n')
                        outcome.clear()
                        # Native env.step advances its command clock after an
                        # automatic reset. Start the next visible execution at
                        # the exact segment head with freshly reset histories.
                        obs = env.reset_all(step_after_reset=False)
                        refresh_reset_reference(cmd)
                        env._compute_observations();env._clip_observations();obs=env.obs_buf_dict
                        if int(cmd.time_steps[0]) != first:
                            raise RuntimeError('Automatic reset left the selected start frame')
                    if label is not None:
                        mode = (f"Motion {options.motion_id} | whole trajectory | action {int(cmd.contact_data['segment_id'][cmd.time_steps[0]])+1}/{int(cmd.segment_counts[0])}"
                                if options.whole_trajectory else f"Motion {options.motion_id}, segment {options.segment_id}")
                        label.text = (f"Student checkpoint: {saved['iter']}\n{mode}\n"
                            f"Tracking termination: {'ON' if options.tracking_termination else 'OFF'}\n"
                            f"Frames {first}..{last} | {(last-first)*env.dt:.2f} s\n"
                            f"Loop {status['episodes']+1} | time {(frame-first)*env.dt:.2f} s\n"
                            f"Target contacts: {', '.join(name for name,active in zip(source['parts'],cmd.goal_active[0].tolist()) if active)}\n"
                            f"Last: {status.get('last_episode',{}).get('reasons','starting')}")
                    if state['follow'] and status['steps']%5==0:
                        target=cmd.robot_root_pos_w[0].detach().cpu().tolist();target[2]+=.15
                        env.simulator.sim.set_camera_view((target[0]+2.8,target[1]-3.0,target[2]+1.6),target)
                    if status['steps']==50 and label is not None:
                        viewport.capture_viewport_to_file(viewport_window.viewport_api,str(output/'viewport.png'))
                    if time.monotonic()>=next_status:
                        status['reference_frame']=int(cmd.time_steps[0])
                        status['current_segment']=int(cmd.contact_data['segment_id'][cmd.time_steps[0]])
                        persist();next_status=time.monotonic()+1.
                    if options.steps and status['steps']>=options.steps: break
                    time.sleep(max(0.,env.dt-(time.monotonic()-begin)))
            status['state']='stopped';persist()
            algo.writer.close()
        except BaseException as error:
            status.update(state='failed',error=str(error));persist()
            raise


if __name__ == '__main__':
    main()
