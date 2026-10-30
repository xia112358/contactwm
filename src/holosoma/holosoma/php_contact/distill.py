"""Segmented PHP DAgger/PPO with native contact rewards; no automatic teacher choice."""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import json
import math
from pathlib import Path
from typing import Literal

from holosoma.agents.ppo.distill_ppo import DistillPPO
from holosoma.agents.ppo.routed_distill_ppo import RoutedDistillPPO


class SegmentDistillPPO(RoutedDistillPPO):
    def load(self, checkpoint):
        result = super().load(checkpoint)
        if checkpoint:
            # Native learn() treats num_learning_iterations as remaining work;
            # display the global endpoint when this phase resumes an old run.
            self.logging_helper.num_learning_iterations = self.current_learning_iteration+self.config.num_learning_iterations
        return result

    def _init_config(self):
        # Native PPO infers observation dimensions before its first reset.
        # Place every robot on its task terrain before physical height queries.
        self.env.reset_all(step_after_reset=False)
        super()._init_config()

    def _compute_dagger_valid_mask(self):
        from holosoma.php_contact.terms import command
        from holosoma.php_contact.teacher_reference import reference_view
        cmd = command(self.env)
        if cmd._teacher_view is None:
            cmd._teacher_view = reference_view(self.env, cmd)
        # A state close to the fitted student reference can still be outside
        # the original expert's task envelope. Require both validity checks.
        teacher_context = copy.copy(self)
        teacher_context.env = cmd._teacher_view
        return (DistillPPO._compute_dagger_valid_mask(self).bool()
                & DistillPPO._compute_dagger_valid_mask(teacher_context).bool()).float()

    def _rollout_step(self, obs_dict):
        # Whole-motion probe resets would overwrite the sampled interval.
        # Uniform segment sampling uses no full-motion schedule adapter.
        return DistillPPO._rollout_step(self, obs_dict)

    def _post_epoch_logging(self, it, loss_dict):
        result = DistillPPO._post_epoch_logging(self, it, loss_dict)
        cmd = self.env.command_manager.get_state('motion_command')
        statistics = getattr(cmd, 'trajectory_statistics', cmd.segment_statistics)
        if self.is_main_process and it % self.config.save_interval == 0:
            statistics.write_csv(self.log_dir, it)
        statistics.clear_interval()
        return result


@dataclass(frozen=True)
class Options:
    manifest: str
    teacher_spec: str
    output: str = 'runtime/current/models/php_segment_distill'
    name: str = 'contact_segments'
    num_envs: int = 4096
    iterations: int = 20000
    save_interval: int = 100
    checkpoint: str | None = None
    segment_start_probability: float = .2
    contact_weight: float = 1.
    goal_contact_weight: float = 1.
    pure_distill_iters: int | None = None
    regularizer_mse_coef: float = .1
    segment_mode: Literal['independent','continuous'] = 'independent'
    validate_only: bool = False


def distillation_schedule(config, pure_iters, mse_coef):
    """Keep the legacy schedule unless the two-stage experiment is explicit."""
    if pure_iters is None:
        return config
    if (type(pure_iters) is not int or pure_iters < 0
            or not math.isfinite(mse_coef) or mse_coef <= 0
            or config.dagger_coef <= 0 or mse_coef > config.dagger_coef):
        raise ValueError('Invalid pure-distillation phase or MSE regularizer')
    return replace(config, pure_distill_iters=pure_iters,
        lambda_dagger_init=1., lambda_dagger_final=mse_coef/config.dagger_coef,
        lambda_ppo_init=0., lambda_ppo_final=.99)


def build_config(options):
    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.reward import RewardTermCfg
    from holosoma.config_types.termination import TerminationTermCfg
    from holosoma.config_values.wbt.g1.modular import wbt_student
    from holosoma.php_contact.train import Options as ContactOptions, build_config as contact_config
    from holosoma.php_contact.dataset import load_prepared, digest
    from holosoma.utils.wbt_experiment import prepare_routed_distillation
    from somaforge_core.published_data import load_published_json, resolve_published_path
    from holosoma.utils.eval_utils import _experiment_config_from_serialized
    import torch

    prepared = load_prepared(options.manifest)
    if prepared.get('reference_origin') != 'matched_teacher_existing_extraction_standard':
        raise ValueError('Segment distillation requires task-matched teacher rollout extraction, not old augmentations')
    spec_path = resolve_published_path(options.teacher_spec)
    specs = load_published_json(spec_path)['teachers']
    if len(specs) != 1:
        raise ValueError('This task-matched segment adapter requires one explicitly supplied teacher')
    checkpoint = resolve_published_path(specs[0]['checkpoint'], base_dir=spec_path.parent)
    if digest(checkpoint) != prepared['teacher']['sha256']:
        raise ValueError('Segment labels and selected teacher checkpoint differ')
    teacher = _experiment_config_from_serialized(torch.load(checkpoint, map_location='cpu',
        weights_only=False, mmap=True)['experiment_config'])
    original_manifest = str(resolve_published_path(teacher.command.setup_terms['motion_command'].params['motion_config'].motion_manifest))
    if digest(original_manifest) != prepared['teacher']['training_manifest_sha256']:
        raise ValueError('Original teacher training task manifest changed')
    if (not 0 <= options.segment_start_probability <= 1
            or min(options.contact_weight, options.goal_contact_weight) < 0):
        raise ValueError('Invalid segment sampling probability or contact reward weights')
    cfg = contact_config(ContactOptions(manifest=options.manifest, output=options.output, name=options.name,
        num_envs=options.num_envs, iterations=options.iterations, save_interval=options.save_interval,
        checkpoint=options.checkpoint, probes=False))
    original_motion = teacher.command.setup_terms['motion_command'].params['motion_config']
    student_motion = cfg.command.setup_terms['motion_command'].params['motion_config']
    if (original_motion.body_names_to_track != student_motion.body_names_to_track
            or original_motion.body_name_ref != student_motion.body_name_ref):
        raise ValueError('Teacher and student reference-body layouts must match')
    source = cfg.command.setup_terms['motion_command']
    motion = replace(student_motion, reset_sampler='uniform', use_start_probe_envs=False,
        use_group_probe_envs=False, start_at_timestep_zero_prob=0., use_completion_learning_sampler=False)
    prefix = 'holosoma.php_contact.segments:'
    func = prefix+'SegmentCommand'
    if options.segment_mode == 'continuous':
        func = 'holosoma.php_contact.continuous:ContinuousContactCommand'
    command = replace(cfg.command,
        setup_terms={'motion_command': replace(source, func=func, params={**source.params,
            'motion_config': motion, 'teacher_training_manifest': original_manifest,
            'segment_start_probability': options.segment_start_probability, 'segment_mode':options.segment_mode})},
        reset_terms={'motion_command': CommandTermCfg(func=func)},
        step_terms={'motion_command': CommandTermCfg(func=func)})
    actor = cfg.observation.groups['actor_obs']
    observation = replace(cfg.observation, groups={**cfg.observation.groups,
        'actor_obs': replace(actor, terms={**actor.terms,
            'contacts': replace(actor.terms['contacts'], func=prefix+'future_contacts'),
            'goal': replace(actor.terms['goal'], func=prefix+'goal')})})
    rewards = replace(cfg.reward, terms={**cfg.reward.terms,
        'native_contact_match': RewardTermCfg(func=prefix+'contact_reward', weight=options.contact_weight),
        'segment_goal_contact': RewardTermCfg(func=prefix+'goal_contact_reward', weight=options.goal_contact_weight)})
    termination = replace(cfg.termination, terms={**cfg.termination.terms,
        'segment_boundary': TerminationTermCfg(func=prefix+'segment_boundary', is_timeout=True)})
    if options.segment_mode == 'continuous':
        termination = replace(termination, terms={**termination.terms,
            'motion_ends':TerminationTermCfg(func='holosoma.php_contact.continuous:trajectory_complete',is_timeout=True)})
    algorithm = replace(wbt_student.algo, _target_='holosoma.php_contact.distill.SegmentDistillPPO',
        config=replace(wbt_student.algo.config, module_dict=cfg.algo.config.module_dict,
            num_learning_iterations=options.iterations, save_interval=options.save_interval,
            init_at_random_ep_len=False, distill=replace(wbt_student.algo.config.distill,
                teacher_spec_file=str(spec_path), teacher_obs_keys=('expert_route',),
                dagger_bad_ref_pos_threshold=teacher.termination.terms['bad_tracking'].params['bad_ref_pos_threshold'],
                dagger_bad_ref_ori_threshold=teacher.termination.terms['bad_tracking'].params['bad_ref_ori_threshold'],
                dagger_bad_motion_body_pos_threshold=teacher.termination.terms['bad_tracking'].params['bad_motion_body_pos_threshold'],
                dagger_bad_motion_body_pos_body_names=tuple(teacher.termination.terms['bad_tracking'].params[
                    'bad_motion_body_pos_body_names']))))
    cfg = replace(cfg, command=command, observation=observation, reward=rewards, termination=termination,
        algo=replace(algorithm, config=replace(algorithm.config,
            distill=distillation_schedule(algorithm.config.distill,
                options.pure_distill_iters, options.regularizer_mse_coef))),
        robot=replace(cfg.robot, control=teacher.robot.control),
        training=replace(cfg.training, project='PHPSegmentDistill'))
    return prepare_routed_distillation(cfg)


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args(__doc__)
    import tyro
    options = tyro.cli(Options)
    config = sync_launcher_headless_config(build_config(options), launcher)
    print(json.dumps(dict(num_envs=options.num_envs, iterations=options.iterations, algorithm=config.algo._target_,
        segment_mode=options.segment_mode, teacher_spec=options.teacher_spec,
        contact_weight=options.contact_weight, goal_contact_weight=options.goal_contact_weight,
        pure_distill_iters=options.pure_distill_iters,
        post_distill_actor_loss=(f'0.99 * PPO + {options.regularizer_mse_coef} * MSE'
            if options.pure_distill_iters is not None else 'legacy linear curriculum'),
        training_started=not options.validate_only), indent=2), flush=True)
    if not options.validate_only:
        from holosoma.train_agent import train
        train(config, launcher_args=launcher)


if __name__ == '__main__':
    main()
