"""PHP/WBT contact-conditioned tracker; data preparation and long training stay separate."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class Options:
    manifest: str
    output: str = 'runtime/current/models/php_contact'
    name: str = 'predictor_intervals'
    num_envs: int = 4096
    iterations: int = 20000
    save_interval: int = 100
    checkpoint: str | None = None
    probes: bool = True
    validate_only: bool = False
    allow_fixed_q_reference: bool = False


def build_config(options):
    from holosoma.config_types.command import CommandTermCfg
    from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
    from holosoma.config_types.reward import RewardTermCfg
    from holosoma.config_types.termination import TerminationTermCfg
    from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_climb00_php_expert as base
    from holosoma.config_values.wbt.g1.command import make_wbt_command
    from holosoma.config_values.wbt.g1.observation import php_student_actor_terms
    from holosoma.php_contact.dataset import load_prepared
    manifest = load_prepared(options.manifest)
    if manifest['split'] != 'train':
        raise ValueError('Training accepts only the Predictor train split; held-out motions stay separate')
    if min(options.num_envs, options.iterations, options.save_interval) < 1:
        raise ValueError('Environment, update and checkpoint counts must be positive')
    if options.probes and options.num_envs <= 3*len(manifest['motion_files']):
        raise ValueError('Not enough environments for PHP probes; use --no-probes for diagnostics')
    cmd = make_wbt_command(options.manifest, use_start_probe_envs=options.probes)
    source = cmd.setup_terms['motion_command']
    func = 'holosoma.php_contact.terms:ContactCommand'
    cmd = replace(cmd, setup_terms={'motion_command': replace(source, func=func)},
        reset_terms={'motion_command': CommandTermCfg(func=func)},
        step_terms={'motion_command': CommandTermCfg(func=func)})
    prefix = 'holosoma.php_contact.terms:'
    observations = ObservationManagerCfg(groups={
        'actor_obs': ObsGroupCfg(terms={
            'contacts': ObsTermCfg(func=prefix+'future_contacts'),
            'goal': ObsTermCfg(func=prefix+'goal'),
            'height': ObsTermCfg(func=prefix+'height_scan'),
            'actual_contacts': ObsTermCfg(func='holosoma.managers.observation.terms.wbt_experiment:actual_contacts',
                                         params={'mode': 'state'})}),
        'actor_proprio_history': ObsGroupCfg(history_length=5, enable_noise=True,
            terms={name: term for name, term in php_student_actor_terms.items() if name != 'velocity_command'}),
        'critic_obs': base.observation.groups['critic_obs']})
    rewards = replace(base.reward, terms={**base.reward.terms,
        'native_contact_match': RewardTermCfg(func=prefix+'contact_match', weight=1.)})
    termination = replace(base.termination, terms={**base.termination.terms,
        'motion_ends': TerminationTermCfg(func=prefix+'reference_complete', is_timeout=True)})
    algo = replace(base.algo, config=replace(base.algo.config,
        num_learning_iterations=options.iterations, num_steps_per_env=24,
        save_interval=options.save_interval, init_at_random_ep_len=False,
        module_dict=replace(base.algo.config.module_dict,
            actor=replace(base.algo.config.module_dict.actor,
                          input_dim=['actor_obs', 'actor_proprio_history']))))
    spacing = manifest['terrain_spacing']
    return replace(base, command=cmd, observation=observations, reward=rewards, termination=termination, algo=algo,
        training=replace(base.training, project='PHPContact', name=options.name, num_envs=options.num_envs,
                         checkpoint=options.checkpoint),
        logger=replace(base.logger, base_dir=options.output), nightly=None,
        terrain=replace(base.terrain, terrain_term=replace(base.terrain.terrain_term,
            motion_matched_manifest=options.manifest, terrain_length=spacing[0], terrain_width=spacing[1],
            spawn=replace(base.terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.))),
        simulator=replace(base.simulator, config=replace(base.simulator.config,
            sim=replace(base.simulator.config.sim, max_episode_length_s=25.))))


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args(__doc__)
    import tyro
    options = tyro.cli(Options)
    config = sync_launcher_headless_config(build_config(options), launcher)
    record = dict(num_envs=config.training.num_envs, iterations=config.algo.config.num_learning_iterations,
        steps_per_env=config.algo.config.num_steps_per_env, actor_inputs=config.algo.config.module_dict.actor.input_dim,
        reference_role='fixed_q_contact_intent; dynamic support/load unknown',
        manifest=str(Path(options.manifest).resolve()), validate_only=options.validate_only)
    print(json.dumps(record, indent=2), flush=True)
    if options.validate_only:
        return
    if not options.allow_fixed_q_reference:
        raise ValueError('Source remains training_ready=false: experimental tracking requires '
                         '--allow-fixed-q-reference; this does not certify dynamic execution')
    from holosoma.train_agent import train
    train(config, launcher_args=launcher)


if __name__ == '__main__':
    main()
