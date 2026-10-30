"""Evaluate an ACP Stage-1 controller repeatedly from reference frame zero."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random


@dataclass(frozen=True)
class Options:
    checkpoint: Path
    output_dir: Path
    episodes: int = 10
    seed: int = 42
    max_steps: int = 0
    randomization: bool = False


def summarize_episode(rows, *, completed, terminated, truncated, invalid_policy=False):
    frames = len(rows)
    contacts = sum(row['matched_contacts'] for row in rows)
    return dict(frames=frames,
        complete=bool(completed and not terminated and not truncated and not invalid_policy),
        terminated=bool(terminated), truncated=bool(truncated), invalid_policy=bool(invalid_policy),
        root_position_rms_m=math.sqrt(sum(row['root_error_sq'] for row in rows)/frames) if frames else None,
        reference_contact_position_rms_m=math.sqrt(sum(row['contact_error_sq'] for row in rows)/contacts) if contacts else None,
        matched_contact_samples=contacts,
        missed_contact_samples=sum(row['missed_contacts'] for row in rows),
        unexpected_contact_samples=sum(row['unexpected_contacts'] for row in rows))


def main():
    from holosoma.utils.sim_utils import parse_isaaclab_launcher_args, sync_launcher_headless_config
    launcher = parse_isaaclab_launcher_args('Evaluate ACP Stage 1 from frame zero')
    import tyro
    options = tyro.cli(Options)
    if options.episodes < 1 or options.max_steps < 0:
        raise ValueError('Evaluation needs positive episodes and a nonnegative step limit')
    import numpy as np
    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
    from holosoma.train_agent import TrainingContext
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.wbt_experiment import checkpoint_hash
    from holosoma.acp.contacts import BODY_MAPPING, measured_contacts
    from holosoma.acp.labels import PARTS
    from somaforge_core.contact_face_selection import FACE_POLICY
    from somaforge_core.newton_contacts import SCHEMA as CONTACT_SCHEMA
    from somaforge_core.robot_assets import canonical_g1_asset_metadata

    config, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(options.checkpoint)))
    term = config.command.setup_terms['motion_command']
    settings = {**term.params.get('acp', {})}
    if settings.get('stage') != 1:
        raise ValueError('This evaluator requires an ACP Stage-1 checkpoint')
    settings.update(observation_noise=options.randomization, domain_randomization=options.randomization,
                    evaluation_randomization=options.randomization)
    motion = term.params['motion_config']
    motion = replace(motion, reset_sampler='uniform', use_start_probe_envs=False, use_group_probe_envs=False,
        fixed_motion_assignment_in_eval=True, start_at_timestep_zero_prob=1., freeze_at_timestep_zero_prob=0.,
        noise_to_initial_pose=replace(motion.noise_to_initial_pose, overall_noise_scale=float(options.randomization)))
    command = replace(config.command, setup_terms={**config.command.setup_terms,
        'motion_command': replace(term, params={**term.params, 'motion_config': motion, 'acp': settings})})
    dynamics = 'holosoma.acp.randomization:PaperRandomization'
    randomization = RandomizationManagerCfg(
        setup_terms={'acp_dynamics': RandomizationTermCfg(func=dynamics, params={'enabled': options.randomization})},
        reset_terms={'acp_dynamics': RandomizationTermCfg(func=dynamics)})
    config = replace(config, command=command, randomization=randomization,
        training=replace(config.training, num_envs=1, checkpoint=None, seed=options.seed),
        terrain=replace(config.terrain, terrain_term=replace(config.terrain.terrain_term,
            spawn=replace(config.terrain.terrain_term.spawn, randomize_tiles=False, xy_offset_range=0.))))
    config = sync_launcher_headless_config(config, launcher)
    output = options.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.seed(options.seed); np.random.seed(options.seed); torch.manual_seed(options.seed)
    reports = []
    with TrainingContext(config, launcher_args=launcher):
        env = get_class(config.env_class)(get_tyro_env_config(config), device=launcher.device)
        env.set_is_evaluating()
        algo = get_class(config.algo._target_)(env=env, config=config.algo.config,
            log_dir=str(output/'inference'), device=launcher.device)
        algo.setup()
        algo.load_for_inference(str(options.checkpoint))
        algo._eval_mode()
        cmd = env.command_manager.get_state('motion_command')
        native_reward = env._compute_reward
        rows = []
        terminal = {}

        def capture():
            native_reward()
            actual = measured_contacts(env)
            index = cmd.time_steps[0]
            binding_path = output/'binding.json'
            if not binding_path.exists():
                binding_path.write_text(json.dumps(env._acp_contact_reader.binding, indent=2)+'\n')
            snapshot = actual['snapshot']
            count, capacity = int(snapshot['count']), len(snapshot['dist'])
            # Preserve every native candidate pair, including excluded faces;
            # selection changes only the aggregate metrics, never this audit.
            raw = {key: value[:count] if getattr(value, 'ndim', 0) > 0 and len(value) == capacity else value
                   for key, value in snapshot.items()}
            np.savez_compressed(episode_dir/f'frame_{len(rows):06d}.npz', **raw,
                reference_frame=int(index), post_physics_root_state=env.simulator.robot_root_states[0].cpu().numpy(),
                post_physics_joint_pos=env.simulator.dof_pos[0].cpu().numpy(),
                observed_active=actual['active'][0].cpu().numpy(), observed_known=actual['known'][0].cpu().numpy())
            origin = cmd._reference_env_origins[0]
            root_error = env.simulator.robot_root_states[0, :3]-cmd.motion.body_pos_w[index, 0]-origin
            known = actual['known'][0] & cmd.acp_reference_known[index]
            reference = cmd.acp_reference_active[index]
            active = actual['active'][0]
            matched = known & reference & active
            error = actual['position_w'][0]-cmd.acp_reference_position[index]-origin
            rows.append(dict(root_error_sq=float(root_error.square().sum()),
                contact_error_sq=float(error[matched].square().sum()), matched_contacts=int(matched.sum()),
                missed_contacts=int((known & reference & ~active).sum()),
                unexpected_contacts=int((known & ~reference & active).sum())))
            if bool(env.reset_buf[0]):
                terminal.update(completed=bool(env.termination_manager.term_dones['motion_ends'][0]),
                    terminated=bool(env.termination_manager.terminated[0]),
                    reasons=[name for name, value in env.termination_manager.term_dones.items() if bool(value[0])])

        env._compute_reward = capture
        with torch.inference_mode():
            for episode in range(options.episodes):
                rows, terminal = [], {}
                episode_dir = output/f'episode_{episode:03d}'
                episode_dir.mkdir()
                obs = env.reset_all(step_after_reset=False)
                start = cmd.motion.motion_start_idx[cmd.motion_ids[0]]
                if cmd.time_steps[0] != start:
                    raise RuntimeError('Independent ACP evaluation must start at reference frame zero')
                limit = options.max_steps or int(cmd.motion.motion_end_idx[cmd.motion_ids[0]]-start)+10
                invalid = False
                for _ in range(limit):
                    value = torch.cat([obs[key] for key in algo.actor_obs_keys], dim=1)
                    value = algo._normalize_actor_obs(value, update=False)
                    action = algo.actor.act_inference({'actor_obs': value})
                    if not torch.isfinite(action).all():
                        invalid = True
                        break
                    obs, _, done, _ = env.step({'actions': action})
                    if bool(done[0]):
                        break
                report = summarize_episode(rows, completed=terminal.get('completed', False),
                    terminated=terminal.get('terminated', False), truncated=not terminal and not invalid,
                    invalid_policy=invalid)
                report.update(episode=episode, termination_reasons=terminal.get('reasons', []))
                reports.append(report)
                print(json.dumps(report, allow_nan=False), flush=True)
        # Kit shutdown can terminate the process; persist before leaving its context.
        result = dict(schema='somaforge_acp_stage1_evaluation_v2', checkpoint=str(options.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_hash(options.checkpoint), seed=options.seed, from_frame_zero=True,
            randomization=options.randomization, startup_dynamics_samples=1,
            robot_asset=canonical_g1_asset_metadata(), contact_semantics=CONTACT_SCHEMA,
            activation='CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc',
            parts=PARTS, body_names_by_part=BODY_MAPPING, face_selection=FACE_POLICY,
            binding=str(output/'binding.json'), raw_contacts='episode_*/frame_*.npz',
            strict_pose_contact_same_state=False, sampling='post_physics_pose_and_last_native_solve_contacts',
            episodes=reports, successes=sum(row['complete'] for row in reports),
            success_rate=sum(row['complete'] for row in reports)/len(reports))
        (output/'evaluation_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'output': str(output), 'success_rate': result['success_rate']}), flush=True)


if __name__ == '__main__':
    main()
