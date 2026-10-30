"""Small native ExperimentConfig defaults; observation/reward/algo remain replaceable."""
from dataclasses import replace

from holosoma.config_values.wbt.g1 import observation, reward
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_baseline_29 as dense,
    g1_29dof_wbt_climb00_php_student as student_base,
)


def expert(observations, rewards, name, actor_keys=("actor_obs",)):
    return replace(dense,
        training=replace(dense.training, name=name, num_envs=4096),
        observation=observations, reward=rewards,
        algo=replace(dense.algo, config=replace(dense.algo.config,
            module_dict=replace(dense.algo.config.module_dict,
                actor=replace(dense.algo.config.module_dict.actor, input_dim=list(actor_keys))))),
    )


wbt_expert_dense = expert(observation.g1_29dof_wbt_observation, reward.g1_29dof_wbt_reward, "wbt_expert_dense")
wbt_expert_sparse = expert(observation.g1_29dof_wbt_sparse_native_observation,
    reward.g1_29dof_wbt_native_sparse_reward, "wbt_expert_sparse", ("actor_obs", "actor_proprio_history"))
wbt_expert_force = expert(observation.g1_29dof_wbt_native_force_observation,
    reward.g1_29dof_wbt_native_force_reward, "wbt_expert_force")

_student_command = dense.command.setup_terms["motion_command"]
_student_motion = replace(_student_command.params["motion_config"],
    reset_sampler="uniform", use_start_probe_envs=False, use_group_probe_envs=False,
    probe_env_per_motion=1)
wbt_student = replace(student_base,
    training=replace(student_base.training, name="wbt_student", num_envs=2048),
    robot=dense.robot,
    terrain=dense.terrain,
    observation=observation.g1_29dof_wbt_student_contact_force_observation,
    reward=reward.g1_29dof_wbt_reward,
    command=replace(dense.command, setup_terms={**dense.command.setup_terms,
        "motion_command": replace(_student_command, params={**_student_command.params, "motion_config": _student_motion})}),
    algo=replace(student_base.algo,
        _target_="holosoma.agents.ppo.routed_distill_ppo.RoutedDistillPPO",
        config=replace(student_base.algo.config,
            save_interval=200,
            distill=replace(student_base.algo.config.distill,
                teacher_obs_keys=("expert_route",), motion_schedule_start=1000, probes_per_motion=1))),
    simulator=replace(student_base.simulator, config=replace(student_base.simulator.config,
        depth_ray_camera=replace(student_base.simulator.config.depth_ray_camera,
            width=87, height=58, max_distance=5.0))),
)

DEFAULTS = {name: value for name, value in globals().copy().items() if name in {
    "wbt_expert_dense", "wbt_expert_sparse", "wbt_expert_force", "wbt_student"}}
