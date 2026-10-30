from __future__ import annotations

import itertools
import os
import time
from typing import TypedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.config_types.algo import PPOConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.rotations import quaternion_to_matrix, subtract_frame_transforms

console = Console()


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0e6, max=1.0e6)
        if self.training and update:
            self.update(x)
        if center:
            out = (x - self._mean) / (self._std + self.eps)
        else:
            out = x / (self._std + self.eps)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-1.0e6, max=1.0e6)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            batch_mean = batch_mean_shifted + self._mean
        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


def execution_mean_consistency_error(
    exec_mu: torch.Tensor,
    teacher_mu: torch.Tensor,
    fixed_scale: float,
) -> torch.Tensor:
    """Match execution-reference action means without involving policy variance."""
    if fixed_scale <= 0.0:
        raise ValueError(f"fixed_scale must be positive, got {fixed_scale}")
    normalized_error = (exec_mu - teacher_mu.detach()) / float(fixed_scale)
    return 0.5 * normalized_error.square().sum(dim=-1).mean()


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    actor_obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, actor_obs_dim), dtype: torch.float32
    """

    critic_obs: torch.Tensor
    """The observation of the critic.

    Shape: (mini_batch_size, critic_obs_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """


class PPO(BaseAlgo):
    config: PPOConfig

    def __init__(self, env: BaseTask, config: PPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self._init_config()

        self.current_learning_iteration = 0
        self._pending_env_state = None
        self._pending_env_state_restore = False
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        # Observation manager system - history is defined per-module in module_dict
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            group_name: group_cfg.history_length
            for group_name, group_cfg in self.env.observation_manager.cfg.groups.items()
        }

        self.num_act = self.env.robot_config.actions_dim

        self.actor_learning_rate = self.config.actor_learning_rate
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(self.actor_learning_rate, 1e-2)
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(self.actor_learning_rate, 1e-5)
        self.critic_learning_rate = self.config.critic_learning_rate
        self.max_critic_learning_rate = self.config.max_critic_learning_rate or max(self.critic_learning_rate, 1e-2)
        self.min_critic_learning_rate = self.config.min_critic_learning_rate or min(self.critic_learning_rate, 1e-5)

        # Observation related Config
        self.use_symmetry = self.config.use_symmetry
        self.empirical_normalization = self.config.empirical_normalization
        self._init_obs_keys()

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self):
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
            action_clip=self.config.action_clip,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self._actor_first_layer_gradient_mask: torch.Tensor | None = None
        self._actor_first_layer_weight: nn.Parameter | None = None
        self._configure_actor_finetune()
        self.anchor_actor: nn.Module | None = None
        if self.config.anchor_kl_checkpoint and self.config.anchor_kl_coef > 0.0:
            self.anchor_actor = setup_ppo_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=self.config.module_dict.actor,
                num_actions=self.num_act,
                init_noise_std=self.config.init_noise_std,
                device=self.device,
                history_length=self.algo_history_length_dict,
                action_clip=self.config.action_clip,
            )
            checkpoint_path = os.path.expanduser(self.config.anchor_kl_checkpoint)
            checkpoint = self._load_checked_checkpoint(checkpoint_path)
            self.anchor_actor.load_state_dict(checkpoint["actor_model_state_dict"])
            self.anchor_actor.eval()
            for param in self.anchor_actor.parameters():
                param.requires_grad_(False)
            logger.info(
                f"Loaded frozen PPO anchor actor from {checkpoint_path} "
                f"with anchor_kl_coef={self.config.anchor_kl_coef}"
            )

        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic.parameters(), lr=self.critic_learning_rate
        )

    def _configure_actor_finetune(self) -> None:
        mode = self.config.actor_finetune_mode
        valid_modes = {"none", "ref_q", "ref_qd", "last_layers", "full_actor"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown actor_finetune_mode={mode!r}; expected one of {sorted(valid_modes)}")
        if mode in {"none", "full_actor"}:
            return

        actor_layers = self.actor.actor_module.module
        linear_layers = [layer for layer in actor_layers if isinstance(layer, nn.Linear)]
        if not linear_layers:
            raise ValueError("Actor fine-tuning requires at least one Linear layer.")
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

        if mode == "last_layers":
            for layer in linear_layers[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            logger.info("Actor fine-tuning: updating the final two Linear layers only.")
            return

        first_layer = linear_layers[0]
        expected_reference_dim = 4 * 67
        if first_layer.in_features < expected_reference_dim:
            raise ValueError(
                f"Actor fine-tuning mode {mode!r} expects a four-frame 67D reference window at input start, "
                f"but first layer has only {first_layer.in_features} inputs."
            )
        mask = torch.zeros_like(first_layer.weight)
        within_frame = range(0, 29) if mode == "ref_q" else range(29, 58)
        selected_columns = [frame * 67 + column for frame in range(4) for column in within_frame]
        mask[:, selected_columns] = 1.0
        first_layer.weight.requires_grad_(True)
        first_layer.weight.register_hook(lambda grad: grad * mask)
        self._actor_first_layer_weight = first_layer.weight
        self._actor_first_layer_gradient_mask = mask
        logger.info(
            f"Actor fine-tuning: updating first-layer {mode} columns only "
            f"({len(selected_columns)}/{first_layer.in_features} input columns)."
        )

    def _mask_actor_optimizer_state(self) -> None:
        weight = self._actor_first_layer_weight
        mask = self._actor_first_layer_gradient_mask
        if weight is None or mask is None:
            return
        if weight.grad is not None:
            weight.grad.mul_(mask)
        state = self.actor_optimizer.state.get(weight, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.shape == weight.shape:
                value.mul_(mask)

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        """Compute total observation dimension for given observation keys."""
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            # Note: algo_obs_dim_dict from observation_manager.get_obs_dims() already includes history
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self):
        """
        Create a dummy (all-zero) input for the actor.

        During training, we cannot use the logic in `self.get_example_obs()`, since it resets environments mid-rollout.
        """
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _normalize_actor_obs(self, actor_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        actor_obs = self._finite_tensor(actor_obs, clamp=1.0e6)
        if self.empirical_normalization:
            actor_obs = self.actor_obs_normalizer(
                actor_obs,
                update=update and not self.config.freeze_actor_obs_normalizer,
            )
        return self._finite_tensor(actor_obs, clamp=1.0e6)

    def _normalize_critic_obs(self, critic_obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        critic_obs = self._finite_tensor(critic_obs, clamp=1.0e6)
        if self.empirical_normalization:
            critic_obs = self.critic_obs_normalizer(critic_obs, update=update)
        return self._finite_tensor(critic_obs, clamp=1.0e6)

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
            ("ppo_valid", (1,), torch.bool),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(key, shape=shape, dtype=dtype)

        if self.config.exec_consistency_coef > 0.0:
            if actor_obs_dim < 152:
                raise ValueError("Execution consistency requires a 152D q-only future reference prefix.")
            self.storage.register("exec_joint_pos", shape=(29,), dtype=torch.float)
            self.storage.register("exec_torso_pos_w", shape=(3,), dtype=torch.float)
            self.storage.register("exec_torso_quat_w", shape=(4,), dtype=torch.float)
            self.storage.register("exec_motion_id", shape=(), dtype=torch.long)
            self.storage.register("exec_motion_time_step", shape=(), dtype=torch.long)
            self.storage.register("exec_actor_obs", shape=(actor_obs_dim,), dtype=torch.float)
            self.storage.register("exec_consistency_valid", shape=(1,), dtype=torch.bool)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    def learn(self):
        self._train_mode()

        obs_dict = self.env.reset_all()

        # Environment reset initializes command-term buffers.  Restore learned
        # sampler/curriculum state only after that reset so resume does not
        # immediately erase it.  Call through for legacy checkpoints as well,
        # allowing environments to record an explicit non-restored status.
        self._restore_pending_env_state_after_reset()

        # Initialize environments with different episode length buffers
        # Must happen AFTER reset_all() to avoid being overwritten by reset
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        last_saved_iteration: int | None = None
        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                if self.config.export_onnx:
                    self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))
                last_saved_iteration = it

        if self.is_main_process and last_saved_iteration != self.current_learning_iteration:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            if self.config.export_onnx:
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _rollout_observation_extras(self, obs_dict):
        """Capture subclass observations before the environment advances."""
        return {}

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                profile_remaining = int(os.environ.get("HOLOSOMA_PROFILE_ROLLOUT_STEPS", "0") or 0)
                profile_this_step = profile_remaining > 0
                profile_sync = os.environ.get("HOLOSOMA_PROFILE_ROLLOUT_SYNC", "1") != "0"

                def _profile_mark():
                    if profile_this_step and profile_sync and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    return time.perf_counter()

                t0 = _profile_mark()
                # Environment step
                actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                actor_obs = self._normalize_actor_obs(actor_obs_raw)
                critic_obs = self._normalize_critic_obs(critic_obs_raw)
                observation_extras = self._rollout_observation_extras(obs_dict)
                t_obs = _profile_mark()

                actions = self.actor.act({"actor_obs": actor_obs})
                actions = self._finite_tensor(actions, clamp=self.config.action_clip)
                motion_command = self.env.command_manager.get_state("motion_command")
                probe_env_mask = getattr(motion_command, "_probe_env_mask", None)
                if probe_env_mask is None:
                    probe_env_mask = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                else:
                    probe_env_mask = probe_env_mask.to(device=self.device, dtype=torch.bool).clone()
                if torch.any(probe_env_mask):
                    # Competence probes use the same deterministic mean action
                    # as deployment/evaluation.  Their transitions are marked
                    # invalid for PPO below because they were not sampled from
                    # the behavior distribution.
                    actions = self._replace_competence_probe_actions(
                        actions,
                        self.actor.action_mean,
                        probe_env_mask,
                        self.config.action_clip,
                    )
                values = self._finite_tensor(self.critic.evaluate({"critic_obs": critic_obs}).detach(), clamp=1.0e4)
                exec_state = None
                if self.config.exec_consistency_coef > 0.0:
                    if motion_command is None:
                        raise RuntimeError("Execution consistency requires a motion_command state term.")
                    exec_state = {
                        "exec_joint_pos": self.env.simulator.dof_pos.detach().clone(),
                        "exec_torso_pos_w": motion_command.robot_ref_pos_w.detach().clone(),
                        "exec_torso_quat_w": motion_command.robot_ref_quat_w.detach().clone(),
                        "exec_motion_id": motion_command.motion_ids.detach().clone(),
                        "exec_motion_time_step": motion_command.time_steps.detach().clone(),
                    }
                t_policy = _profile_mark()

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})
                t_env = _profile_mark()

                for obs_key in obs_dict:
                    obs_dict[obs_key] = self._finite_tensor(obs_dict[obs_key].to(self.device), clamp=1.0e6)
                rewards, dones = rewards.to(self.device), dones.to(self.device)
                rewards = self._finite_tensor(rewards, clamp=1.0e4)
                t_post_env = _profile_mark()

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                timeout_env_ids = infos["time_outs"].to(self.device).nonzero(as_tuple=False).flatten()
                if timeout_env_ids.numel() > 0:
                    final_critic_obs = torch.cat(
                        [
                            infos["final_observations"][k][timeout_env_ids].to(self.device)
                            for k in self.critic_obs_keys
                        ],
                        dim=1,
                    )
                    final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                    final_values = self._finite_tensor(
                        self.critic.evaluate({"critic_obs": final_critic_obs}).detach(), clamp=1.0e4
                    )
                    final_rewards[timeout_env_ids] += self.config.gamma * torch.squeeze(final_values, 1)
                final_rewards = self._finite_tensor(final_rewards, clamp=1.0e4)

                actions_log_prob = self._finite_tensor(
                    self.actor.get_actions_log_prob(actions).detach().unsqueeze(1), clamp=1.0e4
                )
                t_logprob = _profile_mark()

                # Add transition to storage
                transition = {
                    "actor_obs": actor_obs,
                    "critic_obs": critic_obs,
                    "actions": actions,
                    "values": values,
                    "actions_log_prob": actions_log_prob,
                    "action_mean": self._finite_tensor(self.actor.action_mean.detach(), clamp=1.0e3),
                    "action_sigma": self._finite_tensor(
                        self.actor.action_std.detach(), fill=1.0, clamp=10.0
                    ).clamp_min(1.0e-6),
                    "ppo_valid": (~probe_env_mask).view(-1, 1),
                    "rewards": self._finite_tensor(rewards + final_rewards, clamp=1.0e4).view(-1, 1),
                    "dones": dones.view(-1, 1),
                }
                if exec_state is not None:
                    transition.update(exec_state)
                transition.update(observation_extras)
                self.storage.add(**transition)
                t_store = _profile_mark()

                # Reset actor and critic for completed envs
                self.actor.reset(dones)
                self.critic.reset(dones)
                t_reset = _profile_mark()

                if self.log_dir is not None:
                    # Update episode stats using logging helper
                    self.logging_helper.update_episode_stats(rewards, dones, infos)
                t_log = _profile_mark()

                if profile_this_step:
                    logger.info(
                        "Rollout profile step: obs={:.4f}s policy={:.4f}s env={:.4f}s post_env={:.4f}s "
                        "logprob={:.4f}s store={:.4f}s reset={:.4f}s log={:.4f}s total={:.4f}s",
                        t_obs - t0,
                        t_policy - t_obs,
                        t_env - t_policy,
                        t_post_env - t_env,
                        t_logprob - t_post_env,
                        t_store - t_logprob,
                        t_reset - t_store,
                        t_log - t_reset,
                        t_log - t0,
                    )
                    os.environ["HOLOSOMA_PROFILE_ROLLOUT_STEPS"] = str(profile_remaining - 1)

            self._synchronize_newton_rollout_boundary()

            # Return / Advantage computation
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
            last_values = self._finite_tensor(
                self.critic.evaluate({"critic_obs": last_critic_obs}).detach().to(self.device), clamp=1.0e4
            )
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
                self.storage["ppo_valid"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages
            if self.config.exec_consistency_coef > 0.0:
                self._build_execution_reference_observations()

        return obs_dict

    def _synchronize_newton_rollout_boundary(self) -> None:
        """Drain Newton's asynchronous CUDA work once per collected rollout."""
        simulator_cfg = getattr(getattr(self.env, "simulator", None), "simulator_config", None)
        simulator_name = str(getattr(simulator_cfg, "name", "")).strip().lower()
        if simulator_name == "isaaclab3_newton" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _build_execution_reference_observations(self) -> None:
        """Build [q, torso position, torso rotation-6D] execution futures after collection."""
        offsets = (1, 2, 4, 8)
        reference_dim = 4 * 38
        actor_obs = self.storage["actor_obs"]
        exec_actor_obs = self.storage["exec_actor_obs"]
        exec_actor_obs.copy_(actor_obs)
        valid = self.storage["exec_consistency_valid"]
        valid.zero_()

        joint_pos = self.storage["exec_joint_pos"]
        torso_pos = self.storage["exec_torso_pos_w"]
        torso_quat = self.storage["exec_torso_quat_w"]
        motion_id = self.storage["exec_motion_id"]
        motion_step = self.storage["exec_motion_time_step"]
        dones = self.storage["dones"].squeeze(-1)
        num_steps, num_envs = joint_pos.shape[:2]
        usable_steps = num_steps - max(offsets)
        if usable_steps <= 0:
            return

        base_pos = torso_pos[:usable_steps]
        base_quat = torso_quat[:usable_steps]
        frames = []
        validity = torch.ones((usable_steps, num_envs), dtype=torch.bool, device=self.device)
        for offset in offsets:
            target_pos = torso_pos[offset : offset + usable_steps]
            target_quat = torso_quat[offset : offset + usable_steps]
            pos_b, quat_b = subtract_frame_transforms(
                base_pos.reshape(-1, 3),
                base_quat.reshape(-1, 4),
                target_pos.reshape(-1, 3),
                target_quat.reshape(-1, 4),
            )
            rotation_6d = quaternion_to_matrix(quat_b, w_last=True)[..., :2].reshape(usable_steps, num_envs, 6)
            frames.append(
                torch.cat(
                    (
                        joint_pos[offset : offset + usable_steps],
                        pos_b.reshape(usable_steps, num_envs, 3),
                        rotation_6d,
                    ),
                    dim=-1,
                )
            )
            validity &= motion_id[offset : offset + usable_steps] == motion_id[:usable_steps]
            validity &= motion_step[offset : offset + usable_steps] == motion_step[:usable_steps] + offset

        # A done after action t resets the state stored at t+1, so no done may occur before t+8.
        for delta in range(max(offsets)):
            validity &= ~dones[delta : delta + usable_steps]

        exec_reference_raw = torch.stack(frames, dim=2).reshape(usable_steps, num_envs, reference_dim)
        validity &= torch.isfinite(exec_reference_raw).all(dim=-1)
        if self.empirical_normalization:
            mean = self.actor_obs_normalizer._mean[..., :reference_dim]
            std = self.actor_obs_normalizer._std[..., :reference_dim]
            exec_reference = (exec_reference_raw - mean) / (std + self.actor_obs_normalizer.eps)
        else:
            exec_reference = exec_reference_raw
        exec_actor_obs[:usable_steps, :, :reference_dim].copy_(
            self._finite_tensor(exec_reference, clamp=1.0e6)
        )
        valid[:usable_steps, :, 0].copy_(validity)

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards, ppo_valid=None):
        last_values = self._finite_tensor(last_values, clamp=1.0e4)
        values = self._finite_tensor(values, clamp=1.0e4)
        rewards = self._finite_tensor(rewards, clamp=1.0e4)
        advantage = torch.zeros_like(last_values)
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            advantage = self._finite_tensor(advantage, clamp=1.0e4)
            returns[step] = advantage + values[step]
        advantages = returns - values
        returns = self._finite_tensor(returns, clamp=1.0e4)
        advantages = self._finite_tensor(advantages, clamp=1.0e4)

        advantages = self._normalize_advantages(advantages, ppo_valid)
        advantages = self._finite_tensor(advantages, clamp=10.0)

        valid_mask = self._advantage_valid_mask(advantages, ppo_valid)
        valid_advantages = advantages[valid_mask]
        self._last_valid_advantage_mean = float(valid_advantages.mean().item())
        self._last_valid_advantage_std = float(valid_advantages.std(unbiased=False).item())

        return returns, advantages

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)

        minibatch: Minibatch
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        action_diagnostics = self._action_diagnostics()
        for minibatch in generator:
            loss_dict = self._update_algo_step(minibatch, loss_dict)

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        loss_dict.update(action_diagnostics)
        loss_dict["valid_advantage_mean"] = getattr(self, "_last_valid_advantage_mean", 0.0)
        loss_dict["valid_advantage_std"] = getattr(self, "_last_valid_advantage_std", 0.0)
        self.storage.clear()
        return loss_dict

    def _action_diagnostics(self) -> dict[str, float]:
        raw_mean = self.storage["action_mean"]
        actions = self.storage["actions"]
        action_clip = self.config.action_clip
        return {
            "raw_action_mean_abs_max": float(raw_mean.abs().max().item()),
            "raw_action_mean_out_of_bounds_fraction": float((raw_mean.abs() > action_clip).float().mean().item()),
            "executed_action_at_bound_fraction": float((actions.abs() >= action_clip).float().mean().item()),
        }

    @staticmethod
    def _filter_ppo_minibatch(minibatch: Minibatch) -> tuple[Minibatch, float]:
        """Remove deterministic competence-probe transitions from a PPO minibatch."""
        ppo_valid = minibatch["ppo_valid"].squeeze(-1).to(torch.bool)
        excluded_fraction = 1.0 - float(ppo_valid.float().mean().item())
        if torch.all(ppo_valid):
            return minibatch, excluded_fraction
        return {key: value[ppo_valid] for key, value in minibatch.items()}, excluded_fraction

    @staticmethod
    def _replace_competence_probe_actions(
        sampled_actions: torch.Tensor,
        action_mean: torch.Tensor,
        probe_env_mask: torch.Tensor,
        action_clip: float,
    ) -> torch.Tensor:
        """Use clipped policy means only for competence-probe environments."""
        actions = sampled_actions.clone()
        actions[probe_env_mask] = torch.nan_to_num(
            action_mean[probe_env_mask], nan=0.0, posinf=action_clip, neginf=-action_clip
        ).clamp(min=-action_clip, max=action_clip)
        return actions

    def _update_algo_step(self, minibatch: Minibatch, loss_dict: dict[str, float]):
        minibatch, excluded_fraction = self._filter_ppo_minibatch(minibatch)
        loss_dict["probe_excluded_fraction"] = loss_dict.get("probe_excluded_fraction", 0.0) + excluded_fraction
        if minibatch["ppo_valid"].shape[0] == 0:
            return loss_dict
        ppo_loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        if not torch.isfinite(ppo_loss):
            logger.warning("Skipping PPO minibatch update because loss is non-finite.")
            self._sanitize_module_parameters(self.actor)
            self._sanitize_module_parameters(self.critic)
            loss_dict["Value"] += 0.0
            loss_dict["Surrogate"] += 0.0
            loss_dict["Entropy"] += 0.0
            loss_dict["KL"] += 0.0
            loss_dict["nonfinite_update_count"] = loss_dict.get("nonfinite_update_count", 0.0) + 1.0
            return loss_dict
        ppo_loss.backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        self._mask_actor_optimizer_state()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        actor_nonfinite = self._sanitize_module_parameters(self.actor)
        critic_nonfinite = self._sanitize_module_parameters(self.critic)
        if actor_nonfinite or critic_nonfinite:
            logger.warning(
                f"Sanitized non-finite parameters after optimizer step: actor={actor_nonfinite}, critic={critic_nonfinite}"
            )
            loss_dict["nonfinite_parameter_count"] = loss_dict.get("nonfinite_parameter_count", 0.0) + float(
                actor_nonfinite + critic_nonfinite
            )

        loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
        loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
        loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
        loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
        for key, loss in ppo_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_value = loss.item() if torch.is_tensor(loss) else loss
            loss_dict[key] += loss_value
        return loss_dict

    @staticmethod
    def _sanitize_module_parameters(module: nn.Module) -> int:
        nonfinite_count = 0
        with torch.no_grad():
            for param in module.parameters():
                finite = torch.isfinite(param)
                if finite.all():
                    continue
                nonfinite_count += int((~finite).sum().item())
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                param.data.clamp_(min=-1.0e3, max=1.0e3)
        return nonfinite_count

    @staticmethod
    def _finite_tensor(tensor: torch.Tensor, fill: float = 0.0, clamp: float | None = None) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=fill, posinf=fill, neginf=fill)
        if clamp is not None:
            tensor = tensor.clamp(min=-clamp, max=clamp)
        return tensor

    @staticmethod
    def _nonfinite_count(tensor: torch.Tensor) -> float:
        return float((~torch.isfinite(tensor)).sum().item())

    def _compute_ppo_loss(self, minibatch: Minibatch):
        actions_batch = self._finite_tensor(minibatch["actions"], clamp=self.config.action_clip)
        target_values_batch = self._finite_tensor(minibatch["values"], clamp=1.0e4)
        advantages_batch = self._finite_tensor(minibatch["advantages"], clamp=10.0)
        returns_batch = self._finite_tensor(minibatch["returns"], clamp=1.0e4)
        old_actions_log_prob_batch = self._finite_tensor(minibatch["actions_log_prob"], clamp=1.0e4)
        old_mu_batch = self._finite_tensor(minibatch["action_mean"], clamp=1.0e3)
        old_sigma_batch = self._finite_tensor(minibatch["action_sigma"], fill=1.0, clamp=10.0).clamp_min(1.0e-6)

        # Symmetry augmentation
        original_batch_size = actions_batch.shape[0]
        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["actor_obs"],
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"],
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_batch = self.symmetry_utils.augment_actions(
                actions=actions_batch,
            )
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
        else:
            actor_obs = self._finite_tensor(minibatch["actor_obs"], clamp=1.0e6)
            critic_obs = self._finite_tensor(minibatch["critic_obs"], clamp=1.0e6)

        self.actor.act({"actor_obs": actor_obs})
        value_batch = self._finite_tensor(self.critic.evaluate({"critic_obs": critic_obs}), clamp=1.0e4)
        actions_log_prob_batch = self._finite_tensor(self.actor.get_actions_log_prob(actions_batch), clamp=1.0e4)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self._finite_tensor(self.actor.entropy[:original_batch_size], clamp=1.0e4)

        old_mu_batch = torch.nan_to_num(old_mu_batch, nan=0.0, posinf=1.0, neginf=-1.0)
        old_sigma_batch = torch.nan_to_num(old_sigma_batch, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6)
        mu_batch = torch.nan_to_num(mu_batch, nan=0.0, posinf=1.0, neginf=-1.0)
        sigma_batch = torch.nan_to_num(sigma_batch, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1.0e-6)

        kl_mean = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.config.desired_kl is not None and self.config.schedule == "adaptive":
            kl_mean = self._compute_kl_div(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
            adaptive_start_iter = int(getattr(self.config, "adaptive_schedule_start_iter", 0))
            if self.current_learning_iteration >= adaptive_start_iter:
                self._update_learning_rate(kl_mean)

        # Surrogate loss
        log_ratio = (actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)).clamp(min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            mean_actions_batch = self.actor.act_inference({"actor_obs": actor_obs.detach().clone()})
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )

            # Symmetry critic loss
            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = self._finite_tensor(entropy_batch.mean(), clamp=1.0e4)
        actor_loss = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )
        exec_consistency_mean_error = torch.zeros((), dtype=torch.float32, device=self.device)
        exec_consistency_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.config.exec_consistency_coef > 0.0:
            exec_valid = minibatch["exec_consistency_valid"].squeeze(-1)
            if exec_valid.any():
                exec_actor_obs = self._finite_tensor(minibatch["exec_actor_obs"][exec_valid], clamp=1.0e6)
                self.actor.act({"actor_obs": exec_actor_obs})
                exec_mu = self._finite_tensor(self.actor.action_mean, clamp=1.0e3)
                teacher_mu = old_mu_batch[exec_valid].detach()
                exec_consistency_mean_error = execution_mean_consistency_error(
                    exec_mu,
                    teacher_mu,
                    fixed_scale=self.config.init_noise_std,
                )
                exec_consistency_mean_error = self._finite_tensor(
                    exec_consistency_mean_error, clamp=1.0e4
                ).clamp(min=0.0)
                exec_consistency_loss = self.config.exec_consistency_coef * exec_consistency_mean_error
                actor_loss = actor_loss + exec_consistency_loss
        anchor_kl = torch.zeros((), dtype=torch.float32, device=self.device)
        anchor_kl_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.anchor_actor is not None and self.config.anchor_kl_coef > 0.0:
            with torch.no_grad():
                self.anchor_actor.act({"actor_obs": actor_obs[:original_batch_size]})
                anchor_mu_batch = self.anchor_actor.action_mean.detach()
                anchor_sigma_batch = self.anchor_actor.action_std.detach()

            action_dim = min(mu_batch.shape[-1], anchor_mu_batch.shape[-1])
            student_mu = self._finite_tensor(mu_batch[:, :action_dim], clamp=1.0e3)
            student_sigma = self._finite_tensor(sigma_batch[:, :action_dim], fill=1.0, clamp=10.0).clamp_min(1.0e-6)
            anchor_mu = self._finite_tensor(anchor_mu_batch[:, :action_dim], clamp=1.0e3)
            anchor_sigma = self._finite_tensor(
                anchor_sigma_batch[:, :action_dim], fill=1.0, clamp=10.0
            ).clamp_min(1.0e-6)
            anchor_kl = kl_divergence(Normal(student_mu, student_sigma), Normal(anchor_mu, anchor_sigma)).sum(-1).mean()
            anchor_kl = self._finite_tensor(anchor_kl, clamp=1.0e4).clamp(min=0.0)
            anchor_kl_loss = self.config.anchor_kl_coef * anchor_kl
            actor_loss = actor_loss + anchor_kl_loss

        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss
        actor_loss = self._finite_tensor(actor_loss, clamp=1.0e6)
        critic_loss = self._finite_tensor(critic_loss, clamp=1.0e6)
        value_loss = self._finite_tensor(value_loss, clamp=1.0e6)
        surrogate_loss = self._finite_tensor(surrogate_loss, clamp=1.0e6)
        entropy_loss = self._finite_tensor(entropy_loss, clamp=1.0e6)

        losses = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
        }
        if self.use_symmetry:
            losses["symmetry_actor_loss"] = symmetry_actor_loss
            losses["symmetry_critic_loss"] = symmetry_critic_loss
        if self.anchor_actor is not None and self.config.anchor_kl_coef > 0.0:
            losses["anchor_kl"] = anchor_kl.detach()
            losses["anchor_kl_loss"] = anchor_kl_loss.detach()
        if self.config.exec_consistency_coef > 0.0:
            losses["exec_consistency_mean_error"] = exec_consistency_mean_error.detach()
            losses["exec_consistency_loss"] = exec_consistency_loss.detach()
            losses["exec_consistency_valid_fraction"] = minibatch["exec_consistency_valid"].float().mean().detach()
        return losses

    def _compute_kl_div(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch) -> torch.Tensor:
        with torch.inference_mode():
            # Compute the KL divergence between the old and new action distributions
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_actor_learning_rate, self.actor_learning_rate / 1.5)
            self.critic_learning_rate = max(self.min_critic_learning_rate, self.critic_learning_rate / 1.5)
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(self.max_actor_learning_rate, self.actor_learning_rate * 1.5)
            self.critic_learning_rate = min(self.max_critic_learning_rate, self.critic_learning_rate * 1.5)

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = self._load_checked_checkpoint(ckpt_path)
            warm = getattr(self.config, "warm_start", False)
            # Resume the ordinary training state with a fresh simulator, even
            # when a historical checkpoint/config contains a runtime snapshot.
            self._pending_full_runtime_state = None
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.empirical_normalization and loaded_dict.get("actor_obs_normalizer_state_dict") is not None:
                self.actor_obs_normalizer.load_state_dict(loaded_dict["actor_obs_normalizer_state_dict"])
            if self.empirical_normalization and loaded_dict.get("critic_obs_normalizer_state_dict") is not None:
                self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer_state_dict"])
            if self.config.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"]["param_groups"][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            # Checkpoints are written after completing ``iter``. Resume at the
            # following iteration instead of applying that update twice.
            self.current_learning_iteration = int(loaded_dict["iter"]) + 1
            self._pending_env_state = None if warm else loaded_dict.get("env_state")
            self._pending_env_state_restore = True
            return loaded_dict.get("infos")
        return None

    def load_for_inference(self, ckpt_path: str | None) -> dict | None:
        """Load network and normalization weights only.

        Evaluation must not inherit optimizer moments, learning rates,
        iteration counters, or checkpointed environment/curriculum state.
        """
        if ckpt_path is None:
            return None
        logger.info(f"Loading inference weights from {ckpt_path}")
        loaded_dict = self._load_checked_checkpoint(ckpt_path)
        self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
        self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
        if self.empirical_normalization and loaded_dict.get("actor_obs_normalizer_state_dict") is not None:
            self.actor_obs_normalizer.load_state_dict(loaded_dict["actor_obs_normalizer_state_dict"])
        if self.empirical_normalization and loaded_dict.get("critic_obs_normalizer_state_dict") is not None:
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_normalizer_state_dict"])
        self._pending_env_state = None
        self._pending_env_state_restore = False
        self._pending_full_runtime_state = None
        return loaded_dict.get("infos")

    def _restore_pending_env_state_after_reset(self) -> None:
        """Restore checkpointed environment state after reset initialized its buffers."""

        if not self._pending_env_state_restore:
            return
        self._restore_env_state(self._pending_env_state)
        self._pending_env_state = None
        self._pending_env_state_restore = False

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_obs_normalizer_state_dict": (
                self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "critic_obs_normalizer_state_dict": (
                self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def export(self, onnx_file_path: str):
        """Export the `.onnx` of the policy to & save it to `path`.

        This is intended to enable deployment, but not resuming training.
        For storing checkpoints to resume training, see `PPO.save()`
        """
        # Save current training state
        was_training = self.actor.training

        # Set model to evaluation mode for export so we don't affect gradients mid-rollout
        self._eval_mode()

        # Save the .onnx file to filesystem
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                self.actor_onnx_wrapper,
                motion_command,
                onnx_file_path,
                self.device,
            )
        else:
            export_policy_as_onnx(
                wrapper=self.actor_onnx_wrapper,
                onnx_file_path=onnx_file_path,
                example_obs_dict={"actor_obs": self._get_zero_input()},
            )

        # Extract control gains and velocity limits & attach to onnx as metadata
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        action_scales = getattr(self.env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        # Extract URDF text from the robot config
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
        }
        metadata.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata=metadata,
        )

        # Upload the .onnx file to wandb
        self.logging_helper.save_to_wandb(onnx_file_path)

        # Restore original training state
        if was_training:
            self._train_mode()

    def _post_epoch_logging(self, it, loss_dict):
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std": self.actor.std.mean().item(),
            },
        }
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate

        command_manager = getattr(self.env, "command_manager", None)
        motion_command = None
        if command_manager is not None:
            motion_command = command_manager.get_state("motion_command")
            update_probe_weights = getattr(motion_command, "update_probe_motion_sampling_weights", None)
            if callable(update_probe_weights):
                update_probe_weights()
            get_motion_progress = getattr(motion_command, "get_motion_learning_progress_metrics", None)
            if callable(get_motion_progress):
                motion_progress = get_motion_progress()
                if motion_progress:
                    extra_log_dicts["MotionLearning"] = motion_progress

        # Use logging helper
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

        if motion_command is not None:
            write_distribution = getattr(motion_command, "write_adaptive_sampling_distribution", None)
            if callable(write_distribution):
                write_distribution(self.log_dir, it, interval=100)
            write_probe_distribution = getattr(motion_command, "write_start_probe_distribution", None)
            if callable(write_probe_distribution):
                write_probe_distribution(self.log_dir, it, interval=100)
            write_group_probe_distribution = getattr(motion_command, "write_group_probe_distribution", None)
            if callable(write_group_probe_distribution):
                write_group_probe_distribution(self.log_dir, it, interval=100)
            write_completion_learning_distribution = getattr(
                motion_command, "write_completion_learning_distribution", None
            )
            if callable(write_completion_learning_distribution):
                write_completion_learning_distribution(self.log_dir, it, interval=100)
            write_failure_window_distribution = getattr(
                motion_command, "write_failure_window_distribution", None
            )
            if callable(write_failure_window_distribution):
                write_failure_window_distribution(self.log_dir, it, interval=100)

    def _reduce_parameters(self):
        grads = [
            param.grad.view(-1)
            for model in [self.actor, self.critic]
            for param in model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for model in [self.actor, self.critic]:
            for param in model.parameters():
                if param.grad is not None:
                    numel = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad))
                    offset += numel

    def _synchronize_model_weights(self):
        """Synchronize actor and critic weights across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast critic weights from rank 0 to all other ranks
        for param in self.critic.parameters():
            torch.distributed.broadcast(param.data, src=0)

        logger.info(f"Synchronized model weights across {self.gpu_world_size} GPUs")

    @staticmethod
    def _advantage_valid_mask(advantages, ppo_valid):
        if ppo_valid is None:
            return torch.ones_like(advantages, dtype=torch.bool)
        valid_mask = ppo_valid.to(device=advantages.device, dtype=torch.bool)
        if valid_mask.shape != advantages.shape:
            valid_mask = valid_mask.expand_as(advantages)
        return valid_mask

    def _normalize_advantages(self, advantages, ppo_valid=None):
        valid_mask = self._advantage_valid_mask(advantages, ppo_valid)
        if self.is_multi_gpu:
            return self._normalize_advantages_multi_gpu(advantages, valid_mask)

        valid_advantages = advantages[valid_mask]
        if valid_advantages.numel() == 0:
            raise RuntimeError("Cannot normalize advantages without any PPO-valid transitions.")
        mean = valid_advantages.mean()
        variance = ((valid_advantages - mean) ** 2).mean()
        return (advantages - mean) / torch.sqrt(variance + 1e-8)

    def _normalize_advantages_multi_gpu(self, advantages, valid_mask):
        valid_advantages = advantages[valid_mask]
        local_stats = torch.stack(
            [
                valid_advantages.sum(),
                (valid_advantages**2).sum(),
                valid_mask.sum().to(dtype=advantages.dtype),
            ]
        )
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)

        global_count = local_stats[2]
        if global_count.item() <= 0:
            raise RuntimeError("Cannot normalize advantages without any PPO-valid transitions.")
        global_mean = local_stats[0] / global_count
        global_sq_mean = local_stats[1] / global_count
        global_variance = global_sq_mean - global_mean**2
        global_std = torch.sqrt(global_variance + 1e-8)

        return (advantages - global_mean) / global_std

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    @property
    def actor_onnx_wrapper(self):
        class ActorWrapper(nn.Module):
            def __init__(self, actor, actor_obs_normalizer, empirical_normalization):
                super().__init__()
                self.actor = actor
                self.actor_obs_normalizer = actor_obs_normalizer
                self.empirical_normalization = empirical_normalization

            def forward(self, actor_obs):
                if self.empirical_normalization:
                    actor_obs = self.actor_obs_normalizer(actor_obs, update=False)
                return self.actor.act_inference({"actor_obs": actor_obs})

        return ActorWrapper(self.actor, self.actor_obs_normalizer, self.empirical_normalization)

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        """Used for exporting policy as onnx."""
        obs_dict = self.env.reset_all()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1),
        }

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        # The policy-bootstrap reset below is the single authoritative reset for
        # an evaluation run.  Resetting once here and again below used to create
        # an extra all-environment episode before the first policy action.
        self._pre_evaluate_policy(reset_env=False)
        actor_state = self._create_actor_state()
        self.eval_policy = self.get_inference_policy()

        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"actions": init_actions})

        obs_dict = self.env.reset_all(
            pre_step_actor_state=actor_state,
            pre_step_callback=self._pre_eval_reset_bootstrap,
        )
        actor_state.update({"obs": obs_dict, "actions": init_actions})

        critic_obs = torch.cat([actor_state["obs"][k] for k in self.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs
        actor_state = self._post_eval_reset(actor_state)

        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            if actor_state.get("stop"):
                break

        self._post_evaluate_policy()

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        actor_obs = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        actions = self.eval_policy({"actor_obs": actor_obs})
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _pre_eval_reset_bootstrap(self, actor_state: dict):
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_reset_bootstrap(actor_state)
        return actor_state

    def _post_eval_reset(self, actor_state: dict):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_reset(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def get_inference_policy(self, device=None):
        self.actor.eval()  # switch to evaluation mode (dropout for example)
        self.actor_obs_normalizer.eval()
        if device is not None:
            self.actor.to(device)
            self.actor_obs_normalizer.to(device)

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = self._normalize_actor_obs(obs["actor_obs"], update=False)
            return self.actor.act_inference({"actor_obs": actor_obs})

        return policy_fn
