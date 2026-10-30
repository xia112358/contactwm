from __future__ import annotations

import math
from copy import deepcopy

import torch
from holosoma.config_types.algo import ModuleConfig
from torch import nn
from torch.distributions import Normal

from .modules import BaseModule


class CensoredNormal:
    """Normal distribution observed after hard clipping to symmetric bounds."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, action_clip: float):
        if action_clip <= 0.0:
            raise ValueError(f"action_clip must be positive, got {action_clip}")
        self.loc = loc
        self.scale = scale
        self.action_clip = float(action_clip)
        self.base_dist = Normal(loc, scale)

    def sample(self) -> torch.Tensor:
        return self.base_dist.sample().clamp(min=-self.action_clip, max=self.action_clip)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        lower = -self.action_clip
        upper = self.action_clip
        lower_z = (lower - self.loc) / self.scale
        upper_z = (upper - self.loc) / self.scale
        interior_log_prob = self.base_dist.log_prob(actions)
        lower_log_prob = torch.special.log_ndtr(lower_z)
        upper_log_prob = torch.special.log_ndtr(-upper_z)
        return torch.where(
            actions <= lower,
            lower_log_prob,
            torch.where(actions >= upper, upper_log_prob, interior_log_prob),
        )

    def entropy(self) -> torch.Tensor:
        lower_z = (-self.action_clip - self.loc) / self.scale
        upper_z = (self.action_clip - self.loc) / self.scale
        lower_mass = torch.special.ndtr(lower_z)
        upper_mass = torch.special.ndtr(-upper_z)
        interior_mass = (1.0 - lower_mass - upper_mass).clamp_min(0.0)

        normalizer = math.sqrt(2.0 * math.pi)
        lower_density = torch.exp(-0.5 * lower_z.square()) / normalizer
        upper_density = torch.exp(-0.5 * upper_z.square()) / normalizer
        continuous_entropy = interior_mass * (self.scale.log() + 0.5 * math.log(2.0 * math.pi * math.e))
        continuous_entropy += 0.5 * (lower_z * lower_density - upper_z * upper_density)
        min_mass = torch.finfo(self.loc.dtype).tiny
        boundary_entropy = -lower_mass * lower_mass.clamp_min(min_mass).log()
        boundary_entropy -= upper_mass * upper_mass.clamp_min(min_mass).log()
        return continuous_entropy + boundary_entropy


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict: ModuleConfig,
        num_actions,
        init_noise_std,
        history_length: dict[str, int],
        action_clip: float = 10.0,
    ):
        super().__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        self.actor_module = BaseModule(obs_dim_dict, module_config_dict, history_length)

        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be positive, got {init_noise_std}")
        self.log_std = nn.Parameter(math.log(init_noise_std) * torch.ones(num_actions))
        self.min_noise_std = module_config_dict.min_noise_std
        self.max_noise_std = module_config_dict.max_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        if action_clip <= 0.0:
            raise ValueError(f"action_clip must be positive, got {action_clip}")
        self.action_clip = float(action_clip)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        print(f"Actor Module: {self.actor_module.module}")

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict.output_dim):
            if output_dim == "robot_action_dim":
                module_config_dict.output_dim[idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.loc

    @property
    def action_std(self):
        return self.distribution.scale

    @property
    def std(self):
        return self._current_std()

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _current_std(self):
        with torch.no_grad():
            self.log_std.data = torch.nan_to_num(self.log_std.data, nan=math.log(0.1), posinf=2.0, neginf=-20.0)
            self.log_std.data.clamp_(min=-20.0, max=2.0)
        std = torch.exp(torch.clamp(self.log_std, min=-20.0, max=2.0))
        if self.max_noise_std:
            std = torch.clamp(std, max=self.max_noise_std)
        if self.min_noise_std:
            return torch.clamp(std, min=self.min_noise_std)
        if self.min_mean_noise_std:
            current_mean = std.mean()
            if current_mean < self.min_mean_noise_std:
                return std * (self.min_mean_noise_std / (current_mean + 1e-6))
        return std

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        self.distribution = CensoredNormal(mean, self._current_std().expand_as(mean), self.action_clip)

    def act(self, policy_state_dict):
        self.update_distribution(policy_state_dict["actor_obs"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict):
        raw_mean = PPOActor.act_raw_inference(self, policy_state_dict)
        return raw_mean.clamp(min=-self.action_clip, max=self.action_clip)

    def act_raw_inference(self, policy_state_dict):
        mean = self.actor(policy_state_dict["actor_obs"])
        return torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

    def to_cpu(self):
        self.actor_module = deepcopy(self.actor_module).to("cpu")
        self.log_std.data = self.log_std.data.to("cpu")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        old_std_key = f"{prefix}std"
        new_log_std_key = f"{prefix}log_std"
        if old_std_key in state_dict and new_log_std_key not in state_dict:
            state_dict[new_log_std_key] = state_dict.pop(old_std_key).clamp_min(1e-8).log()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict, history_length: dict[str, int]):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, module_config_dict, history_length)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        return self.critic(critic_obs)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorEncoder(PPOActor):
    def __init__(
        self,
        obs_dim_dict,
        module_config_dict,
        num_actions,
        init_noise_std,
        history_length: dict[str, int],
        action_clip: float = 10.0,
    ):
        super().__init__(
            obs_dim_dict,
            module_config_dict,
            num_actions,
            init_noise_std,
            history_length,
            action_clip,
        )
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name

    def _get_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != self.actor_module.input_dim:
            raise ValueError(f"Actor Obs must be {self.actor_module.input_dim}, got {actor_obs.shape[-1]}")
        self.encoder_obs = actor_obs[..., self.actor_module.input_indices_dict[self.encoder_input_name]]
        self.actor_encoder_obs = (
            self.actor_module.encoder(self.encoder_obs) if self.actor_module.encoder is not None else self.encoder_obs
        )
        self.actor_state_obs = torch.cat(
            [
                actor_obs[..., self.actor_module.input_indices_dict[actor_input_name]]
                for actor_input_name in self.module_input_name
            ],
            -1,
        )
        return torch.cat((self.actor_encoder_obs, self.actor_state_obs), dim=-1)

    def act(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor = self._get_input(actor_obs)
        return super().act({"actor_obs": input_actor})

    def act_inference(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor = self._get_input(actor_obs)
        return super().act_inference({"actor_obs": input_actor})

    def act_raw_inference(self, policy_state_dict):
        actor_obs = policy_state_dict["actor_obs"]
        input_actor = self._get_input(actor_obs)
        return super().act_raw_inference({"actor_obs": input_actor})


class PPOCriticEncoder(PPOCritic):
    def __init__(self, obs_dim_dict, module_config_dict):
        super().__init__(obs_dim_dict, module_config_dict)
        self.module_input_name = module_config_dict.layer_config.module_input_name
        self.encoder_input_name = module_config_dict.layer_config.encoder_input_name

    def _get_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape[-1] != self.critic_module.input_dim:
            raise ValueError(f"Critic Obs must be {self.critic_module.input_dim}, got {critic_obs.shape[-1]}")
        self.encoder_obs = critic_obs[..., self.critic_module.input_indices_dict[self.encoder_input_name]]
        self.critic_encoder_obs = (
            self.critic_module.encoder(self.encoder_obs) if self.critic_module.encoder is not None else self.encoder_obs
        )
        self.critic_state_obs = torch.cat(
            [
                critic_obs[..., self.critic_module.input_indices_dict[critic_input_name]]
                for critic_input_name in self.module_input_name
            ],
            -1,
        )
        return torch.cat((self.critic_encoder_obs, self.critic_state_obs), dim=-1)

    def evaluate(self, policy_state_dict):
        critic_obs = policy_state_dict["critic_obs"]
        input_critic = self._get_input(critic_obs)
        return super().evaluate({"critic_obs": input_critic})
