from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from holosoma.agents.ppo.ppo import PPO, Minibatch
from holosoma.config_types.algo import KLEarlyStopPPOConfig
from loguru import logger
from torch import nn


class KLEarlyStopPPO(PPO):
    """PPO comparison variant that bounds actor drift within each rollout."""

    config: KLEarlyStopPPOConfig

    def __init__(self, *args, config: KLEarlyStopPPOConfig, **kwargs):
        if config.actor_early_stop_kl <= 0.0:
            raise ValueError("actor_early_stop_kl must be positive")
        if config.actor_rollback_kl <= config.actor_early_stop_kl:
            raise ValueError("actor_rollback_kl must be greater than actor_early_stop_kl")
        self._actor_updates_stopped = False
        self._accepted_actor_updates_current_rollout = 0
        self._suppress_parent_kl_schedule = False
        super().__init__(*args, config=config, **kwargs)

    def _mini_batch_generator(self) -> Iterator[Minibatch]:
        if not self.config.reshuffle_minibatches_each_epoch:
            yield from self.storage.mini_batch_generator(
                self.config.num_mini_batches,
                self.config.num_learning_epochs,
            )
            return

        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.config.num_mini_batches
        sampled_size = self.config.num_mini_batches * mini_batch_size
        flattened = {key: buffer.flatten(0, 1) for key, buffer in self.storage._buffers.items()}

        for _ in range(self.config.num_learning_epochs):
            indices = torch.randperm(sampled_size, requires_grad=False, device=self.storage.device)
            for mini_batch_index in range(self.config.num_mini_batches):
                start = mini_batch_index * mini_batch_size
                end = (mini_batch_index + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield {key: buffer[batch_indices] for key, buffer in flattened.items()}

    def _training_step(self) -> dict[str, float]:
        self._refresh_rollout_policy_statistics()
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        actor_update_attempts = 0
        actor_updates = 0
        critic_updates = 0
        rollback_count = 0
        max_pre_kl = 0.0
        max_post_kl = 0.0
        processed_minibatches = 0
        first_batch_diagnostics: dict[str, float] = {}
        action_diagnostics = self._action_diagnostics()
        self._actor_updates_stopped = False
        self._accepted_actor_updates_current_rollout = 0

        for raw_minibatch in self._mini_batch_generator():
            minibatch, excluded_fraction = self._filter_ppo_minibatch(raw_minibatch)
            loss_dict["probe_excluded_fraction"] = (
                loss_dict.get("probe_excluded_fraction", 0.0) + excluded_fraction
            )
            if minibatch["ppo_valid"].shape[0] == 0:
                continue
            processed_minibatches += 1
            ppo_losses = self._compute_ppo_loss(minibatch)
            pre_kl = float(ppo_losses["kl_mean"].item())
            max_pre_kl = max(max_pre_kl, pre_kl)
            if processed_minibatches == 1:
                new_mu = self._finite_tensor(self.actor.action_mean, clamp=1.0e3)
                new_sigma = self._finite_tensor(self.actor.action_std, fill=1.0, clamp=10.0).clamp_min(1.0e-6)
                old_mu = self._finite_tensor(minibatch["action_mean"], clamp=1.0e3)
                old_sigma = self._finite_tensor(
                    minibatch["action_sigma"], fill=1.0, clamp=10.0
                ).clamp_min(1.0e-6)
                mu_abs_diff = (new_mu - old_mu).abs()
                sigma_abs_diff = (new_sigma - old_sigma).abs()
                first_batch_diagnostics = {
                    "actor_first_pre_kl": pre_kl,
                    "actor_first_mu_abs_diff_mean": float(mu_abs_diff.mean().item()),
                    "actor_first_mu_abs_diff_max": float(mu_abs_diff.max().item()),
                    "actor_first_sigma_abs_diff_max": float(sigma_abs_diff.max().item()),
                    "actor_first_obs_abs_max": float(minibatch["actor_obs"].abs().max().item()),
                    "actor_first_mean_abs_max": float(new_mu.abs().max().item()),
                    "actor_first_mean_gt10_fraction": float((new_mu.abs() > 10.0).float().mean().item()),
                }

            loss_dict["Value"] += float(ppo_losses["value_loss"].item())
            loss_dict["Surrogate"] += float(ppo_losses["surrogate_loss"].item())
            loss_dict["Entropy"] += float(ppo_losses["entropy_loss"].item())
            loss_dict["KL"] += pre_kl
            for key, value in ppo_losses.items():
                if key in {"value_loss", "surrogate_loss", "entropy_loss", "kl_mean"}:
                    continue
                loss_dict[key] = loss_dict.get(key, 0.0) + float(value.item() if torch.is_tensor(value) else value)

            if not self._actor_updates_stopped and pre_kl >= self.config.actor_early_stop_kl:
                self._actor_updates_stopped = True

            actor_loss = ppo_losses["actor_loss"]
            critic_loss = ppo_losses["critic_loss"]
            actor_finite = not self._actor_updates_stopped and bool(torch.isfinite(actor_loss).item())
            critic_finite = bool(torch.isfinite(critic_loss).item())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if actor_finite and critic_finite:
                (actor_loss + critic_loss).backward()
            elif actor_finite:
                actor_loss.backward()
            elif critic_finite:
                critic_loss.backward()

            if self.is_multi_gpu and (actor_finite or critic_finite):
                self._reduce_parameters()

            if actor_finite:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            if critic_finite:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

            actor_state = None
            actor_optimizer_state = None
            if actor_finite:
                actor_state, actor_optimizer_state = self._snapshot_actor_update(
                    self.actor,
                    self.actor_optimizer,
                )
                actor_update_attempts += 1
                self.actor_optimizer.step()
            elif not self._actor_updates_stopped:
                loss_dict["nonfinite_actor_update_count"] = loss_dict.get("nonfinite_actor_update_count", 0.0) + 1.0

            if critic_finite:
                self.critic_optimizer.step()
                critic_updates += 1
            else:
                loss_dict["nonfinite_critic_update_count"] = loss_dict.get("nonfinite_critic_update_count", 0.0) + 1.0

            actor_nonfinite = self._sanitize_module_parameters(self.actor)
            critic_nonfinite = self._sanitize_module_parameters(self.critic)
            if actor_nonfinite or critic_nonfinite:
                logger.warning(
                    "Sanitized non-finite parameters after optimizer step: "
                    f"actor={actor_nonfinite}, critic={critic_nonfinite}"
                )
                loss_dict["nonfinite_parameter_count"] = loss_dict.get("nonfinite_parameter_count", 0.0) + float(
                    actor_nonfinite + critic_nonfinite
                )

            if actor_finite:
                post_kl = self._post_update_kl(minibatch)
                max_post_kl = max(max_post_kl, post_kl)
                if post_kl >= self.config.actor_rollback_kl:
                    assert actor_state is not None and actor_optimizer_state is not None
                    self._restore_actor_update(
                        self.actor,
                        self.actor_optimizer,
                        actor_state,
                        actor_optimizer_state,
                    )
                    self._apply_rollback_learning_rate(post_kl)
                    rollback_count += 1
                    self._actor_updates_stopped = True
                else:
                    actor_updates += 1
                    self._accepted_actor_updates_current_rollout += 1
                    if post_kl >= self.config.actor_early_stop_kl:
                        self._actor_updates_stopped = True

        if processed_minibatches == 0:
            raise RuntimeError("KL early-stop PPO received no minibatches")
        for key in tuple(loss_dict):
            loss_dict[key] /= processed_minibatches

        loss_dict["actor_update_attempt_count"] = float(actor_update_attempts)
        loss_dict["actor_update_count"] = float(actor_updates)
        loss_dict["critic_update_count"] = float(critic_updates)
        loss_dict["actor_early_stop"] = float(self._actor_updates_stopped)
        loss_dict["actor_rollback_count"] = float(rollback_count)
        loss_dict["actor_pre_kl_max"] = max_pre_kl
        loss_dict["actor_post_kl_max"] = max_post_kl
        loss_dict.update(first_batch_diagnostics)
        loss_dict.update(action_diagnostics)
        self.storage.clear()
        return loss_dict

    def _refresh_rollout_policy_statistics(self) -> None:
        """Rebuild unclipped old-policy statistics from the rollout observations."""
        actor_obs = self.storage["actor_obs"].flatten(0, 1)
        action_mean = self.storage["action_mean"].flatten(0, 1)
        action_sigma = self.storage["action_sigma"].flatten(0, 1)
        batch_size = actor_obs.shape[0]
        chunk_size = max(1, batch_size // self.config.num_mini_batches)

        with torch.inference_mode():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                obs_chunk = self._finite_tensor(actor_obs[start:end], clamp=1.0e6)
                mean_chunk = self._finite_tensor(
                    self.actor.act_raw_inference({"actor_obs": obs_chunk}),
                    clamp=1.0e3,
                )
                sigma_chunk = self._finite_tensor(
                    self.actor.std.expand_as(mean_chunk),
                    fill=1.0,
                    clamp=10.0,
                ).clamp_min(1.0e-6)
                action_mean[start:end].copy_(mean_chunk)
                action_sigma[start:end].copy_(sigma_chunk)

    def _compute_ppo_loss(self, minibatch: Minibatch):
        self._suppress_parent_kl_schedule = True
        try:
            losses = super()._compute_ppo_loss(minibatch)
        finally:
            self._suppress_parent_kl_schedule = False

        original_batch_size = minibatch["action_mean"].shape[0]
        old_mu = self._finite_tensor(minibatch["action_mean"], clamp=1.0e3)
        old_sigma = self._finite_tensor(minibatch["action_sigma"], fill=1.0, clamp=10.0).clamp_min(1.0e-6)
        new_mu = self._finite_tensor(self.actor.action_mean[:original_batch_size], clamp=1.0e3)
        new_sigma = self._finite_tensor(
            self.actor.action_std[:original_batch_size], fill=1.0, clamp=10.0
        ).clamp_min(1.0e-6)
        kl_mean = self._compute_kl_div(old_mu, old_sigma, new_mu, new_sigma)
        losses["kl_mean"] = kl_mean

        if self.config.desired_kl is not None and self.config.schedule == "adaptive":
            adaptive_start_iter = int(getattr(self.config, "adaptive_schedule_start_iter", 0))
            if self.current_learning_iteration >= adaptive_start_iter:
                self._update_learning_rate(kl_mean)
        return losses

    @staticmethod
    def _snapshot_actor_update(actor: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
        return copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def _restore_actor_update(
        actor: nn.Module,
        optimizer: torch.optim.Optimizer,
        actor_state: dict,
        optimizer_state: dict,
    ) -> None:
        actor.load_state_dict(actor_state)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def _rollback_learning_rate(
        current_learning_rate: float,
        minimum_learning_rate: float,
        post_kl: float,
        target_kl: float,
    ) -> float:
        kl_backoff = (max(post_kl, target_kl) / target_kl) ** 0.5
        backoff = max(1.5, 1.1 * kl_backoff)
        return max(minimum_learning_rate, current_learning_rate / backoff)

    def _apply_rollback_learning_rate(self, post_kl: float) -> None:
        self.actor_learning_rate = self._rollback_learning_rate(
            self.actor_learning_rate,
            self.min_actor_learning_rate,
            post_kl,
            self.config.actor_early_stop_kl,
        )
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate

    def _post_update_kl(self, minibatch: Minibatch) -> float:
        actor_obs = self._finite_tensor(minibatch["actor_obs"], clamp=1.0e6)
        old_mu = self._finite_tensor(minibatch["action_mean"], clamp=1.0e3)
        old_sigma = self._finite_tensor(minibatch["action_sigma"], fill=1.0, clamp=10.0).clamp_min(1.0e-6)
        with torch.inference_mode():
            self.actor.act({"actor_obs": actor_obs})
            mu = self._finite_tensor(self.actor.action_mean, clamp=1.0e3)
            sigma = self._finite_tensor(self.actor.action_std, fill=1.0, clamp=10.0).clamp_min(1.0e-6)
            return float(self._compute_kl_div(old_mu, old_sigma, mu, sigma).item())

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        if self._suppress_parent_kl_schedule or self._actor_updates_stopped:
            return
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_actor_learning_rate, self.actor_learning_rate / 1.5)
        elif (
            self._accepted_actor_updates_current_rollout > 0
            and 0.0 < kl_mean < self.config.desired_kl / 2.0
        ):
            self.actor_learning_rate = min(self.max_actor_learning_rate, self.actor_learning_rate * 1.5)
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
