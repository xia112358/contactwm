from __future__ import annotations

import copy
import dataclasses
import math
import os

import torch
from loguru import logger
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import EmpiricalNormalization, PPO
from holosoma.config_types.algo import DistillPPOConfig
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.rotations import quat_rotate_inverse


class DistillPPO(PPO):
    """PPO variant with optional frozen-teacher action-distribution distillation.

    This class is intentionally separate from the default PPO implementation so
    existing PPO training remains unchanged unless the algo preset explicitly
    points at ``holosoma.agents.ppo.distill_ppo.DistillPPO``.
    """

    config: DistillPPOConfig

    def _init_config(self) -> None:
        super()._init_config()
        distill_cfg = getattr(self.config, "distill", None)
        teacher_obs_keys = getattr(distill_cfg, "teacher_obs_keys", None)
        self.teacher_obs_keys = list(teacher_obs_keys) if teacher_obs_keys is not None else list(self.actor_obs_keys)
        self._teacher_uses_student_actor_obs = self.teacher_obs_keys == self.actor_obs_keys

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        self.teacher_actor: nn.Module | None = None
        self.teacher_actor_obs_normalizer: nn.Module = nn.Identity()
        distill_cfg = self.config.distill
        if not distill_cfg.enable_kl:
            return
        if not distill_cfg.teacher_checkpoint_path:
            raise ValueError("DistillPPO requires distill.teacher_checkpoint_path when distill.enable_kl=True")

        init_noise_std = distill_cfg.teacher_init_noise_std or self.config.init_noise_std
        teacher_actor_cfg = dataclasses.replace(
            copy.deepcopy(self.config.module_dict.actor),
            input_dim=list(self.teacher_obs_keys),
        )
        if distill_cfg.teacher_module_type is not None:
            teacher_actor_cfg = dataclasses.replace(teacher_actor_cfg, type=distill_cfg.teacher_module_type)
        if distill_cfg.teacher_hidden_dims is not None:
            teacher_actor_cfg = dataclasses.replace(
                teacher_actor_cfg,
                layer_config=dataclasses.replace(
                    teacher_actor_cfg.layer_config,
                    hidden_dims=list(distill_cfg.teacher_hidden_dims),
                ),
            )
        self.teacher_actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=teacher_actor_cfg,
            num_actions=self.num_act,
            init_noise_std=init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
            action_clip=self.config.action_clip,
        )
        self.teacher_actor.eval()
        for param in self.teacher_actor.parameters():
            param.requires_grad_(False)

        teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)
        if self.empirical_normalization:
            self.teacher_actor_obs_normalizer = EmpiricalNormalization(shape=teacher_obs_dim, device=self.device)

        checkpoint_path = os.path.expanduser(distill_cfg.teacher_checkpoint_path)
        checkpoint = self._load_checked_checkpoint(checkpoint_path)
        self.teacher_actor.load_state_dict(checkpoint["actor_model_state_dict"])
        normalizer_state = checkpoint.get("actor_obs_normalizer_state_dict")
        if self.empirical_normalization and normalizer_state is not None:
            self.teacher_actor_obs_normalizer.load_state_dict(normalizer_state)
        self.teacher_actor_obs_normalizer.eval()
        logger.info(f"Loaded frozen DistillPPO teacher actor from {checkpoint_path}")

    def _setup_storage(self):
        super()._setup_storage()
        if not self._teacher_uses_student_actor_obs:
            teacher_obs_dim = self._get_obs_dim(self.teacher_obs_keys)
            self.storage.register("teacher_actor_obs", shape=(teacher_obs_dim,), dtype=torch.float)
        if (
            self.config.distill.enable_kl
            and not self._teacher_uses_student_actor_obs
            and self.config.distill.distill_type in ("kl", "mse_dagger", "php_dagger_ppo")
        ):
            self.storage.register("dagger_valid_mask", shape=(1,), dtype=torch.float)

    def _rollout_step(self, obs_dict):
        self._update_php_student_tracking_boundary()
        return super()._rollout_step(obs_dict)

    def _rollout_observation_extras(self, obs_dict):
        if self._teacher_uses_student_actor_obs:
            return {}
        extras = {
            "teacher_actor_obs": self._finite_tensor(
                torch.cat([obs_dict[k] for k in self.teacher_obs_keys], dim=1), clamp=1.0e6
            ),
        }
        if (
            self.config.distill.enable_kl
            and self.config.distill.distill_type in ("kl", "mse_dagger", "php_dagger_ppo")
        ):
            extras["dagger_valid_mask"] = self._compute_dagger_valid_mask().view(-1, 1)
        return extras

    def _update_php_student_tracking_boundary(self) -> None:
        anneal_iters = int(getattr(self.config.distill, "php_student_termination_anneal_iters", 0))
        if anneal_iters <= 0:
            return
        term = getattr(getattr(self.env, "termination_manager", None), "_term_instances", {}).get(
            "bad_tracking"
        )
        if term is None:
            return
        progress = min(max(float(self.current_learning_iteration) / float(anneal_iters), 0.0), 1.0)
        term.bad_ref_pos_threshold = 0.5 + 0.5 * progress
        term.bad_ref_ori_threshold = 0.8 + 0.8 * progress
        term.bad_motion_body_pos_threshold = 0.25 + 0.25 * progress
        term.bad_object_pos_threshold = 0.5
        term.bad_object_ori_threshold = 0.8 + 0.8 * progress

    def _compute_dagger_valid_mask(self) -> torch.Tensor:
        distill_cfg = self.config.distill
        valid = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        if distill_cfg.dagger_disable_on_term_names and self.env.termination_manager is not None:
            for term_name in distill_cfg.dagger_disable_on_term_names:
                term_done = self.env.termination_manager.term_dones.get(term_name)
                if term_done is not None:
                    valid &= ~term_done.to(device=self.device, dtype=torch.bool)

        if distill_cfg.dagger_valid_use_bad_tracking:
            motion_command = self.env.command_manager.get_state("motion_command")
            bad_ref_pos = (
                torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1)
                > float(distill_cfg.dagger_bad_ref_pos_threshold)
            )
            motion_projected_gravity_b = quat_rotate_inverse(
                motion_command.ref_quat_w,
                gravity_vector(self.env),
                w_last=True,
            )
            robot_projected_gravity_b = quat_rotate_inverse(
                motion_command.robot_ref_quat_w,
                gravity_vector(self.env),
                w_last=True,
            )
            bad_ref_ori = (
                torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2])
                > float(distill_cfg.dagger_bad_ref_ori_threshold)
            )

            body_names = tuple(distill_cfg.dagger_bad_motion_body_pos_body_names)
            bad_body_pos = torch.zeros_like(valid)
            if body_names:
                tracked_body_names = list(motion_command.motion_cfg.body_names_to_track)
                body_idx = [tracked_body_names.index(name) for name in body_names if name in tracked_body_names]
                if body_idx:
                    body_idx_tensor = torch.tensor(body_idx, dtype=torch.long, device=self.device)
                    error = torch.norm(
                        motion_command.body_pos_relative_w.index_select(1, body_idx_tensor)
                        - motion_command.robot_body_pos_w.index_select(1, body_idx_tensor),
                        dim=-1,
                    )
                    bad_body_pos = torch.any(error > float(distill_cfg.dagger_bad_motion_body_pos_threshold), dim=-1)

            valid &= ~(bad_ref_pos | bad_ref_ori | bad_body_pos)

        return valid.to(torch.float32)

    def _normalize_teacher_obs(self, teacher_obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        teacher_obs = self._finite_tensor(teacher_obs, clamp=1.0e6)
        if self.empirical_normalization:
            teacher_obs = self.teacher_actor_obs_normalizer(teacher_obs, update=update)
        return self._finite_tensor(teacher_obs, clamp=1.0e6)

    def _compute_ppo_loss(self, minibatch):
        loss_dict = super()._compute_ppo_loss(minibatch)
        distill_loss, distill_metric, lambda_d, ppo_lambda, valid_rate = self._compute_teacher_distill_loss(
            minibatch
        )
        distill_type = getattr(self.config.distill, "distill_type", "kl")
        if distill_type in ("mse_dagger", "php_dagger_ppo"):
            ppo_actor_loss = loss_dict["actor_loss"]
            pure_iters = getattr(self.config.distill, "pure_distill_iters", None)
            pure_phase = pure_iters is not None and self.current_learning_iteration < pure_iters
            ppo_actor_active = (ppo_lambda > 0).to(torch.float32)
            if pure_phase:
                # Exclude the PPO graph entirely, including entropy and value
                # learning. AdamW skips critic parameters with no gradients.
                loss_dict["actor_loss"] = distill_loss
                loss_dict["critic_loss"] = loss_dict["critic_loss"].detach() * 0
            else:
                loss_dict["actor_loss"] = ppo_lambda * ppo_actor_loss + distill_loss
            loss_dict["dagger_mse"] = distill_metric
            loss_dict["dagger_loss"] = distill_loss
            loss_dict["dagger_valid_rate"] = valid_rate
            loss_dict["lambda_d"] = lambda_d
            loss_dict["lambda_dagger"] = lambda_d
            loss_dict["lambda_ppo"] = ppo_lambda
            loss_dict["php_ppo_actor_active"] = ppo_actor_active
            loss_dict["php_ppo_actor_loss_used"] = ppo_actor_loss.detach()
            loss_dict["teacher_prior_loss"] = distill_loss
            loss_dict["teacher_kl_loss"] = torch.zeros_like(distill_loss)
            loss_dict["kl_to_teacher"] = torch.zeros_like(distill_metric)
        else:
            loss_dict["actor_loss"] = ppo_lambda * loss_dict["actor_loss"] + distill_loss
            loss_dict["teacher_kl_loss"] = distill_loss
            loss_dict["kl_to_teacher"] = distill_metric
            loss_dict["lambda_d"] = lambda_d
            loss_dict["lambda_ppo"] = ppo_lambda
            loss_dict["teacher_prior_loss"] = distill_loss
        return loss_dict

    def _compute_teacher_distill_loss(
        self, minibatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distill_cfg = self.config.distill
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        one = torch.ones((), dtype=torch.float32, device=self.device)
        if not distill_cfg.enable_kl or self.teacher_actor is None:
            return zero, zero, zero, one, one

        actor_obs = self._finite_tensor(minibatch["actor_obs"], clamp=1.0e6)
        teacher_obs = minibatch["actor_obs"] if self._teacher_uses_student_actor_obs else minibatch["teacher_actor_obs"]
        teacher_obs = self._finite_tensor(teacher_obs, clamp=1.0e6)
        lambda_d = torch.tensor(self._current_lambda_d(), dtype=torch.float32, device=self.device)
        ppo_lambda = torch.tensor(self._current_ppo_lambda(), dtype=torch.float32, device=self.device)
        if lambda_d <= 0.0:
            return zero, zero, lambda_d, ppo_lambda, one

        self.actor.act({"actor_obs": actor_obs})
        student_mean = self.actor.action_mean
        student_std = self.actor.action_std

        with torch.no_grad():
            if not self._teacher_uses_student_actor_obs:
                teacher_obs = self._normalize_teacher_obs(teacher_obs, update=False)
            self.teacher_actor.act({"actor_obs": teacher_obs})
            teacher_mean = self.teacher_actor.action_mean.detach()  # type: ignore[attr-defined]
            teacher_std = self.teacher_actor.action_std.detach()  # type: ignore[attr-defined]

        action_dim = min(student_mean.shape[-1], teacher_mean.shape[-1])
        student_mean = student_mean[:, :action_dim]
        student_std = student_std[:, :action_dim]
        teacher_mean = teacher_mean[:, :action_dim]
        teacher_std = teacher_std[:, :action_dim]

        if distill_cfg.kl_action_slice is not None:
            start, stop = distill_cfg.kl_action_slice
            student_mean = student_mean[:, start:stop]
            student_std = student_std[:, start:stop]
            teacher_mean = teacher_mean[:, start:stop]
            teacher_std = teacher_std[:, start:stop]

        student_mean = self._finite_tensor(student_mean, clamp=10.0)
        teacher_mean = self._finite_tensor(teacher_mean, clamp=10.0)
        std_min = float(distill_cfg.kl_std_min)
        std_max = 10.0
        student_std = self._finite_tensor(student_std, fill=std_min, clamp=std_max).clamp(min=std_min, max=std_max)
        teacher_std = self._finite_tensor(teacher_std, fill=std_min, clamp=std_max).clamp(min=std_min, max=std_max)

        distill_type = getattr(distill_cfg, "distill_type", "kl")
        if distill_type in ("mse_dagger", "php_dagger_ppo") or distill_cfg.use_mean_mse_fallback:
            mse_per_sample = (student_mean - teacher_mean).pow(2).mean(dim=-1)
            valid_mask = minibatch.get("dagger_valid_mask")
            if valid_mask is None:
                valid_mask = torch.ones_like(mse_per_sample)
            else:
                valid_mask = self._finite_tensor(valid_mask.view(-1).to(self.device), fill=0.0, clamp=1.0).clamp(
                    min=0.0, max=1.0
                )
            valid_count = valid_mask.sum().clamp_min(1.0)
            teacher_kl = ((mse_per_sample * valid_mask).sum() / valid_count).clamp(max=1.0e4)
            valid_rate = valid_mask.mean().detach()
            if distill_type in ("mse_dagger", "php_dagger_ppo"):
                dagger_coef = torch.tensor(float(getattr(distill_cfg, "dagger_coef", 1.0)), device=self.device)
                return (
                    lambda_d * dagger_coef * teacher_kl,
                    teacher_kl.detach(),
                    lambda_d,
                    ppo_lambda,
                    valid_rate,
                )
        else:
            kl_per_dim = (
                torch.log(teacher_std / student_std)
                + (student_std.pow(2) + (student_mean - teacher_mean).pow(2)) / (2.0 * teacher_std.pow(2))
                - 0.5
            )
            kl_per_dim = self._finite_tensor(kl_per_dim, clamp=1.0e4).clamp(min=0.0, max=1.0e4)
            valid_mask = minibatch.get("dagger_valid_mask")
            if valid_mask is None:
                valid_mask = torch.ones(kl_per_dim.shape[0], dtype=torch.float32, device=self.device)
            else:
                valid_mask = self._finite_tensor(valid_mask.view(-1).to(self.device), fill=0.0, clamp=1.0).clamp(
                    min=0.0, max=1.0
                )
            valid_2d = valid_mask.unsqueeze(-1)
            denom = (valid_mask.sum() * kl_per_dim.shape[-1]).clamp_min(1.0)
            teacher_kl = self._finite_tensor((kl_per_dim * valid_2d).sum() / denom, clamp=1.0e4)
            valid_rate = valid_mask.mean().detach()
        teacher_prior_coef = torch.tensor(
            float(getattr(distill_cfg, "teacher_prior_coef", 1.0)),
            dtype=torch.float32,
            device=self.device,
        )
        return lambda_d * teacher_prior_coef * teacher_kl, teacher_kl.detach(), lambda_d, ppo_lambda, valid_rate

    def _current_lambda_d(self) -> float:
        distill_cfg = self.config.distill
        distill_type = getattr(distill_cfg, "distill_type", "kl")
        if distill_type in ("mse_dagger", "php_dagger_ppo"):
            init = float(distill_cfg.lambda_dagger_init if distill_cfg.lambda_dagger_init is not None else distill_cfg.lambda_kl_init)
            final = float(
                distill_cfg.lambda_dagger_final if distill_cfg.lambda_dagger_final is not None else distill_cfg.lambda_kl_final
            )
        else:
            init = float(distill_cfg.lambda_kl_init)
            final = float(distill_cfg.lambda_kl_final)
        pure_iters = getattr(distill_cfg, "pure_distill_iters", None)
        if pure_iters is not None:
            return init if self.current_learning_iteration < pure_iters else final
        if distill_type == "php_dagger_ppo" or distill_cfg.lambda_kl_anneal_schedule == "php_parkour":
            anneal_iters = self._php_curriculum_anneal_iters()
            progress = min(max(float(self.current_learning_iteration) / float(anneal_iters), 0.0), 1.0)
            return init + (final - init) * progress
        anneal_iters = int(distill_cfg.lambda_kl_anneal_iters)
        if anneal_iters <= 0:
            return init
        progress = min(max(float(self.current_learning_iteration) / float(anneal_iters), 0.0), 1.0)
        if distill_cfg.lambda_kl_anneal_schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final + (init - final) * cosine
        return init + (final - init) * progress

    def _current_ppo_lambda(self) -> float:
        distill_cfg = self.config.distill
        pure_iters = getattr(distill_cfg, "pure_distill_iters", None)
        if pure_iters is not None:
            if self.current_learning_iteration < pure_iters:
                return 0.0
            final = distill_cfg.lambda_ppo_final
            return float(final) if final is not None else max(0.0, 1.0-self._current_lambda_d())
        distill_type = getattr(distill_cfg, "distill_type", "kl")
        if distill_type not in ("mse_dagger", "php_dagger_ppo"):
            if distill_cfg.lambda_kl_anneal_schedule == "php_parkour":
                return max(0.0, 1.0 - self._current_lambda_d())
            init = distill_cfg.lambda_ppo_init
            final = distill_cfg.lambda_ppo_final
            if init is None and final is None:
                return 1.0
            init = float(0.0 if init is None else init)
            final = float(1.0 if final is None else final)
            anneal_iters = self._php_curriculum_anneal_iters()
            progress = max(float(self.current_learning_iteration) / float(anneal_iters), 0.0)
            dagger_like = max(1.0 - final, 1.0 - progress)
            return max(init, 1.0 - dagger_like)
        dagger_lambda = self._current_lambda_d()
        if distill_type == "php_dagger_ppo" or distill_cfg.lambda_kl_anneal_schedule == "php_parkour":
            return max(0.0, 1.0 - dagger_lambda)
        init = distill_cfg.lambda_ppo_init
        final = distill_cfg.lambda_ppo_final
        if init is None and final is None:
            return max(0.0, 1.0 - dagger_lambda)
        init = float(0.0 if init is None else init)
        final = float(max(0.0, 1.0 - dagger_lambda) if final is None else final)
        anneal_iters = int(distill_cfg.lambda_kl_anneal_iters)
        if anneal_iters <= 0:
            return init
        progress = min(max(float(self.current_learning_iteration) / float(anneal_iters), 0.0), 1.0)
        if distill_cfg.lambda_kl_anneal_schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final + (init - final) * cosine
        return init + (final - init) * progress

    def _php_curriculum_anneal_iters(self) -> int:
        configured_iters = int(getattr(self.config.distill, "lambda_kl_anneal_iters", 0))
        if configured_iters > 0:
            return configured_iters
        return max(int(float(self.config.num_learning_iterations) * 0.5), 1)
