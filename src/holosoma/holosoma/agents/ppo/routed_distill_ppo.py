"""Native DistillPPO with explicitly routed, independently configured experts."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.distill_ppo import DistillPPO
from holosoma.agents.ppo.ppo import EmpiricalNormalization, PPO
from holosoma.utils.wbt_experiment import checkpoint_hash


def _assert_restored(actual, expected, path):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype != expected.dtype or not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            raise ValueError(f"Resume state differs at {path}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Resume keys differ at {path}")
        for key in expected:
            _assert_restored(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError(f"Resume length differs at {path}")
        for index, value in enumerate(expected):
            _assert_restored(actual[index], value, f"{path}/{index}")
    elif actual != expected:
        raise ValueError(f"Resume value differs at {path}")


class RoutedTeacherBank(nn.Module):
    """Keep each teacher's network, observation order and normalization intact."""

    def __init__(self, specs, dimensions, histories, num_actions, device):
        super().__init__()
        from holosoma.utils.eval_utils import _experiment_config_from_serialized

        self.actors = nn.ModuleList()
        self.normalizers = nn.ModuleList()
        self.widths = []
        self.hashes = {}
        self.num_actions, self.device = num_actions, device
        for spec in specs:
            self.hashes[spec.checkpoint] = checkpoint_hash(spec.checkpoint)
            checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False, mmap=True)
            cfg = _experiment_config_from_serialized(checkpoint["experiment_config"])
            module = dataclasses.replace(cfg.algo.config.module_dict.actor, input_dim=list(spec.observation_keys))
            actor = setup_ppo_actor_module(dimensions, module, num_actions,
                cfg.algo.config.init_noise_std, device, histories, cfg.algo.config.action_clip)
            actor.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
            width = sum(dimensions[key] for key in spec.observation_keys)
            normalizer = EmpiricalNormalization(shape=width, device=device) if cfg.algo.config.empirical_normalization else nn.Identity()
            if cfg.algo.config.empirical_normalization:
                state = checkpoint.get("actor_obs_normalizer_state_dict")
                if state is None:
                    raise ValueError(f"{spec.checkpoint}: missing expert normalization")
                normalizer.load_state_dict(state, strict=True)
            self.actors.append(actor)
            self.normalizers.append(normalizer)
            self.widths.append(width)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def act(self, state):
        inputs = state["actor_obs"]
        if inputs.shape[1] != sum(self.widths) + 1:
            raise ValueError("Teacher input layout disagrees with the checkpoint configurations")
        routing = inputs[:, -1].long()
        if not torch.equal(routing.to(inputs.dtype), inputs[:, -1]) or ((routing < 0) | (routing >= len(self.actors))).any():
            raise ValueError("Invalid explicit expert routing")
        self.action_mean = torch.zeros(len(inputs), self.num_actions, device=inputs.device)
        self.action_std = torch.ones_like(self.action_mean)
        offset = 0
        for index, (actor, normalizer, width) in enumerate(zip(self.actors, self.normalizers, self.widths)):
            ids = torch.where(routing == index)[0]
            if ids.numel():
                observations = inputs[ids, offset:offset+width]
                observations = normalizer(observations, update=False) if isinstance(normalizer, EmpiricalNormalization) else normalizer(observations)
                # act_inference computes means without advancing an expert's random sampler.
                means = actor.act_inference({"actor_obs": observations})
                self.action_mean[ids] = means
                self.action_std[ids] = actor.std.expand(len(ids), -1)
            offset += width
        return self.action_mean


class RoutedDistillPPO(DistillPPO):
    """Reuse native PPO/DAgger updates, storage, checkpoints and curriculum."""

    def _setup_models_and_optimizer(self):
        PPO._setup_models_and_optimizer(self)
        self.teacher_actor_obs_normalizer = nn.Identity()
        self.teacher_actor = (RoutedTeacherBank(self.config.distill.teachers,
            self.algo_obs_dim_dict, self.algo_history_length_dict, self.num_act, self.device)
            if self.config.distill.enable_kl else None)
        self.motion_schedule = None
        self._routed_resuming = False

    def _normalize_teacher_obs(self, teacher_obs, update=False):
        # Expert normalization is per-teacher, inside the bank.
        return self._finite_tensor(teacher_obs, clamp=1.0e6)

    def _rollout_step(self, obs_dict):
        if self.motion_schedule is None:
            from holosoma.managers.observation.terms.wbt_experiment import route_table
            from holosoma.utils.distillation_motion_schedule import DistillationMotionSchedule
            command = self.env.command_manager.get_state("motion_command")
            table = route_table(self.env, tuple(spec.manifest for spec in self.config.distill.teachers))
            self.motion_schedule = DistillationMotionSchedule(command, table,
                start_iteration=self.config.distill.motion_schedule_start,
                probes_per_motion=self.config.distill.probes_per_motion)
        self.motion_schedule.advance(self.current_learning_iteration, preserve_statistics=self._routed_resuming)
        return super()._rollout_step(obs_dict)

    def _checkpoint_metadata(self, **kwargs):
        metadata = super()._checkpoint_metadata(**kwargs)
        metadata["routed_teacher_sha256"] = dict(self.teacher_actor.hashes) if self.teacher_actor is not None else {}
        metadata["student_observation_layout"] = {
            key: self.algo_obs_dim_dict[key] for key in self.actor_obs_keys
        }
        return metadata

    def load(self, ckpt_path):
        if ckpt_path:
            checkpoint = self._load_checked_checkpoint(ckpt_path)
            if checkpoint.get("routed_teacher_sha256") != self.teacher_actor.hashes:
                raise ValueError("Resume experts differ from the saved distillation run")
            layout = {key: self.algo_obs_dim_dict[key] for key in self.actor_obs_keys}
            if checkpoint.get("student_observation_layout") != layout:
                raise ValueError("Resume student observation layout differs")
        result = super().load(ckpt_path)
        self._routed_resuming = bool(ckpt_path)
        if ckpt_path:
            _assert_restored(self.actor.state_dict(), checkpoint["actor_model_state_dict"], "actor")
            _assert_restored(self.critic.state_dict(), checkpoint["critic_model_state_dict"], "critic")
            if self.config.load_optimizer:
                _assert_restored(self.actor_optimizer.state_dict(), checkpoint["actor_optimizer_state_dict"], "actor_optimizer")
                _assert_restored(self.critic_optimizer.state_dict(), checkpoint["critic_optimizer_state_dict"], "critic_optimizer")
            self._resume_audit = dict(checkpoint=str(ckpt_path), checkpoint_iteration=int(checkpoint["iter"]),
                next_iteration=self.current_learning_iteration, models_exact=True,
                optimizer_exact=bool(self.config.load_optimizer),
                actor_learning_rate=self.actor_learning_rate, critic_learning_rate=self.critic_learning_rate)
            (Path(self.log_dir)/"resume_audit.json").write_text(json.dumps(self._resume_audit, indent=2)+"\n")
        return result

    def _restore_pending_env_state_after_reset(self):
        expected = self._pending_env_state
        super()._restore_pending_env_state_after_reset()
        if expected is not None and "motion_sampler" in expected:
            actual = self.env.command_manager.get_state("motion_command").get_checkpoint_state()
            saved = expected["motion_sampler"]
            if actual["motion_keys"] != saved["motion_keys"]:
                raise ValueError("Resume sampler motion identities differ")
            fields = ("_probe_recent30_values", "_probe_recent30_count", "_probe_recent30_cursor", "_failure_window_failure_hist")
            for field in fields:
                _assert_restored(actual["tensors"][field], saved["tensors"][field], "sampler/"+field)
            self._resume_audit["probe_history_exact"] = True
            (Path(self.log_dir)/"resume_audit.json").write_text(json.dumps(self._resume_audit, indent=2)+"\n")

    def _post_epoch_logging(self, it, loss_dict):
        metrics = self.motion_schedule.metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(f"MotionSchedule/{key}", value, it)
        super()._post_epoch_logging(it, loss_dict)
