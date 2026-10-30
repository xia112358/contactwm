"""Native PPO plus independently optimized hindsight contact planning."""
from __future__ import annotations

import torch

from holosoma.acp.models import planner_loss
from holosoma.agents.ppo.ppo import PPO


class ACPPPO(PPO):
    def __init__(self, env, config, log_dir, device="cpu", multi_gpu_cfg=None):
        if multi_gpu_cfg is not None:
            raise ValueError("Distributed ACP planner/replay synchronization is not implemented")
        super().__init__(env, config, log_dir, device=device, multi_gpu_cfg=multi_gpu_cfg)

    def _setup_models_and_optimizer(self):
        super()._setup_models_and_optimizer()
        self.planner_optimizer = None
        if self.env.acp_planner is not None:
            self.planner_optimizer = torch.optim.Adam(self.env.acp_planner.parameters(), lr=1e-4)

    def _rollout_step(self, obs_dict):
        self.env.acp_iteration = self.current_learning_iteration
        return super()._rollout_step(obs_dict)

    def _training_step(self):
        losses = super()._training_step()
        losses.update(self._planner_training_step())
        return losses

    def _planner_training_step(self):
        losses = {}
        settings = self.env.acp_settings
        if self.planner_optimizer is None or self.env.acp_replay.size < settings["minimum_replay"]:
            return losses
        self.env.acp_planner.train()
        total = 0.0
        for _ in range(10):
            self.planner_optimizer.zero_grad(set_to_none=True)
            batch = self.env.acp_replay.sample(settings["planner_batch"], self.device)
            normalizers = ((batch["occupancy"] & batch["known"]).sum().clamp_min(1),
                           batch["known"].sum().clamp_min(1))
            sub_batches = min(8, len(batch["goal"]))
            for indices in torch.arange(len(batch["goal"]), device=self.device).tensor_split(sub_batches):
                sub = {key: value[indices] for key, value in batch.items()}
                prediction = self.env.acp_planner(sub["history"], sub["goal"], sub["terrain"], sub["valid"])
                objective = planner_loss(prediction, sub["terrain"], sub["target"], sub["occupancy"], sub["known"],
                                         normalizers=normalizers)
                loss = objective["loss"]
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.env.acp_planner.parameters(), 1.0)
            self.planner_optimizer.step()
        self.env.acp_planner.eval()
        losses["ACPPlanner"] = total/10
        return losses

    def _checkpoint_metadata(self, **kwargs):
        metadata = super()._checkpoint_metadata(**kwargs)
        metadata["acp"] = {"schema": "somaforge_acp_checkpoint_v1", "settings": self.env.acp_settings,
            "planner": None if self.env.acp_planner is None else self.env.acp_planner.state_dict(),
            "planner_optimizer": None if self.planner_optimizer is None else self.planner_optimizer.state_dict(),
            "replay": self.env.acp_replay.state_dict()}
        return metadata

    def _load_planner(self, path, *, training):
        checkpoint = self._load_checked_checkpoint(path)
        saved = checkpoint.get("acp")
        if not saved or saved.get("schema") != "somaforge_acp_checkpoint_v1":
            raise ValueError("ACP requires an ACP checkpoint; dense WBT weights are not interchangeable")
        saved_mode = saved["settings"].get("terrain_observation", "ground_truth")
        current_mode = self.env.acp_settings.get("terrain_observation", "ground_truth")
        if training and saved_mode != current_mode:
            raise ValueError(f"ACP checkpoint used {saved_mode} terrain observations; resume with "
                             f"--terrain-observation {saved_mode} to preserve the observation/replay contract")
        if self.env.acp_planner is not None and saved["planner"] is not None:
            self.env.acp_planner.load_state_dict(saved["planner"])
            if training and self.config.load_optimizer and saved["planner_optimizer"] is not None:
                self.planner_optimizer.load_state_dict(saved["planner_optimizer"])
        if training and saved["settings"]["stage"] == self.env.acp_settings["stage"]:
            self.env.acp_replay.load_state_dict(saved["replay"])
        elif training:
            # Stage transition is a new schedule, while retaining compatible controller weights.
            self.current_learning_iteration = 0
            self._pending_env_state = None

    def load(self, ckpt_path):
        if ckpt_path:
            saved = self._load_checked_checkpoint(ckpt_path).get('acp', {})
            old_version = saved.get('settings', {}).get('interface_version', 1)
            if old_version != self.env.acp_settings.get('interface_version', 1):
                raise ValueError('ACP critic/CSI interface changed; start an aligned run from scratch')
        result = super().load(ckpt_path)
        if ckpt_path:
            self._load_planner(ckpt_path, training=True)
        return result

    def load_for_inference(self, ckpt_path):
        result = super().load_for_inference(ckpt_path)
        if ckpt_path:
            self._load_planner(ckpt_path, training=False)
        return result
