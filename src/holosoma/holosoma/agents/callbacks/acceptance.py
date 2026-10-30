"""Lightweight per-motion acceptance callback for deterministic eval."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.config_types.eval_callback import AcceptanceConfig
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.rotations import quat_rotate_inverse


class EvalAcceptanceCallback(RLEvalCallback):
    """Tracks pass/fail status for one deterministic eval episode per env."""

    def __init__(self, config: AcceptanceConfig, training_loop: Any = None):
        super().__init__(config, training_loop)
        self.output_path = self._resolve_output_path(config.output_path)
        self.summary_path = self._resolve_optional_path(
            config.summary_path,
            default=self.output_path.with_name(f"{self.output_path.stem}_summary.json"),
        )
        self.fail_output_path = self._resolve_optional_path(
            config.fail_output_path,
            default=self.output_path.with_name(f"{self.output_path.stem}_fail.csv"),
        )
        self._rows: list[dict[str, Any]] = []
        self._motion_start_idx: list[int] = []
        self._motion_end_idx: list[int] = []
        self._motion_files: list[str] = []
        self._pre_motion_ids: list[int] | None = None
        self._pre_time_steps: list[int] | None = None
        self._initialized = False

    def _resolve_output_path(self, path: str) -> Path:
        output_path = Path(path)
        if not output_path.is_absolute() and self.training_loop is not None and hasattr(self.training_loop, "log_dir"):
            output_path = Path(self.training_loop.log_dir) / output_path
        return output_path

    def _resolve_optional_path(self, path: str, default: Path) -> Path:
        if path:
            return self._resolve_output_path(path)
        return default

    def _get_env(self):
        return self.training_loop._unwrap_env()

    def _get_motion_command(self):
        env = self._get_env()
        if not hasattr(env, "command_manager") or env.command_manager is None:
            raise RuntimeError("EvalAcceptanceCallback requires a command_manager.")
        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is None:
            raise RuntimeError("EvalAcceptanceCallback requires a motion_command state.")
        return motion_command

    @staticmethod
    def _tensor_to_list(value: Any) -> list[int]:
        try:
            return [int(x) for x in value.detach().to("cpu").reshape(-1).tolist()]
        except AttributeError:
            return [int(x) for x in value.reshape(-1).tolist()]

    @staticmethod
    def _bool_tensor_to_list(value: Any, size: int) -> list[bool]:
        if value is None:
            return [False for _ in range(size)]
        try:
            return [bool(x) for x in value.detach().to("cpu").reshape(-1).tolist()]
        except AttributeError:
            return [bool(x) for x in value.reshape(-1).tolist()]

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        motion_command = self._get_motion_command()
        motion = motion_command.motion
        num_envs = int(env.num_envs)
        num_motions = int(motion.num_motions)
        if bool(self.config.require_one_env_per_motion) and num_envs != num_motions:
            raise ValueError(
                "Acceptance eval requires one env per motion: "
                f"num_envs={num_envs}, num_motions={num_motions}."
            )

        self._motion_start_idx = self._tensor_to_list(motion.motion_start_idx)
        self._motion_end_idx = self._tensor_to_list(motion.motion_end_idx)
        self._motion_files = list(getattr(motion, "motion_files", []))
        self._rows = [
            {
                "env_id": env_id,
                "motion_id": "",
                "motion_file": "",
                "status": "unresolved",
                "done_step": "",
                "local_step": -1,
                "motion_len": "",
                "progress": 0.0,
                "terms": "",
            }
            for env_id in range(num_envs)
        ]
        logger.info(
            "EvalAcceptanceCallback: tracking "
            f"{num_envs} envs, {num_motions} motions, output={self.output_path}"
        )

    def on_post_eval_reset(self, actor_state: dict) -> dict:
        self._capture_assignments()
        return actor_state

    def on_pre_eval_env_step(self, actor_state: dict) -> dict:
        motion_command = self._get_motion_command()
        self._pre_motion_ids = self._tensor_to_list(motion_command.motion_ids)
        self._pre_time_steps = self._tensor_to_list(motion_command.time_steps)
        return actor_state

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        if not self._initialized:
            self._capture_assignments()
        if self._pre_motion_ids is None or self._pre_time_steps is None:
            self.on_pre_eval_env_step(actor_state)

        num_envs = len(self._rows)
        extras = actor_state.get("extras") if isinstance(actor_state, dict) else {}
        if not isinstance(extras, dict):
            extras = {}
        dones = self._bool_tensor_to_list(actor_state.get("dones"), num_envs)
        timeouts = self._bool_tensor_to_list(extras.get("time_outs"), num_envs)
        term_dones = extras.get("term_dones")
        motion_end_done = self._term_mask(term_dones, "motion_ends", num_envs)
        step = int(actor_state.get("step", 0))

        for env_id, row in enumerate(self._rows):
            if row["status"] != "unresolved":
                continue
            motion_id = int(self._pre_motion_ids[env_id]) if self._pre_motion_ids is not None else int(row["motion_id"])
            start = self._motion_start_idx[motion_id]
            end = self._motion_end_idx[motion_id]
            motion_len = max(end - start, 1)
            pre_local_step = int(self._pre_time_steps[env_id]) - start if self._pre_time_steps is not None else -1
            pre_local_step = max(0, min(pre_local_step, motion_len - 1))

            if motion_end_done[env_id] or pre_local_step >= motion_len - 1:
                self._mark_done(env_id, "pass", step, motion_len - 1, "")
                continue

            if dones[env_id]:
                terms = self._active_done_terms_for_env(term_dones, env_id)
                if timeouts[env_id] and not terms:
                    terms = ["timeout"]
                local_step = max(0, min(pre_local_step + 1, motion_len - 1))
                status = "pass" if local_step >= motion_len - 1 else "fail"
                details = self._bad_tracking_details(env_id) if status == "fail" and "bad_tracking" in terms else None
                self._mark_done(
                    env_id,
                    status,
                    step,
                    local_step,
                    "" if status == "pass" else "|".join(terms),
                    details=details,
                )
                continue

            self._update_progress(env_id, pre_local_step)

        if bool(self.config.stop_when_complete) and all(row["status"] != "unresolved" for row in self._rows):
            actor_state["stop"] = True
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        self._write_outputs()

    def _capture_assignments(self) -> None:
        motion_command = self._get_motion_command()
        motion_ids = self._tensor_to_list(motion_command.motion_ids)
        time_steps = self._tensor_to_list(motion_command.time_steps)
        seen: set[int] = set()
        for env_id, motion_id in enumerate(motion_ids[: len(self._rows)]):
            start = self._motion_start_idx[motion_id]
            end = self._motion_end_idx[motion_id]
            motion_len = max(end - start, 1)
            motion_file = self._motion_files[motion_id] if motion_id < len(self._motion_files) else ""
            row = self._rows[env_id]
            row.update(
                {
                    "motion_id": motion_id,
                    "motion_file": motion_file,
                    "motion_len": motion_len,
                    "local_step": max(0, min(int(time_steps[env_id]) - start, motion_len - 1)),
                }
            )
            row["progress"] = self._progress(row["local_step"], motion_len)
            seen.add(motion_id)

        if bool(self.config.require_one_env_per_motion) and len(seen) != len(self._rows):
            raise ValueError(
                "Acceptance eval expected a unique motion assignment per env, "
                f"got {len(seen)} unique motion ids for {len(self._rows)} envs."
            )
        self._initialized = True

    def _mark_done(
        self,
        env_id: int,
        status: str,
        done_step: int,
        local_step: int,
        terms: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        row = self._rows[env_id]
        motion_len = int(row["motion_len"])
        if status == "pass":
            local_step = motion_len - 1
        row.update(
            {
                "status": status,
                "done_step": done_step,
                "local_step": local_step,
                "progress": self._progress(local_step, motion_len),
                "terms": terms,
            }
        )
        if details:
            row.update(details)

    def _update_progress(self, env_id: int, local_step: int) -> None:
        row = self._rows[env_id]
        motion_len = int(row["motion_len"])
        if local_step > int(row["local_step"]):
            row["local_step"] = local_step
            row["progress"] = self._progress(local_step, motion_len)

    @staticmethod
    def _progress(local_step: int, motion_len: int) -> float:
        return min(1.0, max(0.0, (int(local_step) + 1) / float(max(int(motion_len), 1))))

    @staticmethod
    def _term_mask(term_dones: Any, name: str, size: int) -> list[bool]:
        if not isinstance(term_dones, dict) or name not in term_dones:
            return [False for _ in range(size)]
        return EvalAcceptanceCallback._bool_tensor_to_list(term_dones[name], size)

    def _active_done_terms_for_env(self, term_dones: Any, env_id: int) -> list[str]:
        if not isinstance(term_dones, dict):
            return []
        active_terms: list[str] = []
        for name, values in term_dones.items():
            try:
                is_active = bool(values[env_id].detach().cpu().item())
            except (AttributeError, IndexError, TypeError):
                is_active = bool(values[env_id])
            if is_active and name != "motion_ends":
                active_terms.append(str(name))
        return active_terms

    @staticmethod
    def _scalar(value: Any) -> float:
        try:
            return float(value.detach().cpu().item())
        except AttributeError:
            return float(value)

    def _bad_tracking_details(self, env_id: int) -> dict[str, Any]:
        env = self._get_env()
        motion_command = self._get_motion_command()
        term = getattr(getattr(env, "termination_manager", None), "_term_instances", {}).get("bad_tracking")
        if term is None:
            return {}

        ref_pos_error = getattr(term, "_last_ref_pos_error", None)
        ref_ori_error = getattr(term, "_last_ref_ori_error", None)
        if ref_pos_error is None or ref_ori_error is None:
            ref_pos_error = (motion_command.ref_pos_w - motion_command.robot_ref_pos_w).norm(dim=1)
            motion_projected_gravity_b = quat_rotate_inverse(
                motion_command.ref_quat_w, gravity_vector(env), w_last=True
            )
            robot_projected_gravity_b = quat_rotate_inverse(
                motion_command.robot_ref_quat_w, gravity_vector(env), w_last=True
            )
            ref_ori_error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()

        body_error_value = 0.0
        body_error_body = ""
        body_threshold = float(getattr(term, "bad_motion_body_pos_threshold", 0.0))
        if getattr(term, "check_motion_body_pos", False):
            cached_body_error = getattr(term, "_last_motion_body_pos_error", None)
            cached_body_index = getattr(term, "_last_motion_body_pos_body_index", None)
            if cached_body_error is not None and cached_body_index is not None:
                body_error_value = self._scalar(cached_body_error[env_id])
                body_i = int(cached_body_index[env_id].detach().cpu().item())
            else:
                body_idx = term.bad_motion_body_pos_body_indexes
                motion_body_pos_error = (
                    motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx]
                ).norm(dim=-1)
                env_body_errors = motion_body_pos_error[env_id]
                max_error, max_index = env_body_errors.max(dim=0)
                body_error_value = self._scalar(max_error)
                body_i = int(max_index.detach().cpu().item())
            names = list(getattr(term, "bad_motion_body_pos_body_names", []))
            body_error_body = str(names[body_i]) if body_i < len(names) else str(body_i)

        ref_pos_value = self._scalar(ref_pos_error[env_id])
        ref_ori_value = self._scalar(ref_ori_error[env_id])
        ref_pos_threshold = float(getattr(term, "bad_ref_pos_threshold", 0.0))
        ref_ori_threshold = float(getattr(term, "bad_ref_ori_threshold", 0.0))
        components = []
        if ref_pos_value > ref_pos_threshold:
            components.append("ref_pos")
        if ref_ori_value > ref_ori_threshold:
            components.append("ref_ori")
        if body_error_value > body_threshold:
            components.append("motion_body_pos")

        return {
            "bad_ref_pos_error": ref_pos_value,
            "bad_ref_pos_threshold": ref_pos_threshold,
            "bad_ref_ori_error": ref_ori_value,
            "bad_ref_ori_threshold": ref_ori_threshold,
            "bad_motion_body_pos_error": body_error_value,
            "bad_motion_body_pos_threshold": body_threshold,
            "bad_motion_body_pos_body": body_error_body,
            "bad_tracking_components": "|".join(components),
        }

    def _write_outputs(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "env_id",
            "motion_id",
            "motion_file",
            "status",
            "done_step",
            "local_step",
            "motion_len",
            "progress",
            "terms",
            "bad_ref_pos_error",
            "bad_ref_pos_threshold",
            "bad_ref_ori_error",
            "bad_ref_ori_threshold",
            "bad_motion_body_pos_error",
            "bad_motion_body_pos_threshold",
            "bad_motion_body_pos_body",
            "bad_tracking_components",
        ]
        sorted_rows = sorted(self._rows, key=lambda row: int(row["motion_id"]) if row["motion_id"] != "" else 10**12)
        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fields} for row in sorted_rows])

        failed_rows = [row for row in sorted_rows if row["status"] == "fail"]
        with self.fail_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fields} for row in failed_rows])

        counts = Counter(str(row["status"]) for row in self._rows)
        total = len(self._rows)
        summary = {
            "total": total,
            "pass": int(counts.get("pass", 0)),
            "fail": int(counts.get("fail", 0)),
            "unresolved": int(counts.get("unresolved", 0)),
            "pass_rate": float(counts.get("pass", 0) / total) if total else 0.0,
            "csv": str(self.output_path),
            "fail_csv": str(self.fail_output_path),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "EvalAcceptanceCallback: "
            f"pass={summary['pass']}/{summary['total']} ({summary['pass_rate']:.4f}), "
            f"fail={summary['fail']}, unresolved={summary['unresolved']}; "
            f"saved {self.output_path}"
        )
