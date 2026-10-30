from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.eval_callback import EvaluationConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_eval_log_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.motion_matched_config import normalize_motion_matched_config
from holosoma.utils.sim_utils import (
    close_simulation_app,
    parse_isaaclab_launcher_args,
    setup_simulation_environment,
    sync_launcher_headless_config,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma.utils.viewport_camera import prime_overview_camera


def _acceptance_repeat_count(evaluation: EvaluationConfig) -> int:
    acceptance_cfg = evaluation.acceptance.config
    return max(1, int(getattr(acceptance_cfg, "repeats", 1)))


def _with_repeat_suffix(path_str: str, rep_index: int) -> str:
    if not path_str:
        return path_str
    path = Path(path_str)
    stem = path.stem
    suffix = path.suffix
    replacements = (
        ("_acceptance_summary", f"_rep{rep_index}_acceptance_summary"),
        ("_acceptance", f"_rep{rep_index}_acceptance"),
        ("_fail", f"_rep{rep_index}_fail"),
    )
    for old, new in replacements:
        if stem.endswith(old):
            return str(path.with_name(f"{stem[: -len(old)]}{new}{suffix}"))
    return str(path.with_name(f"{stem}_rep{rep_index}{suffix}"))


def _eval_callbacks_for_repeat(
    evaluation: EvaluationConfig,
    rep_index: int,
    repeat_count: int,
) -> dict:
    if repeat_count <= 1:
        return evaluation.collect_active_callbacks()

    acceptance_cb_cfg = evaluation.acceptance
    acceptance_cfg = acceptance_cb_cfg.config
    rep_acceptance_cfg = dataclasses.replace(
        acceptance_cfg,
        output_path=_with_repeat_suffix(acceptance_cfg.output_path, rep_index),
        summary_path=_with_repeat_suffix(acceptance_cfg.summary_path, rep_index),
        fail_output_path=_with_repeat_suffix(acceptance_cfg.fail_output_path, rep_index),
    )
    rep_acceptance_cb_cfg = dataclasses.replace(acceptance_cb_cfg, config=rep_acceptance_cfg)
    repeated_evaluation = dataclasses.replace(evaluation, acceptance=rep_acceptance_cb_cfg)
    return repeated_evaluation.collect_active_callbacks()


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    evaluation: EvaluationConfig,
    launcher_args: argparse.Namespace | None = None,
):
    tyro_config = normalize_motion_matched_config(tyro_config)
    eval_log_dir = get_eval_log_dir(tyro_config.logger, tyro_config.training, get_timestamp())
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using eval staging directory {eval_log_dir}")
    algo: BaseAlgo | None = None
    simulation_app = None
    try:
        requested_device = getattr(launcher_args, "device", None) if launcher_args is not None else None
        env, device, simulation_app = setup_simulation_environment(
            tyro_config,
            device=requested_device,
            launcher_args=launcher_args,
        )
        prime_overview_camera(env, label="Eval")

        assert checkpoint_cfg.checkpoint is not None
        checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, None)
        checkpoint_path = str(checkpoint)

        algo_class = get_class(tyro_config.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=tyro_config.algo.config,
            log_dir=str(eval_log_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
        algo.load_for_inference(checkpoint_path)

        if evaluation.export_onnx:
            exported_policy_dir = eval_log_dir / "exported"
            exported_policy_dir.mkdir(parents=True, exist_ok=True)
            exported_path = exported_policy_dir / f"{Path(checkpoint_path).stem}.onnx"
            if not hasattr(algo, "export"):
                raise AttributeError(
                    f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
                )
            algo.export(onnx_file_path=str(exported_path))  # type: ignore[attr-defined]
            logger.info(f"Exported policy as onnx to: {exported_path}")

        repeat_count = _acceptance_repeat_count(evaluation)
        for rep_index in range(1, repeat_count + 1):
            cb_configs = _eval_callbacks_for_repeat(evaluation, rep_index, repeat_count)
            object.__setattr__(algo.config, "eval_callbacks", cb_configs or None)
            if hasattr(algo, "eval_callbacks"):
                algo.eval_callbacks.clear()
            logger.info(f"Starting eval repeat {rep_index}/{repeat_count}")
            prime_overview_camera(env, label="Eval")
            algo.evaluate_policy(max_eval_steps=evaluation.max_steps)
    finally:
        writer = getattr(algo, "writer", None)
        try:
            if writer is not None and hasattr(writer, "close"):
                writer.close()
        finally:
            if simulation_app is not None:
                close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    launcher_args = parse_isaaclab_launcher_args("Evaluate a Holosoma agent.")
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    if checkpoint_cfg.checkpoint is None and any(arg in {"--help", "-h"} for arg in remaining_args):
        tyro.cli(CheckpointConfig, args=remaining_args,
            description="Select a checkpoint to evaluate. Pass --checkpoint /path/model.pt --help "
                        "to inspect the saved experiment and evaluation overrides.")
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    evaluation, remaining_args = tyro.cli(
        EvaluationConfig,
        default=saved_cfg.evaluation,
        return_unknown_args=True,
        add_help=False,
        args=remaining_args,
        config=TYRO_CONIFG,
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=saved_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    overwritten_tyro_config = overwritten_tyro_config.get_eval_config(evaluation)
    overwritten_tyro_config = sync_launcher_headless_config(overwritten_tyro_config, launcher_args)
    from somaforge_core.robot_assets import validate_g1_robot_config

    validate_g1_robot_config(overwritten_tyro_config.robot)

    run_eval_with_tyro(
        overwritten_tyro_config,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        evaluation=evaluation,
        launcher_args=launcher_args,
    )


if __name__ == "__main__":
    main()
