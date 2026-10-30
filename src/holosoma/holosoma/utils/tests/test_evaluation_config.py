from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import tyro
from holosoma.config_types.eval_callback import EvaluationConfig
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_tracking_precision_curriculum_10k
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination
from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import configure_newton_cuda_graph_for_visualizer, sync_launcher_headless_config
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma.utils.viewport_camera import prime_overview_camera


def test_evaluation_help_without_checkpoint_never_loads_a_model(monkeypatch, capsys) -> None:
    from holosoma import eval_agent

    monkeypatch.setattr("sys.argv", ["eval_agent", "--help"])
    monkeypatch.setattr(eval_agent, "init_eval_logging", lambda: None)
    monkeypatch.setattr(eval_agent, "parse_isaaclab_launcher_args", lambda description: argparse.Namespace())
    with mock.patch.object(eval_agent, "load_saved_experiment_config") as load:
        with pytest.raises(SystemExit) as result:
            eval_agent.main()
    assert result.value.code == 0
    load.assert_not_called()
    assert "--checkpoint" in capsys.readouterr().out


def test_evaluation_cli_owns_runtime_settings() -> None:
    evaluation, remaining = tyro.cli(
        EvaluationConfig,
        args=[
            "--num-envs",
            "8",
            "--max-steps",
            "42",
            "--export-onnx",
            "True",
            "--video.enabled",
            "True",
            "--training.seed",
            "7",
        ],
        return_unknown_args=True,
        add_help=False,
        config=TYRO_CONIFG,
    )

    assert evaluation.num_envs == 8
    assert evaluation.max_steps == 42
    assert evaluation.export_onnx is True
    assert evaluation.video.enabled is True
    assert remaining == ["--training.seed", "7"]


def test_experiment_override_cli_does_not_expose_duplicate_evaluation_settings() -> None:
    with pytest.raises(SystemExit):
        tyro.cli(
            ExperimentConfig,
            args=["--evaluation.num-envs", "8"],
            add_help=False,
            config=TYRO_CONIFG,
        )


def test_evaluation_settings_win_during_final_config_composition() -> None:
    config = ExperimentConfig()
    advanced = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, num_envs=99),
        terrain=dataclasses.replace(
            config.terrain,
            terrain_term=dataclasses.replace(
                config.terrain.terrain_term,
                spawn=dataclasses.replace(
                    config.terrain.terrain_term.spawn,
                    randomize_tiles=True,
                    xy_offset_range=3.0,
                ),
            ),
        ),
    )
    evaluation = EvaluationConfig(num_envs=7, randomize_tiles=False, xy_offset_range=0.25)

    resolved = advanced.get_eval_config(evaluation)

    assert resolved.training.num_envs == 7
    assert resolved.terrain.terrain_term.spawn.randomize_tiles is False
    assert resolved.terrain.terrain_term.spawn.xy_offset_range == 0.25


def test_eval_config_removes_training_only_runtime_state() -> None:
    config = ExperimentConfig()
    algo_config = dataclasses.replace(
        config.algo.config,
        load_optimizer=True,
        actor_finetune_mode="full_actor",
        anchor_kl_checkpoint="anchor.pt",
        anchor_kl_coef=0.2,
        exec_consistency_coef=0.01,
    )
    bad_tracking = g1_29dof_wbt_termination
    if bad_tracking is not None and "bad_tracking" in bad_tracking.terms:
        term = bad_tracking.terms["bad_tracking"]
        bad_tracking = dataclasses.replace(
            bad_tracking,
            terms={
                **bad_tracking.terms,
                "bad_tracking": dataclasses.replace(
                    term,
                    params={**term.params, "probe_fixed_qualification_boundary": True},
                ),
            },
        )
    training = dataclasses.replace(
        config,
        algo=dataclasses.replace(config.algo, config=algo_config),
        curriculum=g1_29dof_wbt_tracking_precision_curriculum_10k,
        termination=bad_tracking,
    )

    resolved = training.get_eval_config(EvaluationConfig())

    assert resolved.curriculum is not None
    assert resolved.curriculum.setup_terms == {}
    assert resolved.curriculum.reset_terms == {}
    assert resolved.curriculum.step_terms == {}
    assert resolved.algo.config.load_optimizer is False
    assert resolved.algo.config.actor_finetune_mode == "none"
    assert resolved.algo.config.anchor_kl_checkpoint is None
    assert resolved.algo.config.anchor_kl_coef == 0.0
    assert resolved.algo.config.exec_consistency_coef == 0.0
    if resolved.termination is not None and "bad_tracking" in resolved.termination.terms:
        assert "probe_fixed_qualification_boundary" not in resolved.termination.terms["bad_tracking"].params


@pytest.mark.parametrize(
    ("visualizers", "explicit", "expected_headless"),
    [
        (None, False, True),
        (["none"], True, True),
        (["newton"], True, True),
        (["kit"], True, False),
        (["kit", "newton"], True, False),
    ],
)
def test_visualizer_selection_is_the_display_source_of_truth(
    visualizers: list[str] | None, explicit: bool, expected_headless: bool
) -> None:
    launcher_args = argparse.Namespace(
        headless=False,
        headless_explicit=False,
        visualizer=visualizers,
        visualizer_explicit=explicit,
    )

    resolved = sync_launcher_headless_config(ExperimentConfig(), launcher_args)

    assert resolved.training.headless is expected_headless
    assert launcher_args.headless is expected_headless


def test_kit_visualizer_disables_newton_cuda_graph_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HOLOSOMA_NEWTON_USE_CUDA_GRAPH", raising=False)
    launcher_args = argparse.Namespace(visualizer=["kit"], headless=False)

    configure_newton_cuda_graph_for_visualizer(launcher_args)

    assert os.environ["HOLOSOMA_NEWTON_USE_CUDA_GRAPH"] == "0"


def test_headless_keeps_newton_cuda_graph_default_unset(monkeypatch) -> None:
    monkeypatch.delenv("HOLOSOMA_NEWTON_USE_CUDA_GRAPH", raising=False)
    launcher_args = argparse.Namespace(visualizer=[], headless=True)

    configure_newton_cuda_graph_for_visualizer(launcher_args)

    assert "HOLOSOMA_NEWTON_USE_CUDA_GRAPH" not in os.environ


def test_explicit_newton_cuda_graph_override_wins_in_kit(monkeypatch) -> None:
    monkeypatch.setenv("HOLOSOMA_NEWTON_USE_CUDA_GRAPH", "1")
    launcher_args = argparse.Namespace(visualizer=["kit"], headless=False)

    configure_newton_cuda_graph_for_visualizer(launcher_args)

    assert os.environ["HOLOSOMA_NEWTON_USE_CUDA_GRAPH"] == "1"


class _FakeSimulationContext:
    def __init__(self, *, active_visualizer: bool) -> None:
        self.visualizers = [object()] if active_visualizer else []
        self.camera_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        self.render_count = 0

    def set_camera_view(self, eye, target) -> None:
        self.camera_calls.append((tuple(eye), tuple(target)))

    def render(self) -> None:
        self.render_count += 1


@pytest.mark.parametrize(("active_visualizer", "expected_calls", "expected_renders"), [(False, 1, 0), (True, 2, 12)])
def test_overview_camera_uses_simulation_context_only(
    active_visualizer: bool, expected_calls: int, expected_renders: int
) -> None:
    sim = _FakeSimulationContext(active_visualizer=active_visualizer)
    simulator = SimpleNamespace(sim=sim, env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]]))
    env = SimpleNamespace(simulator=simulator, headless=True)

    prime_overview_camera(env)

    assert len(sim.camera_calls) == expected_calls
    assert sim.render_count == expected_renders


def test_eval_staging_is_cleaned_when_setup_fails(tmp_path) -> None:
    staging = tmp_path / "eval-staging"
    config = ExperimentConfig()
    evaluation = EvaluationConfig(max_steps=1)

    with (
        mock.patch("holosoma.eval_agent.get_eval_log_dir", return_value=staging),
        mock.patch("holosoma.eval_agent.setup_simulation_environment", side_effect=RuntimeError("setup failed")),
        pytest.raises(RuntimeError, match="setup failed"),
    ):
        run_eval_with_tyro(
            config.get_eval_config(evaluation),
            CheckpointConfig(checkpoint="unused.pt"),
            config,
            None,
            evaluation=evaluation,
        )

    assert staging.is_dir()


def test_eval_passes_launcher_device_to_environment_setup(tmp_path) -> None:
    staging = tmp_path / "eval-staging"
    config = ExperimentConfig()
    evaluation = EvaluationConfig(max_steps=1)
    launcher_args = argparse.Namespace(device="cpu")

    with (
        mock.patch("holosoma.eval_agent.get_eval_log_dir", return_value=staging),
        mock.patch(
            "holosoma.eval_agent.setup_simulation_environment",
            side_effect=RuntimeError("setup failed"),
        ) as setup,
        pytest.raises(RuntimeError, match="setup failed"),
    ):
        run_eval_with_tyro(
            config.get_eval_config(evaluation),
            CheckpointConfig(checkpoint="unused.pt"),
            config,
            None,
            evaluation=evaluation,
            launcher_args=launcher_args,
        )

    setup.assert_called_once_with(
        config.get_eval_config(evaluation),
        device="cpu",
        launcher_args=launcher_args,
    )


def test_eval_export_keeps_only_explicit_output(tmp_path) -> None:
    staging = tmp_path / "eval-output"
    checkpoint = tmp_path / "model_5000.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = ExperimentConfig()
    evaluation = EvaluationConfig(max_steps=1, export_onnx=True)
    writer = SimpleNamespace(close=mock.Mock())

    class FakeAlgo:
        def __init__(self, **kwargs) -> None:
            self.config = SimpleNamespace(eval_callbacks=None)
            self.eval_callbacks = []
            self.writer = writer

        def setup(self) -> None:
            pass

        def attach_checkpoint_metadata(self, saved_config, saved_wandb_path) -> None:
            pass

        def load_for_inference(self, checkpoint_path: str) -> None:
            assert checkpoint_path == str(checkpoint)

        def export(self, onnx_file_path: str) -> None:
            Path(onnx_file_path).write_bytes(b"onnx")

        def evaluate_policy(self, max_eval_steps: int | None) -> None:
            assert max_eval_steps == 1

    env = SimpleNamespace()
    simulation_app = object()
    with (
        mock.patch("holosoma.eval_agent.get_eval_log_dir", return_value=staging),
        mock.patch(
            "holosoma.eval_agent.setup_simulation_environment",
            return_value=(env, "cpu", simulation_app),
        ),
        mock.patch("holosoma.eval_agent.prime_overview_camera"),
        mock.patch("holosoma.eval_agent.load_checkpoint", return_value=checkpoint) as load_checkpoint,
        mock.patch("holosoma.eval_agent.get_class", return_value=FakeAlgo),
        mock.patch("holosoma.eval_agent.close_simulation_app") as close_simulation_app,
    ):
        run_eval_with_tyro(
            config.get_eval_config(evaluation),
            CheckpointConfig(checkpoint=str(checkpoint)),
            config,
            None,
            evaluation=evaluation,
        )

    load_checkpoint.assert_called_once_with(str(checkpoint), None)
    assert (staging / "exported" / "model_5000.onnx").read_bytes() == b"onnx"
    assert sorted(path.relative_to(staging) for path in staging.rglob("*")) == [
        Path("exported"),
        Path("exported/model_5000.onnx"),
    ]
    writer.close.assert_called_once_with()
    close_simulation_app.assert_called_once_with(simulation_app)
