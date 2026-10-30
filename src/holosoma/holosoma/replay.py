from __future__ import annotations

import argparse

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.motion_matched_config import normalize_motion_matched_config
from holosoma.utils.sim_utils import close_simulation_app, parse_isaaclab_launcher_args, sync_launcher_headless_config
from holosoma.utils.tyro_utils import TYRO_CONIFG


def replay(tyro_config: ExperimentConfig, launcher_args: argparse.Namespace | None = None):
    tyro_config = normalize_motion_matched_config(tyro_config)
    simulation_app = init_sim_imports(tyro_config, launcher_args=launcher_args)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)
    selection_guard = _disable_replay_scene_selection()

    try:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        _set_replay_camera(env)
        while not done:
            env.simulator.sim.step()
            done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
    finally:
        del selection_guard
        close_simulation_app(simulation_app)


def _set_replay_camera(env) -> None:
    target = _get_replay_camera_target(env)
    eye = (target[0] + 3.2, target[1] + 3.8, target[2] + 3.6)
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return

    sim = getattr(simulator, "sim", None)
    if sim is not None and hasattr(sim, "set_camera_view"):
        sim.set_camera_view(eye, target)


def _disable_replay_scene_selection():
    """Keep replay mouse interaction camera-only without affecting other Kit workflows."""
    try:
        import omni.kit.viewport.utility as viewport_utils
        import omni.usd

        selection = omni.usd.get_context().get_selection()
        selection.set_selected_prim_paths([], False)
        viewport_window = viewport_utils.get_active_viewport_window()
        if viewport_window is None:
            return None
        return viewport_utils.disable_selection(viewport_window)
    except (AttributeError, ImportError):
        return None


def _get_replay_camera_target(env) -> tuple[float, float, float]:
    try:
        motion_command = env.command_manager.get_state("motion_command")
        root_pos = motion_command.root_pos_w[0].detach().cpu().tolist()
        return (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]) + 0.15)
    except Exception:
        return (0.3, -0.3, 0.45)


def main() -> None:
    launcher_args = parse_isaaclab_launcher_args("Replay a Holosoma motion.")
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    tyro_cfg = sync_launcher_headless_config(tyro_cfg, launcher_args)
    replay(tyro_cfg, launcher_args=launcher_args)


if __name__ == "__main__":
    main()
