"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import threading
import time
import traceback
from typing import Any

from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.motion_matched_config import normalize_motion_matched_config
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch


# Re-export the established API for WBT and external launchers.
from somaforge_core.launcher import (
    _add_launcher_arguments, _mark_launcher_intent, fold_preset_tokens,
    parse_isaaclab_launcher_args,
)


def get_launcher_visualizers(launcher_args: argparse.Namespace | None) -> set[str]:
    """Return normalized visualizer names requested through AppLauncher CLI args."""
    if launcher_args is None:
        return set()
    visualizers = getattr(launcher_args, "visualizer", None)
    if not visualizers:
        return set()
    if isinstance(visualizers, str):
        visualizers = [token.strip() for token in visualizers.split(",")]
    return {str(v).strip().lower() for v in visualizers if str(v).strip()}


def configure_newton_cuda_graph_for_visualizer(launcher_args: argparse.Namespace) -> None:
    """Avoid the known Cubric/Newton CUDA-graph conflict in Kit mode.

    Newton's Kit renderer enables Cubric transform propagation. With the
    current Isaac Lab 3.0 stack, combining it with Newton's relaxed CUDA graph
    corrupts the first simulated state. Headless runs do not use Cubric and
    retain the graph-enabled default. An explicit environment setting wins.
    """
    env_name = "HOLOSOMA_NEWTON_USE_CUDA_GRAPH"
    if env_name in os.environ:
        return
    visualizers = get_launcher_visualizers(launcher_args)
    use_kit = "kit" in visualizers or not bool(getattr(launcher_args, "headless", True))
    if use_kit:
        os.environ[env_name] = "0"
        logger.warning(
            "Disabled Newton CUDA graph for Kit visualization because the current "
            "Cubric + relaxed CUDA-graph combination corrupts the first physics step."
        )


def sync_launcher_headless_config(
    tyro_config: ExperimentConfig | RunSimConfig, launcher_args: argparse.Namespace | None
) -> ExperimentConfig | RunSimConfig:
    """Mirror IsaacLab launcher display intent into Holosoma's config."""
    if launcher_args is None:
        return tyro_config

    if getattr(launcher_args, "headless_explicit", False) and getattr(launcher_args, "headless", False):
        # Headless remains the strongest explicit AppLauncher request.
        launcher_args.visualizer = []
        headless = True
    elif getattr(launcher_args, "visualizer_explicit", False):
        visualizer_set = get_launcher_visualizers(launcher_args)
        headless = False if "kit" in visualizer_set else True
    else:
        headless = True
        launcher_args.visualizer = []

    launcher_args.headless = headless
    if tyro_config.training.headless == headless:
        return tyro_config
    return dataclasses.replace(tyro_config, training=dataclasses.replace(tyro_config.training, headless=headless))


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    # IsaacSim imports handled in setup_isaaclab_launcher


def setup_isaaclab_launcher(
    config: ExperimentConfig | RunSimConfig,
    device: str | None = None,
    launcher_args: argparse.Namespace | None = None,
) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher

    unknown_args: list[str] = []
    if launcher_args is None:
        parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
        parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
        parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the output.")
        _add_launcher_arguments(parser)
        args_cli, unknown_args = parser.parse_known_args()
        _mark_launcher_intent(args_cli, sys.argv[1:])
    else:
        args_cli = launcher_args

    # Set values from config — divide by world_size for multi-GPU so each rank's
    # AppLauncher only allocates resources for its share of environments.
    # (The full num_envs is divided again in train_agent.train(), but AppLauncher
    # needs the per-rank count at init time to avoid over-allocating GPU memory.)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args_cli.num_envs = config.training.num_envs // world_size if world_size > 1 else config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.simulator.config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    configure_newton_cuda_graph_for_visualizer(args_cli)
    if getattr(args_cli, "experience", None) in (None, ""):
        headless_experience = os.environ.get("HOLOSOMA_ISAACLAB3_NEWTON_HEADLESS_EXPERIENCE")
        kit_experience = os.environ.get("HOLOSOMA_ISAACLAB3_NEWTON_KIT_EXPERIENCE")
        visualizer_set = get_launcher_visualizers(args_cli)
        use_kit_experience = "kit" in visualizer_set or not bool(getattr(args_cli, "headless", True))
        default_experience = kit_experience if use_kit_experience else headless_experience
        if default_experience:
            args_cli.experience = default_experience
    # GA Lab packages are Python packages, not Kit extensions.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        args_cli.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        args_cli.distributed = True
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    # Check if video recording is enabled and add --enable_cameras flag
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    if video_enabled:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    logger.info(f"IsaacSim args_cli: {args_cli}")
    logger.info(f"IsaacSim unknown_args: {unknown_args}")
    if launcher_args is None:
        sys.argv = [sys.argv[0]] + unknown_args

    return simulation_app


def setup_keyboard_listener(env) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key, env):
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env):
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig,
    device: str | None = None,
    launcher_args: argparse.Namespace | None = None,
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.
    launcher_args : argparse.Namespace | None, optional
        Pre-parsed IsaacLab AppLauncher arguments.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    if isinstance(config, ExperimentConfig):
        config = normalize_motion_matched_config(config)

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() in (SimulatorType.ISAACSIM, SimulatorType.ISAACLAB3_NEWTON):
        simulation_app = setup_isaaclab_launcher(config, device, launcher_args=launcher_args)
        if get_simulator_type() == SimulatorType.ISAACLAB3_NEWTON:
            from somaforge_core.newton_runtime_compat import install as install_newton_runtime_compat
            install_newton_runtime_compat()

    # Set random seed if specified (only for ExperimentConfig)
    if isinstance(config, ExperimentConfig) and config.training.seed is not None:
        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        logger.info(f"Seed: {config.training.seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Create FullSimConfig from RunSimConfig
        # Extract SimulatorInitConfig from SimulatorConfig
        full_config = FullSimConfig(
            simulator=config.simulator.config,  # Extract .config from SimulatorConfig
            robot=config.robot,
            training=config.training,
            logger=config.logger,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env
        class EnvProxy:
            def __init__(self, device):
                self.num_envs = 1
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator):
                self.sim = simulator
                self.simulator = simulator
                self.headless = config.training.headless

            def reset(self):
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self):
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_manager = TerrainManager(config.terrain, env=EnvProxy(device), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env, device, simulation_app


def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() in (SimulatorType.ISAACSIM, SimulatorType.ISAACLAB3_NEWTON):
        logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                logger.info("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(self, config: RunSimConfig, env: Any, device: str, simulation_app: Any):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        self.cleanup()

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Need to manually set headless since it's in training config currently
        self.simulator.set_headless(False)

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Create environments (need to provide required parameters)
        # Create env_origins (single environment at origin)
        env_origins = torch.zeros(1, 3, device=self.device)

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")

        # Step 5.5: Initialize episode (positions virtual gantry, etc.)
        self.simulator.on_episode_start(env_id=0)
        logger.debug("simulator.on_episode_start() completed")

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        logger.info("Simulator initialized")

        # Step 7: Toggle start recording if enabled
        if self.simulator.video_recorder and self.simulator.video_recorder.enabled:
            # arbitrary episode ID given this is sim2sim, we may want to
            # actually support toggling recording and with better filenames too
            self.simulator.video_recorder.start_recording(episode_id=0)

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and FPS logging.

        Manages the complete simulation loop including rate limiting,
        viewer synchronization, FPS logging, and error handling.
        """
        # Setup rate limiting
        sim_frequency = self.config.simulator.config.sim.fps
        rate_limiter = RateLimiter(sim_frequency)

        # Calculate viewer sync frequency
        viewer_steps = self._calculate_viewer_steps()

        logger.info(f"Simulation rate: {sim_frequency} Hz ({1.0 / sim_frequency * 1000:.2f} ms)")
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")
        logger.info("Starting direct simulation loop...")
        logger.info("Press Ctrl+C to stop simulation")

        # Determine refresh strategy based on simulator type
        # IsaacGym/IsaacSim: need pre-step to refresh tensors to sync simulator state
        # MuJoCo: no pre-step refresh needed because we are NOT running an envs/tasks requiring
        #         those tensors e.g, _rigid_body_rot, _rigid_body_vel, etc.
        simulator_type = get_simulator_type()
        pre_step_refresh = self.simulator.refresh_sim_tensors

        # Direct simulation loop (like holosoma_inference's simulation_thread)
        step_count = 0
        start_time = time.time()
        fps_start_time = start_time

        while True:
            try:
                # Refresh tensors if needed (no-op for MuJoCo)
                pre_step_refresh()

                # Direct simulator step - this triggers bridge.step() inside simulate_at_each_physics_step()
                self.simulator.simulate_at_each_physics_step()

                # Update viewer at display rate
                if step_count % viewer_steps == 0:
                    self.simulator.render()

                # Periodic FPS logging (every 1000 steps)
                if step_count > 0 and step_count % 1000 == 0:
                    fps_start_time = self._log_fps(step_count, fps_start_time)

                step_count += 1
                rate_limiter.sleep()

            except KeyboardInterrupt:  # noqa: PERF203
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during simulation step {step_count}: {e}")
                traceback.print_exc()
                break

        # Final statistics
        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        # Cleanup environment
        if hasattr(self.env, "close"):
            self.env.close()

        if self.simulator.video_recorder:
            self.simulator.video_recorder.cleanup()

        # Cleanup simulation app
        if self.simulation_app:
            close_simulation_app(self.simulation_app)

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        """Log FPS statistics for simulation performance monitoring.

        Parameters
        ----------
        step_count : int
            Current step count.
        fps_start_time : float
            Start time for FPS measurement.

        Returns
        -------
        float
            New start time for next FPS measurement.
        """
        elapsed = time.time() - fps_start_time
        fps = 1000 / elapsed
        logger.info(f"Simulation FPS: {fps:.1f}")
        return time.time()
