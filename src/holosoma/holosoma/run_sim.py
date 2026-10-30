#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

import dataclasses
import sys
import traceback

import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import (
    DirectSimulation,
    parse_isaaclab_launcher_args,
    setup_simulation_environment,
    sync_launcher_headless_config,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma.utils.viewport_camera import prime_overview_camera


def run_simulation(config: RunSimConfig, launcher_args=None):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    config = dataclasses.replace(config, device=config.device)

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(
            config, device=config.device, launcher_args=launcher_args
        )

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(config, env, device, simulation_app) as sim:
            prime_overview_camera(env, label="RunSim")
            sim.run()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function using tyro configuration with compositional subcommands."""
    launcher_args = parse_isaaclab_launcher_args("Run Holosoma direct simulation.")

    # Initialize logging
    init_eval_logging()

    logger.info("Holosoma Direct Simulation Runner")
    logger.info("Compositional configuration via subcommands (like eval_agent.py)")

    # Parse configuration with tyro - same pattern as ExperimentConfig
    config = tyro.cli(
        RunSimConfig,
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:isaaclab3-newton robot:g1_29dof terrain:terrain_motion_matched",
        config=TYRO_CONIFG,
    )
    config = sync_launcher_headless_config(config, launcher_args)

    # Run simulation directly with parsed config
    run_simulation(config, launcher_args=launcher_args)


if __name__ == "__main__":
    main()
