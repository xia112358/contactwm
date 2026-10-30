from __future__ import annotations

from loguru import logger


def prime_overview_camera(env, *, label: str = "Visualizer") -> None:
    """Set one overview pose through the Isaac Lab visualizer abstraction."""
    simulator = _get_simulator(env)
    sim = getattr(simulator, "sim", None)
    set_camera_view = getattr(sim, "set_camera_view", None)
    if not callable(set_camera_view):
        return

    target, eye = get_overview_camera(env)
    try:
        set_camera_view(eye, target)
    except Exception as exc:
        logger.debug(f"{label} camera setup skipped: {exc}")
        return

    # SimulationContext keeps the pose pending for visualizers initialized on
    # reset. Avoid headless render warmup when no visualizer is active.
    if not (getattr(sim, "visualizers", None) or []):
        return

    for _ in range(12):
        try:
            sim.render()
        except Exception as exc:
            logger.debug(f"{label} viewport sim.render() warmup skipped: {exc}")
            break

    try:
        set_camera_view(eye, target)
    except Exception as exc:
        logger.debug(f"{label} final camera setup skipped: {exc}")
    logger.info(f"{label} overview camera primed: eye={eye}, target={target}")


def get_overview_camera(env) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        simulator = _get_simulator(env)
        origins = simulator.env_origins
        origins_cpu = origins.detach().cpu()
        min_xyz = origins_cpu.min(dim=0).values.tolist()
        max_xyz = origins_cpu.max(dim=0).values.tolist()
        center = [(float(min_xyz[i]) + float(max_xyz[i])) * 0.5 for i in range(3)]
        span_x = max(float(max_xyz[0]) - float(min_xyz[0]), 1.0)
        span_y = max(float(max_xyz[1]) - float(min_xyz[1]), 1.0)
        span = max(span_x, span_y, 20.0)

        target = (center[0], center[1], center[2] + 0.5)
        eye = (center[0] + 0.35 * span, center[1] - 0.75 * span, center[2] + 0.85 * span)
        return target, eye
    except Exception:
        target = (0.0, 0.0, 0.5)
        eye = (12.0, -28.0, 24.0)
        return target, eye


def _get_simulator(env):
    return getattr(env, "simulator", None) or getattr(env, "sim", None)
