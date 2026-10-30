from __future__ import annotations

from typing import Any


def contact_force_view(simulator: Any) -> tuple[list[str], Any]:
    """Return the highest-resolution body names and force history available."""

    link_names = getattr(simulator, "contact_link_body_names", None)
    link_history = getattr(simulator, "contact_link_forces_history", None)
    if link_names is not None and link_history is not None:
        return _validated_contact_force_view(link_names, link_history, source="contact link")

    contact_sensor = getattr(simulator, "contact_sensor", None)
    sensor_data = getattr(contact_sensor, "data", None)
    sensor_names = getattr(contact_sensor, "body_names", None)
    sensor_history = getattr(sensor_data, "net_forces_w_history", None)
    if sensor_names is not None and sensor_history is not None:
        return _validated_contact_force_view(sensor_names, sensor_history, source="contact sensor")

    return _validated_contact_force_view(
        simulator.body_names,
        simulator.contact_forces_history,
        source="simulator contact",
    )


def _validated_contact_force_view(body_names: Any, history: Any, *, source: str) -> tuple[list[str], Any]:
    names = list(body_names)
    if history.ndim != 4 or history.shape[2] != len(names) or history.shape[-1] != 3:
        raise ValueError(
            f"{source} force history must have shape [num_envs, history, num_bodies, 3], "
            f"got {tuple(history.shape)} for {len(names)} bodies"
        )
    return names, history
