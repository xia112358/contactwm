from types import SimpleNamespace

import pytest
from holosoma.utils.contact_forces import contact_force_view
from holosoma.utils.safe_torch_import import torch


def test_contact_force_view_prefers_managed_link_history() -> None:
    managed = torch.ones(1, 2, 2, 3)
    sensor = SimpleNamespace(
        body_names=["left_wrist_yaw_link", "left_sphere_hand_link"],
        data=SimpleNamespace(net_forces_w_history=torch.full((1, 2, 2, 3), 2.0)),
    )
    simulator = SimpleNamespace(
        contact_link_body_names=["left_sphere_hand_link", "left_sphere_hand_tip_link"],
        contact_link_forces_history=managed,
        contact_sensor=sensor,
        body_names=["left_wrist_yaw_link"],
        contact_forces_history=torch.zeros(1, 2, 1, 3),
    )

    names, history = contact_force_view(simulator)

    assert names == ["left_sphere_hand_link", "left_sphere_hand_tip_link"]
    assert history is managed


def test_contact_force_view_uses_sensor_before_articulation_fallback() -> None:
    sensor_history = torch.ones(1, 2, 2, 3)
    simulator = SimpleNamespace(
        contact_sensor=SimpleNamespace(
            body_names=["left_wrist_yaw_link", "left_sphere_hand_link"],
            data=SimpleNamespace(net_forces_w_history=sensor_history),
        ),
        body_names=["left_wrist_yaw_link"],
        contact_forces_history=torch.zeros(1, 2, 1, 3),
    )

    names, history = contact_force_view(simulator)

    assert names == ["left_wrist_yaw_link", "left_sphere_hand_link"]
    assert history is sensor_history


def test_contact_force_view_rejects_name_shape_mismatch() -> None:
    simulator = SimpleNamespace(
        body_names=["left_wrist_yaw_link"],
        contact_forces_history=torch.zeros(1, 2, 2, 3),
    )

    with pytest.raises(ValueError, match="for 1 bodies"):
        contact_force_view(simulator)
