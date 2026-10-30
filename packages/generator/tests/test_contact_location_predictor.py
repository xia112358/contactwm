import torch

from generator.conditioned_pose_predictor import (
    interaction_root_heading_loss,
    interaction_root_progress_loss,
)
from generator.contact_location_predictor import (
    HeightmapContactLocationPredictor,
    RelationalTerrainCrossBlock,
    contact_points_to_heightmap_map,
)
from generator.next_interaction_heightmap import HEIGHTMAP_COLS, HEIGHTMAP_ROWS
from generator.next_interaction_heightmap_v3 import DenseHeightmapInteractionPredictor


def observation(batch: int = 2):
    q = torch.zeros((batch, 36))
    q[:, 2] = 0.8
    q[:, 3] = 1.0
    contact = torch.zeros((batch, 6), dtype=torch.bool)
    contact[:, 0] = True
    point = torch.zeros((batch, 6, 3))
    point[:, 0] = torch.tensor((0.20, -0.10, 0.0))
    contact_map, cell, valid = contact_points_to_heightmap_map(q, contact, point)
    heightmap = torch.zeros((batch, HEIGHTMAP_ROWS, HEIGHTMAP_COLS))
    return q, contact, contact_map, heightmap, cell, valid


def test_actual_contact_is_rasterized_in_root_yaw_observation_coordinates():
    q, contact, contact_map, _, cell, valid = observation(1)
    assert valid[0, 0]
    assert not bool(valid[0, 1:].any())
    assert contact_map[0, 0].sum().item() == 1.0
    assert contact_map[0, 0].flatten().argmax() == cell[0, 0]

    # A 180 degree root yaw maps the same world point behind the robot.
    q[:, 3] = 0.0
    q[:, 6] = 1.0
    points = torch.zeros((1, 6, 3))
    points[0, 0] = torch.tensor((0.20, -0.10, 0.0))
    _, rotated_cell, rotated_valid = contact_points_to_heightmap_map(q, contact, points)
    assert rotated_valid[0, 0]
    assert rotated_cell[0, 0] != cell[0, 0]


def test_model_contract_is_one_embodied_interaction_output():
    model = HeightmapContactLocationPredictor(width=24, layers=1, location_width=8)
    names = tuple(dict(model.named_parameters()))
    assert not any("surface" in name for name in names)
    assert not any("role_head" in name for name in names)
    assert not any("location_query" in name for name in names)
    assert not any("plan_" in name for name in names)
    q, contact, contact_map, heightmap, _, _ = observation()
    prediction = model(q, contact, contact_map, heightmap)
    assert prediction.qpos.shape == (2, 36)
    assert tuple(prediction.__dataclass_fields__) == ("qpos",)


def test_relational_terrain_jointly_queries_root_and_six_contact_parts():
    model = HeightmapContactLocationPredictor(
        width=24, layers=1, location_width=8, relational_terrain=True
    )
    assert isinstance(model.terrain_reasoning[0], RelationalTerrainCrossBlock)
    assert model.height.conv[0].in_channels == 1
    captured = {}

    def capture(_module, args):
        captured["query_position"] = args[2].detach().clone()

    handle = model.terrain_reasoning[0].register_forward_pre_hook(capture)
    q, contact, contact_map, heightmap, _, _ = observation(1)
    prediction = model(q, contact, contact_map, heightmap)
    handle.remove()
    assert prediction.qpos.shape == (1, 36)
    assert captured["query_position"].shape == (1, 7, 3)
    torch.testing.assert_close(
        captured["query_position"][:, 0],
        torch.zeros_like(captured["query_position"][:, 0]),
    )


def test_interaction_state_is_the_single_pose_readout_state():
    model = HeightmapContactLocationPredictor(width=24, layers=1, location_width=8)
    q, contact, contact_map, heightmap, _, _ = observation()
    latent, basis, yaw_quaternion = model.interaction_state(
        q, contact, contact_map, heightmap
    )
    assert latent.shape == (2, 24)
    assert basis.shape == (2, 3, 3)
    assert yaw_quaternion.shape == (2, 4)
    with torch.no_grad():
        expected = model(q, contact, contact_map, heightmap).qpos
        raw = model.pose_head(latent)
        actual = DenseHeightmapInteractionPredictor._decode_pose(
            raw, q, basis, yaw_quaternion
        )
    torch.testing.assert_close(actual, expected)


def test_warm_start_rejects_old_surface_path_by_construction():
    model = HeightmapContactLocationPredictor(width=24, layers=1, location_width=8)
    source = {
        **{key: value.clone() for key, value in model.state_dict().items()},
        "surface_embedding.weight": torch.randn(3, 24),
        "surface_head.weight": torch.randn(2, 24),
    }
    loaded, _ = model.load_observation_encoder(source)
    assert loaded
    assert not any("surface" in key for key in loaded)


def test_forward_has_no_teacher_or_future_interaction_arguments():
    import inspect

    parameters = inspect.signature(HeightmapContactLocationPredictor.forward).parameters
    assert tuple(parameters) == (
        "self", "current_q", "current_contact", "current_contact_map", "heightmap"
    )


def test_residual_joint_output_is_relative_to_current_proprioception():
    model = HeightmapContactLocationPredictor(
        width=24, layers=1, location_width=8, joint_residual_output=True
    )
    with torch.no_grad():
        model.pose_head.weight.zero_()
        model.pose_head.bias.zero_()
        model.pose_head.bias[3] = 1.0
    q, contact, contact_map, heightmap, _, _ = observation()
    q[:, 7:] = torch.linspace(-0.2, 0.2, 29)
    prediction = model(q, contact, contact_map, heightmap)
    torch.testing.assert_close(prediction.qpos[:, 7:], q[:, 7:])


def test_root_only_supervision_has_no_joint_readout_gradient():
    model = HeightmapContactLocationPredictor(width=24, layers=1, location_width=8)
    q, contact, contact_map, heightmap, _, _ = observation(1)
    prediction = model(q, contact, contact_map, heightmap)
    target = prediction.qpos.detach().clone()
    target[:, 0] += 0.20
    yaw = torch.deg2rad(torch.tensor(20.0))
    target[:, 3:7] = torch.tensor((
        torch.cos(yaw / 2.0), 0.0, 0.0, torch.sin(yaw / 2.0)
    ))
    loss = (
        interaction_root_progress_loss(q, prediction.qpos, target)
        + interaction_root_heading_loss(prediction.qpos, target)
    ).mean()
    weight_gradient, bias_gradient = torch.autograd.grad(
        loss, (model.pose_head.weight, model.pose_head.bias)
    )
    assert bool((weight_gradient[:7].abs().sum(-1) > 0).any())
    assert bool((bias_gradient[:7].abs() > 0).any())
    torch.testing.assert_close(weight_gradient[7:], torch.zeros_like(weight_gradient[7:]))
    torch.testing.assert_close(bias_gradient[7:], torch.zeros_like(bias_gradient[7:]))
