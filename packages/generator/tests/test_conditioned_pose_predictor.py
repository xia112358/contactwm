import torch

import contact_solver.newton_plan_realization as realization_module
from generator.conditioned_pose_predictor import (
    balanced_squared_loss_weights,
    canonical_contact_first_pose_loss,
    ConditionedHeightmapPosePredictor,
    ConditionedPosePrediction,
    conditioned_pose_objective,
    interaction_root_contact_frame_loss,
    interaction_root_heading_loss,
    interaction_root_progress_loss,
    topology_gated_pose_loss,
)
from generator.next_interaction_heightmap import HEIGHTMAP_COLS, HEIGHTMAP_ROWS


def inputs(batch: int = 2):
    q = torch.zeros((batch, 36))
    q[:, 2] = 0.8
    q[:, 3] = 1.0
    contact = torch.zeros((batch, 6), dtype=torch.bool)
    surface = torch.full((batch, 6), -1, dtype=torch.long)
    anchor = torch.zeros((batch, 6, 3))
    heightmap = torch.zeros((batch, HEIGHTMAP_ROWS, HEIGHTMAP_COLS))
    return q, contact, anchor, surface, heightmap


def test_no_binding_pose_depends_on_the_explicit_plan_only() -> None:
    torch.manual_seed(7)
    model = ConditionedHeightmapPosePredictor(width=48, layers=1)
    q, current, anchor, current_surface, heightmap = inputs()
    planned = torch.zeros((2, 6), dtype=torch.bool)
    planned_surface = torch.full((2, 6), -1, dtype=torch.long)
    planned[1, 3] = True
    planned_surface[1, 3] = 1
    output = model(
        q, current, anchor, current_surface, heightmap, planned, planned_surface
    )
    assert output.qpos.shape == (2, 36)
    assert not torch.equal(output.qpos[0], output.qpos[1])
    assert not any("binding" in name for name, _ in model.named_parameters())
    assert not hasattr(output, "role_logits")
    assert not hasattr(output, "surface_logits")


def test_active_plan_requires_an_actual_surface() -> None:
    model = ConditionedHeightmapPosePredictor(width=48, layers=1)
    q, current, anchor, current_surface, heightmap = inputs(1)
    planned = torch.tensor([[True, False, False, False, False, False]])
    planned_surface = torch.full((1, 6), -1, dtype=torch.long)
    try:
        model(q, current, anchor, current_surface, heightmap, planned, planned_surface)
    except ValueError as error:
        assert "active planned contacts" in str(error)
    else:
        raise AssertionError("invalid contact plan was accepted")


def test_missing_new_contact_has_first_class_establishment_loss(monkeypatch) -> None:
    """A limb outside narrow phase must not silently lose its contact loss."""

    model = ConditionedHeightmapPosePredictor(width=48, layers=1)
    q, _, _, _, _ = inputs(1)
    with torch.no_grad():
        position, _ = model.fk(q)
    planned = torch.zeros((1, 6), dtype=torch.bool)
    planned[:, 2] = True
    surface = torch.full((1, 6), -1, dtype=torch.long)
    surface[:, 2] = 1
    missing = planned.clone()

    def fake_realization(_model, qpos, *_args, **_kwargs):
        zero = qpos[:, 0] * 0.0
        return zero, {
            "newton_penetration_cm": zero,
            "newton_collision_loss": zero,
            "newton_missing_intended_mask": missing,
            "newton_plan_realization_loss": zero,
            "newton_unwanted_contact_loss": zero,
            "newton_plan_exact": zero,
            "newton_generation_valid": zero,
            "newton_intended_contacts": torch.ones_like(zero),
            "newton_realized_intended_contacts": zero,
            "newton_unrealized_intended_contacts": torch.ones_like(zero),
            "newton_missing_intended_pairs": torch.ones_like(zero),
            "newton_unwanted_contact_parts": zero,
            "newton_invalid_fullbody_witnesses": zero,
            "newton_configured_includemargin": torch.full_like(zero, 0.02),
        }

    monkeypatch.setattr(realization_module, "newton_plan_realization", fake_realization)
    target = {
        "current_q": q.clone(),
        "q": q.clone(),
        "planned_contact": planned,
        "planned_surface": surface,
        "material_local": torch.zeros((1, 6, 3)),
        "anchor": position[:, 1:7].detach().clone(),
        "body_position": position.detach().clone(),
        "newton_world_origin": torch.zeros((1, 3)),
        "newton_world_basis": torch.eye(3)[None],
        "newton_model_fingerprint": torch.zeros((1, 32), dtype=torch.uint8),
    }
    scene = {
        "box_center": torch.tensor(((2.0, 0.0, 0.5),)),
        "box_rotation": torch.eye(3)[None],
        "box_half_extents": torch.tensor(((0.5, 0.5, 0.5),)),
        "ground_height": torch.zeros(1),
    }
    loss, metrics = conditioned_pose_objective(
        model, ConditionedPosePrediction(q), target, scene
    )
    assert metrics["missing_pair_approach_loss"].item() > 0.0
    torch.testing.assert_close(
        metrics["contact_establishment_loss"],
        metrics["missing_pair_approach_loss"],
    )
    torch.testing.assert_close(loss, metrics["contact_establishment_loss"])


def test_topology_gated_pose_loss_blocks_wrong_plan_task_gradients() -> None:
    imitation = torch.tensor([2.0, 3.0], requires_grad=True)
    realization = torch.tensor([5.0, 7.0], requires_grad=True)
    establishment = torch.tensor([11.0, 13.0], requires_grad=True)
    safety = torch.tensor([17.0, 19.0], requires_grad=True)
    loss = topology_gated_pose_loss(
        {
            "imitation_loss": imitation,
            "own_plan_realization_loss": realization,
            "contact_establishment_loss": establishment,
            "safety_loss": safety,
        },
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(loss, torch.tensor([35.0, 19.0]))
    loss.sum().backward()
    torch.testing.assert_close(imitation.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(realization.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(establishment.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(safety.grad, torch.ones(2))


def test_canonical_contact_first_loss_keeps_safety_on_wrong_plan() -> None:
    values = {
        "imitation_loss": torch.tensor([10.0, 20.0], requires_grad=True),
        "interaction_pose_prior_loss": torch.tensor([12.0, 14.0], requires_grad=True),
        "nominal_taskspace_teacher_loss": torch.tensor([0.0, 0.0], requires_grad=True),
        "interaction_location_loss": torch.tensor([1.0, 1.5], requires_grad=True),
        "newton_plan_realization_loss": torch.tensor([2.0, 3.0], requires_grad=True),
        "newton_unwanted_contact_loss": torch.tensor([4.0, 5.0], requires_grad=True),
        "contact_establishment_loss": torch.tensor([6.0, 7.0], requires_grad=True),
        "newton_collision_loss": torch.tensor([8.0, 9.0], requires_grad=True),
        "joint_limit_loss": torch.tensor([1.0, 2.0], requires_grad=True),
    }
    loss = canonical_contact_first_pose_loss(
        values, torch.tensor([True, False]), imitation_weight=0.1
    )
    torch.testing.assert_close(loss, torch.tensor([23.0, 11.0]))
    loss.sum().backward()
    torch.testing.assert_close(values["imitation_loss"].grad, torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(
        values["interaction_pose_prior_loss"].grad, torch.zeros(2)
    )
    torch.testing.assert_close(
        values["newton_plan_realization_loss"].grad, torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        values["interaction_location_loss"].grad, torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(values["newton_collision_loss"].grad, torch.ones(2))
    torch.testing.assert_close(values["joint_limit_loss"].grad, torch.ones(2))


def test_root_progress_is_local_coarse_and_rejects_reverse_motion() -> None:
    current = torch.zeros((1, 36))
    current[:, 3] = 1.0
    target = current.clone()
    target[:, 0] = 0.20
    close = current.clone()
    close[:, 0] = 0.17
    reverse = current.clone()
    reverse[:, 0] = -0.10
    torch.testing.assert_close(
        interaction_root_progress_loss(current, close, target), torch.zeros(1)
    )
    assert interaction_root_progress_loss(current, reverse, target).item() > 1.0

    offset = torch.tensor((3.0, -2.0, 0.7))
    shifted_current = current.clone()
    shifted_target = target.clone()
    shifted_close = close.clone()
    shifted_current[:, :3] += offset
    shifted_target[:, :3] += offset
    shifted_close[:, :3] += offset
    torch.testing.assert_close(
        interaction_root_progress_loss(shifted_current, shifted_close, shifted_target),
        torch.zeros(1),
    )


def test_root_heading_is_periodic_and_has_a_small_dead_zone() -> None:
    def pose(yaw_degrees: float) -> torch.Tensor:
        q = torch.zeros((1, 36))
        yaw = torch.deg2rad(torch.tensor(yaw_degrees))
        q[:, 3] = torch.cos(yaw / 2.0)
        q[:, 6] = torch.sin(yaw / 2.0)
        return q

    target = pose(179.0)
    across_wrap = pose(-179.0)
    outside = pose(-166.0)
    torch.testing.assert_close(
        interaction_root_heading_loss(across_wrap, target), torch.zeros(1),
        atol=1.0e-6, rtol=0.0,
    )
    expected = torch.tensor([((15.0 - 2.0) / 15.0) ** 2])
    torch.testing.assert_close(
        interaction_root_heading_loss(outside, target), expected,
        atol=1.0e-5, rtol=1.0e-5,
    )


def test_root_contact_frame_supervision_is_geometry_relative_with_a_dead_zone() -> None:
    reference = torch.zeros((1, 36))
    reference[:, 3] = 1.0
    planned = torch.zeros((1, 6), dtype=torch.bool)
    planned[:, 2] = True
    anchor = torch.zeros((1, 6, 3))
    anchor[:, 2] = torch.tensor((0.60, 0.20, 0.80))

    close = reference.clone()
    close[:, 0] = 0.02
    torch.testing.assert_close(
        interaction_root_contact_frame_loss(close, reference, anchor, planned),
        torch.zeros(1),
    )

    displaced = reference.clone()
    displaced[:, 0] = 0.10
    assert interaction_root_contact_frame_loss(
        displaced, reference, anchor, planned
    ).item() > 0.0

    # With a yaw error, moving the interaction itself changes the loss.  This
    # distinguishes the objective from a fixed event-to-root-step target.
    yaw = torch.deg2rad(torch.tensor(10.0))
    turned = reference.clone()
    turned[:, 3] = torch.cos(yaw / 2.0)
    turned[:, 6] = torch.sin(yaw / 2.0)
    near = anchor.clone()
    near[:, 2, :2] = torch.tensor((0.10, 0.00))
    far = anchor.clone()
    far[:, 2, :2] = torch.tensor((1.00, 0.00))
    near_loss = interaction_root_contact_frame_loss(turned, reference, near, planned)
    far_loss = interaction_root_contact_frame_loss(turned, reference, far, planned)
    assert far_loss.item() > near_loss.item()


def test_balanced_squared_loss_weights_equalize_quadratic_gradient_scale() -> None:
    losses = torch.tensor((0.25, 25.0), requires_grad=True)
    weights = balanced_squared_loss_weights(losses, floor=0.01)
    # A quadratic residual has |grad| proportional to sqrt(loss).  The hard
    # sample keeps ten times less scalar weight, not one hundred times less.
    torch.testing.assert_close(weights[0] / weights[1], torch.tensor(10.0))
