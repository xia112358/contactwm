from __future__ import annotations

from dataclasses import replace
from types import MethodType

import numpy as np
import torch

from somaforge_core.motion_contracts import CONTACT_BODY_INDEX, InteractionBoundary
from generator.unified_interaction import UnifiedInteractionPredictor, unified_interaction_loss
from contact_solver.trajectory_projection import contact_rotation_losses, rollout_dense_collision_loss
from somaforge_core.g1_kinematics import _matrix_from_rotation6d
from generator.unified_interaction import _endpoint_root_shift_penalty, _limit_endpoint_root_shift


def test_endpoint_root_shift_penalty_only_activates_above_limit() -> None:
    initial = torch.zeros((2, 3, 36), dtype=torch.float32)
    candidate = initial.clone()
    candidate[0, -1, 0] = 0.10
    candidate[1, -1, 0] = 0.20

    penalty = _endpoint_root_shift_penalty(candidate, initial, 0.15)

    torch.testing.assert_close(penalty, torch.tensor(0.5))


def test_contact_rotation_losses_only_update_active_contact_orientation() -> None:
    identity = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    quarter_turn = torch.tensor((0.0, -1.0, 1.0, 0.0, 0.0, 0.0))
    predicted = identity.repeat(1, 3, 6, 1)
    predicted[0, 1:, 0] = quarter_turn
    predicted.requires_grad_()
    target = identity.repeat(1, 3, 6, 1)
    mask = torch.zeros(1, 3, 6, dtype=torch.bool)
    mask[:, :, 0] = True

    orientation, angular_velocity = contact_rotation_losses(predicted, target, mask)

    assert orientation > 0.0
    assert angular_velocity > 0.0
    (orientation + angular_velocity).backward()
    assert predicted.grad is not None
    assert float(predicted.grad[..., 0, :].abs().sum()) > 0.0
    torch.testing.assert_close(predicted.grad[..., 1:, :], torch.zeros_like(predicted.grad[..., 1:, :]))


def test_contact_rotation_losses_can_balance_rare_contact_parts() -> None:
    identity = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    quarter_turn = torch.tensor((0.0, -1.0, 1.0, 0.0, 0.0, 0.0))
    predicted = identity.repeat(4, 6, 1)
    predicted[0, 5] = quarter_turn
    target = identity.repeat(4, 6, 1)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    mask[:, 0] = True
    mask[0, 5] = True

    sample_mean, _ = contact_rotation_losses(predicted, target, mask)
    part_mean, _ = contact_rotation_losses(
        predicted,
        target,
        mask,
        balance_contact_parts=True,
    )

    assert part_mean > sample_mean


def test_endpoint_root_shift_is_projected_to_hard_limit() -> None:
    tail = torch.zeros((1, 2, 36), dtype=torch.float32)
    tail[0, -1, :3] = torch.tensor((0.3, 0.4, 0.0))

    limited = _limit_endpoint_root_shift(tail, torch.zeros((1, 3)), 0.15)

    torch.testing.assert_close(limited[0, -1, :3], torch.tensor((0.09, 0.12, 0.0)))
    torch.testing.assert_close(limited[0, 0], tail[0, 0])


def test_rollout_dense_collision_loss_penalizes_depth_duration_and_worst_frame() -> None:
    penetration = torch.tensor((((0.001, 0.002), (0.012, 0.002)),), dtype=torch.float32)

    loss = rollout_dense_collision_loss(penetration)

    expected = 0.25 + 0.125 + 0.25 + 0.125 * (1.0 - torch.exp(torch.tensor(-1.0)))
    torch.testing.assert_close(loss, expected)


def test_rollout_dense_collision_loss_increases_when_collision_lasts_longer() -> None:
    brief = torch.tensor((((0.012,), (0.002,)),), dtype=torch.float32)
    sustained = torch.tensor((((0.012,), (0.012,)),), dtype=torch.float32)

    assert rollout_dense_collision_loss(sustained) > rollout_dense_collision_loss(brief)


def test_exponential_collision_value_and_gradient() -> None:
    from generator.unified_interaction import _collision_depth_cost
    x = torch.tensor([0.8, 1.8, 3.8], requires_grad=True)
    cost = _collision_depth_cost(x, "exponential")
    torch.testing.assert_close(cost, torch.expm1(x))
    cost.sum().backward()
    torch.testing.assert_close(x.grad, torch.exp(x.detach()))


def test_exponential_collision_extreme_depth_keeps_finite_positive_gradient() -> None:
    from generator.unified_interaction import rollout_ground_collision_loss
    for loss_fn in (rollout_dense_collision_loss, rollout_ground_collision_loss):
        depth = torch.tensor([[[0.0], [0.001], [0.002], [0.042], [0.5]]], requires_grad=True)
        loss = loss_fn(depth, mode="exponential")
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(depth.grad).all()
        assert (depth.grad[:, :3] == 0).all()
        assert (depth.grad[:, 3:] > 0).all()


def test_exponential_collision_does_not_dilute_worst_frame() -> None:
    brief = torch.tensor([[[0.042]]])
    padded = torch.cat((brief, torch.zeros(1, 99, 1)), dim=1)
    assert rollout_dense_collision_loss(padded, mode="exponential") >= 0.25 * torch.expm1(torch.tensor(4.0))


def model() -> UnifiedInteractionPredictor:
    return UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
    )


def test_exact_no_velocity_predictor_keeps_legacy_layout_and_ignores_velocity() -> None:
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
        predict_boundary_velocity=False,
    )
    assert predictor.state_encoder[0].weight.shape == (32, 103)
    assert predictor.interaction_head.weight.shape == (85, 32)

    q = torch.zeros(1, 36)
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    contact = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    anchor = torch.zeros(1, 6, 3)
    surface = torch.zeros(1, 6, dtype=torch.long)
    scan = torch.zeros(1, 340)
    still = predictor(q, contact, anchor, surface, scan, current_q_velocity=torch.zeros_like(q))
    moving = predictor(q, contact, anchor, surface, scan, current_q_velocity=torch.full_like(q, 10.0))

    torch.testing.assert_close(still.qpos, moving.qpos)
    torch.testing.assert_close(still.duration, moving.duration)
    torch.testing.assert_close(still.q_velocity, torch.zeros_like(still.q_velocity))
    torch.testing.assert_close(still.contact_velocity, torch.zeros_like(still.contact_velocity))


def test_persistent_constraint_removes_contact_dofs_and_remains_differentiable() -> None:
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
        persistent_constraint_iterations=4,
        persistent_constraint_damping=1.0e-4,
    )
    initial = torch.zeros(1, 36)
    initial[:, 3] = 1.0
    surface = torch.zeros(1, 6, dtype=torch.long)
    local_offset = torch.zeros(1, 6, 3)
    anchor = predictor.contact_points(initial, surface, local_offset).detach()
    candidate = initial.clone()
    candidate[:, :3] += torch.tensor((0.04, -0.03, 0.02))
    candidate[:, 7:] += 0.03
    candidate.requires_grad_(True)
    persistent = torch.tensor(((1, 1, 1, 1, 0, 0),), dtype=torch.bool)

    constrained = predictor.constrain_persistent_contacts(
        candidate, surface, local_offset, persistent, anchor
    )
    error = torch.linalg.vector_norm(
        predictor.contact_points(constrained, surface, local_offset) - anchor, dim=-1
    )[persistent]
    constrained.square().mean().backward()

    assert float(error.max().detach()) < 1.0e-5
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert float(candidate.grad.abs().sum()) > 0.0


def test_dls_teacher_override_keeps_deployment_models_raw() -> None:
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
        persistent_constraint_iterations=0,
        persistent_constraint_damping=1.0e-4,
        persistent_constraint_maximum_step=1.0,
    )
    initial = torch.zeros(1, 36)
    initial[:, 2] = 0.75
    initial[:, 3] = 1.0
    surface = torch.zeros(1, 6, dtype=torch.long)
    offset = torch.zeros(1, 6, 3)
    anchor = predictor.contact_points(initial, surface, offset).detach()
    candidate = initial.clone()
    candidate[:, :3] += torch.tensor((0.04, -0.03, 0.02))
    persistent = torch.tensor(((1, 1, 1, 1, 0, 0),), dtype=torch.bool)

    raw = predictor.constrain_persistent_contacts(candidate, surface, offset, persistent, anchor)
    teacher = predictor.constrain_persistent_contacts(
        candidate, surface, offset, persistent, anchor, iterations=4
    )
    raw_error = torch.linalg.vector_norm(
        predictor.contact_points(raw, surface, offset) - anchor, dim=-1
    )[persistent]
    teacher_error = torch.linalg.vector_norm(
        predictor.contact_points(teacher, surface, offset) - anchor, dim=-1
    )[persistent]

    torch.testing.assert_close(raw, candidate)
    assert float(teacher_error.mean()) < float(raw_error.mean())


def test_synchronized_keyframe_recomputes_q_and_contact_geometry_together() -> None:
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
        persistent_constraint_iterations=4,
        persistent_constraint_damping=1.0e-4,
        persistent_constraint_maximum_step=1.0,
    )
    q = torch.zeros(1, 36)
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    contact = torch.tensor(((1, 0, 0, 0, 0, 0),), dtype=torch.float32)
    surface = torch.zeros(1, 6, dtype=torch.long)
    anchor = predictor.contact_points(q, surface)
    prediction = predictor(q, contact, anchor, surface, torch.zeros(1, 340))
    prediction = replace(
        prediction,
        touchdown=torch.tensor(((1, 0, 0, 0, 0, 0),), dtype=torch.bool),
        persistent_support=torch.zeros(1, 6, dtype=torch.bool),
    )
    target = prediction.contact_position.detach().clone()
    target[:, 0, 2] += 0.04

    synchronized = predictor.synchronize_contact_targets(prediction, target)
    recomputed = predictor.contact_points(
        synchronized.qpos,
        synchronized.contact_surface,
        synchronized.contact_local_offset,
    )

    torch.testing.assert_close(synchronized.contact_position, recomputed)
    assert float((synchronized.contact_position[:, 0] - target[:, 0]).norm().detach()) < 1.0e-5
    assert float((synchronized.qpos - prediction.qpos).abs().max().detach()) > 0.0


def test_boundary_derives_four_transition_sets_without_requiring_all_parts() -> None:
    pose = np.zeros((7, 3), dtype=np.float32)
    rotation = np.zeros((7, 6), dtype=np.float32)
    anchor = np.zeros((6, 3), dtype=np.float32)
    current = InteractionBoundary(
        pose, rotation, np.array((1, 1, 0, 0, 0, 0), dtype=np.bool_), np.zeros(6, dtype=np.bool_), np.array((1, 1, 0, 0, 0, 0), dtype=np.bool_), anchor, np.zeros(6, dtype=np.int64), 0.3
    )
    end = InteractionBoundary(
        pose, rotation, np.array((1, 0, 1, 0, 0, 0), dtype=np.bool_), np.array((0, 0, 1, 0, 0, 0), dtype=np.bool_), np.array((1, 0, 0, 0, 0, 0), dtype=np.bool_), anchor, np.zeros(6, dtype=np.int64), 0.4
    )
    np.testing.assert_array_equal(end.persistent_from(current), (1, 0, 0, 0, 0, 0))
    np.testing.assert_array_equal(end.touchdown_from(current), (0, 0, 1, 0, 0, 0))
    np.testing.assert_array_equal(end.liftoff_from(current), (0, 1, 0, 0, 0, 0))
    np.testing.assert_array_equal(end.swing_from(current), (0, 0, 0, 1, 1, 1))


def test_predictor_has_no_action_bottleneck_and_contact_geometry_is_fk_synchronous() -> None:
    torch.manual_seed(11)
    predictor = model()
    assert not hasattr(predictor, "action_head")
    q = torch.zeros(2, 36)
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    contact = torch.tensor(((1, 1, 0, 0, 0, 0),) * 2, dtype=torch.float32)
    prediction = predictor(q, contact, torch.zeros(2, 6, 3), torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 340))
    assert prediction.keypoint_position.shape == (2, 7, 3)
    assert prediction.contact_logits.shape == (2, 6)
    rotation = _matrix_from_rotation6d(prediction.keypoint_rotation6d)
    expected = []
    for part, body in enumerate(CONTACT_BODY_INDEX):
        offset = prediction.contact_local_offset[:, part]
        expected.append(
            prediction.keypoint_position[:, body] - torch.einsum("bij,bj->bi", rotation[:, body], offset)
        )
    torch.testing.assert_close(prediction.contact_position, torch.stack(expected, dim=1))


def test_structured_topology_decoder_rejects_unseen_contact_transition() -> None:
    transition = torch.tensor(
        (
            ((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
            ((1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
        ),
        dtype=torch.bool,
    )
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3), torch.zeros(36), torch.ones(36), torch.zeros(340), torch.ones(340),
        torch.tensor(0.0), torch.tensor(1.0), width=32, blocks=1, topology_transitions=transition
    )
    current = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    illegal_contact = torch.tensor(((2.0, 2.0, 9.0, -9.0, -9.0, -9.0),))
    persistent = torch.tensor(((-2.0, 2.0, -9.0, -9.0, -9.0, -9.0),))
    touchdown = torch.tensor(((2.0, -2.0, 9.0, -9.0, -9.0, -9.0),))
    active, support, landing = predictor.decode_topology(current, illegal_contact, persistent, touchdown)
    torch.testing.assert_close(active, transition[0, 3][None])
    torch.testing.assert_close(support, transition[0, 1][None])
    torch.testing.assert_close(landing, transition[0, 2][None])


def test_phase_topology_overrides_ambiguous_legal_branch_only_when_start_matches() -> None:
    transition = torch.tensor(
        (
            ((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
            ((1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
        ), dtype=torch.bool,
    )
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3), torch.zeros(36), torch.ones(36), torch.zeros(340), torch.ones(340),
        torch.tensor(0.0), torch.tensor(1.0), width=32, blocks=1,
        topology_transitions=transition, phase_topologies=transition,
    )
    current = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    logits = torch.zeros(1, 6)
    active, support, landing = predictor.decode_topology(
        current, logits, logits, logits, torch.ones(1)
    )
    torch.testing.assert_close(active, transition[1, 3][None])
    torch.testing.assert_close(support, transition[1, 1][None])
    torch.testing.assert_close(landing, transition[1, 2][None])


def test_phase_q_prototype_is_the_zero_residual_pose_base() -> None:
    topology = torch.tensor(
        (((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),),
        dtype=torch.bool,
    )
    prototype = torch.zeros(1, 36)
    prototype[:, :3] = torch.tensor((0.2, -0.1, 0.8))
    prototype[:, 3] = 1.0
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3), torch.zeros(36), torch.ones(36), torch.zeros(340), torch.ones(340),
        torch.tensor(0.0), torch.tensor(1.0), width=32, blocks=1,
        topology_transitions=topology, phase_topologies=topology, phase_q_prototypes=prototype,
    )
    current = torch.zeros(1, 36)
    current[:, 2] = 0.7
    current[:, 3] = 1.0
    contact = topology[:, 0].float()
    prediction = predictor(
        current, contact, torch.zeros(1, 6, 3), torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 340),
        interaction_phase=torch.zeros(1),
    )
    torch.testing.assert_close(prediction.qpos, prototype)


def test_phase_q_adapter_changes_only_the_selected_phase_pose() -> None:
    topology = torch.tensor(
        (
            ((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
            ((1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
        ),
        dtype=torch.bool,
    )
    prototype = torch.zeros(2, 36)
    prototype[:, 3] = 1.0
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3), torch.zeros(36), torch.ones(36), torch.zeros(340), torch.ones(340),
        torch.tensor(0.0), torch.tensor(1.0), width=32, blocks=1,
        topology_transitions=topology, phase_topologies=topology, phase_q_prototypes=prototype,
    )
    with torch.no_grad():
        predictor.phase_q_residual_adapter[1, 0] = 1.0
    current = prototype.clone()
    contact = topology[:, 0].float()
    prediction = predictor(
        current, contact, torch.zeros(2, 6, 3), torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 340),
        interaction_phase=torch.tensor((0.0, 1.0)),
    )
    torch.testing.assert_close(prediction.qpos[0], prototype[0])
    torch.testing.assert_close(prediction.qpos[1, 0], 0.5 * torch.tanh(torch.tensor(1.0)))


def test_phase_q_feature_adapter_gradient_is_isolated_by_phase() -> None:
    topology = torch.tensor(
        (
            ((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
            ((1, 1, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)),
        ),
        dtype=torch.bool,
    )
    prototype = torch.zeros(2, 36)
    prototype[:, 3] = 1.0
    predictor = UnifiedInteractionPredictor(
        torch.zeros(6, 2, 3), torch.zeros(36), torch.ones(36), torch.zeros(340), torch.ones(340),
        torch.tensor(0.0), torch.tensor(1.0), width=32, blocks=1,
        topology_transitions=topology, phase_topologies=topology, phase_q_prototypes=prototype,
    )
    current = prototype[1:2].clone()
    prediction = predictor(
        current, topology[1:2, 0].float(), torch.zeros(1, 6, 3), torch.zeros(1, 6, dtype=torch.long),
        torch.ones(1, 340), interaction_phase=torch.ones(1),
    )
    prediction.qpos[:, 0].sum().backward()
    assert predictor.phase_q_feature_adapter.grad is not None
    torch.testing.assert_close(
        predictor.phase_q_feature_adapter.grad[0],
        torch.zeros_like(predictor.phase_q_feature_adapter.grad[0]),
    )
    assert float(predictor.phase_q_feature_adapter.grad[1].abs().sum()) > 0.0


def test_joint_loss_backpropagates_through_shared_interaction_head() -> None:
    torch.manual_seed(12)
    predictor = model()
    q = torch.zeros(2, 36)
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    current_contact = torch.tensor(((1, 1, 0, 0, 0, 0),) * 2, dtype=torch.float32)
    current_anchor = torch.zeros(2, 6, 3)
    prediction = predictor(
        q, current_contact, current_anchor, torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 340)
    )
    target_contact = torch.tensor(((1, 0, 1, 0, 0, 0),) * 2, dtype=torch.float32)
    target_anchor = prediction.contact_position.detach().clone()
    loss = unified_interaction_loss(
        prediction,
        target_qpos=q,
        target_contact=target_contact,
        target_touchdown=torch.tensor(((0, 0, 1, 0, 0, 0),) * 2, dtype=torch.float32),
        target_persistent=torch.tensor(((1, 0, 0, 0, 0, 0),) * 2, dtype=torch.float32),
        target_surface=torch.zeros(2, 6, dtype=torch.long),
        target_contact_position=target_anchor,
        target_duration=torch.ones(2),
        current_contact=current_contact,
        current_contact_position=target_anchor,
    )
    loss.total.backward()
    assert predictor.interaction_head.weight.grad is not None
    assert torch.isfinite(predictor.interaction_head.weight.grad).all()








def test_contact_velocity_is_derived_from_same_q_state() -> None:
    predictor = model()
    qpos = torch.zeros(1, 36)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    surface = torch.zeros(1, 6, dtype=torch.long)
    offset = predictor.endpoint_offsets[:, 0][None]
    velocity = torch.zeros_like(qpos)
    velocity[:, 0] = 0.2

    contact_velocity = predictor.contact_point_velocity(qpos, velocity, surface, offset)

    torch.testing.assert_close(
        contact_velocity,
        torch.tensor((0.2, 0.0, 0.0)).repeat(1, 6, 1),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_velocity_decoder_is_exactly_compatible_with_stationary_contact() -> None:
    predictor = model()
    qpos = torch.zeros(1, 36)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    surface = torch.zeros(1, 6, dtype=torch.long)
    offset = predictor.endpoint_offsets[:, 0][None]
    raw_velocity = torch.randn_like(qpos)
    stationary = torch.tensor(((1, 0, 0, 0, 0, 0),), dtype=torch.bool)

    velocity = predictor.decode_contact_consistent_velocity(
        qpos, raw_velocity, surface, offset, stationary
    )
    contact_velocity = predictor.contact_point_velocity(qpos, velocity, surface, offset)

    assert float(torch.linalg.vector_norm(contact_velocity[:, 0], dim=-1).max()) < 2.0e-4
