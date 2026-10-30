from __future__ import annotations

import torch
from generator.contact_q_generator import SequentialContactQPredictor


def predictor(
    *,
    refinement_steps: int = 0,
    action_joint_state: bool = True,
    contact_conditioned_selector: bool = False,
) -> SequentialContactQPredictor:
    touchdown = torch.tensor(((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0)), dtype=torch.bool)
    end_contact = torch.tensor(((1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0)), dtype=torch.bool)
    base_start = torch.zeros(2, 36)
    base_start[:, 3] = 1.0
    base_start[:, 2] = 0.75
    base_end = base_start.clone()
    base_end[:, 0] = 0.25
    return SequentialContactQPredictor(
        touchdown,
        end_contact,
        base_start,
        base_end,
        torch.zeros(6, 2, 3),
        torch.zeros(36),
        torch.ones(36),
        torch.zeros(340),
        torch.ones(340),
        torch.zeros(6, 3),
        torch.ones(6, 3),
        torch.tensor(0.0),
        torch.tensor(1.0),
        width=32,
        blocks=1,
        action_embedding_dim=8,
        refinement_steps=refinement_steps,
        action_joint_state=action_joint_state,
        contact_conditioned_selector=contact_conditioned_selector,
    )


def test_contact_plan_is_realized_before_fk_boundary() -> None:
    torch.manual_seed(3)
    model = predictor()
    current_q = model.action_base_start_qpos[:2].clone()
    current_contact = torch.tensor(((1, 1, 0, 0, 0, 0),) * 2, dtype=torch.float32)
    scan = torch.zeros(2, 340)
    hidden, plan = model.plan(current_q, current_contact, scan, torch.tensor((0, 1)))
    first = model.realize_pose(hidden, current_q, plan)
    shifted = plan.with_geometry(target_position=plan.target_position + 0.1)
    second = model.realize_pose(hidden, current_q, shifted)
    assert first.qpos.shape == (2, 36)
    assert first.keypoint_position.shape == (2, 7, 3)
    assert not torch.allclose(first.qpos, second.qpos)
    expected_position, expected_rotation = model.fk(first.qpos)
    torch.testing.assert_close(first.keypoint_position, expected_position)
    torch.testing.assert_close(first.keypoint_rotation6d, expected_rotation)


def test_current_joint_tracking_error_is_not_reinterpreted_as_action_shape() -> None:
    model = predictor()
    current = model.action_base_start_qpos[:1].clone()
    perturbed = current.clone()
    perturbed[:, 7:] += 0.2
    action = torch.tensor((0,))
    nominal_endpoint = model.aligned_action_endpoint(action, current)
    perturbed_endpoint = model.aligned_action_endpoint(action, perturbed)
    torch.testing.assert_close(perturbed_endpoint[:, 7:], nominal_endpoint[:, 7:])


def test_pose_refiner_consumes_its_fk_contact_residual() -> None:
    torch.manual_seed(5)
    model = predictor(refinement_steps=2)
    current = model.action_base_start_qpos[:1].clone()
    contact = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    scan = torch.zeros(1, 340)
    hidden, plan = model.plan(current, contact, scan, torch.tensor((0,)))
    output = model.realize_pose(hidden, current, plan)
    assert output.qpos.shape == (1, 36)
    output.qpos.sum().backward()
    assert all(parameter.grad is not None for refiner in model.pose_refiners for parameter in refiner.parameters())


def test_action_selection_does_not_use_arbitrary_joint_realization() -> None:
    torch.manual_seed(6)
    model = predictor(action_joint_state=False)
    current = model.action_base_start_qpos[:1].clone()
    perturbed = current.clone()
    perturbed[:, 7:] += 0.3
    contact = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    scan = torch.zeros(1, 340)
    nominal = model.plan(current, contact, scan)[1].action_logits
    changed = model.plan(perturbed, contact, scan)[1].action_logits
    torch.testing.assert_close(changed, nominal)


def test_contact_conditioned_selector_and_realized_contact_contract() -> None:
    torch.manual_seed(7)
    model = predictor(refinement_steps=1, contact_conditioned_selector=True)
    current = model.action_base_start_qpos[:1].clone()
    perturbed = current.clone()
    perturbed[:, 7:] += 0.3
    contact = torch.tensor(((1, 1, 0, 0, 0, 0),), dtype=torch.float32)
    scan = torch.zeros(1, 340)
    contact_position = torch.zeros(1, 6, 3)
    contact_rotation = torch.zeros(1, 6, 6)
    contact_rotation[..., 0] = 1.0
    contact_rotation[..., 4] = 1.0
    hidden, plan = model.plan(
        current,
        contact,
        scan,
        current_contact_position=contact_position,
        current_contact_rotation6d=contact_rotation,
    )
    changed_plan = model.plan(
        perturbed,
        contact,
        scan,
        current_contact_position=contact_position,
        current_contact_rotation6d=contact_rotation,
    )[1]
    torch.testing.assert_close(changed_plan.action_logits, plan.action_logits)
    torch.testing.assert_close(changed_plan.target_position, plan.target_position)
    prediction = model.realize_pose(hidden, current, plan)
    torch.testing.assert_close(prediction.plan.target_position, prediction.keypoint_position[:, (1, 2, 3, 4, 5, 6)])
    torch.testing.assert_close(
        prediction.plan.target_rotation6d,
        prediction.keypoint_rotation6d[:, (1, 2, 3, 4, 5, 6)],
    )
