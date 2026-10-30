from types import SimpleNamespace

import pytest
import torch

from generator.limb_contact import limb_logits, decode_limb_roles, limb_targets, role_objective
from generator.role_validity import mask_role_logits


def test_joint_choice_resolves_two_confident_part_heads():
    raw = torch.zeros(1, 6, 4)
    raw[0, 1, 2] = 12  # Keep right foot.
    raw[0, 5, 1] = 9   # Independently confident right knee establish.
    contact = torch.tensor([[False, True, False, False, False, False]])
    joint = limb_logits(mask_role_logits(raw, contact))
    roles = decode_limb_roles(joint)
    assert roles[0, 1] == 2 and roles[0, 5] == 0
    raw[0, 5, 1] = 15
    roles = decode_limb_roles(limb_logits(mask_role_logits(raw, contact)))
    assert roles[0, 1] == 0 and roles[0, 5] == 1


def test_all_legal_states_round_trip_and_keep_validity():
    for state in range(7):
        logits = torch.zeros(1, 4, 7)
        logits[:, :2, state] = 10
        roles = decode_limb_roles(logits)
        assert limb_targets(roles)[0, :2].tolist() == [state, state]
    joint = limb_logits(mask_role_logits(torch.zeros(2, 6, 4), torch.zeros(2, 6, dtype=torch.bool)))
    assert torch.isneginf(joint[:, :2, [2, 5]]).all()
    assert torch.isneginf(joint[:, 2:, 2]).all()


def test_joint_supervision_penalizes_competing_knee_and_rejects_conflict():
    raw = torch.zeros(1, 6, 4, requires_grad=True)
    pred = SimpleNamespace(limb_role_logits=limb_logits(raw), role_logits=raw)
    roles = torch.tensor([[0, 2, 0, 0, 0, 0]])
    role_objective(pred, roles).sum().backward()
    assert raw.grad[0, 1, 2] < 0 and raw.grad[0, 5, 1] > 0
    roles[0, 5] = 1
    with pytest.raises(ValueError, match='independent foot and knee'):
        role_objective(pred, roles)


def test_role_selection_does_not_restrict_regional_contact_subsets():
    from generator.position_plan import PositionPlan
    regions = torch.tensor([[[1, 1, 1, 1]]*2 + [[1, 1, 0, 0]]*2 + [[0, 0, 0, 0]]*2], dtype=torch.bool)
    raw = torch.zeros(1, 6, 4)
    raw[:, :4, 1] = 10
    joint = limb_logits(raw)
    role = decode_limb_roles(joint)
    plan = PositionPlan(raw, torch.zeros(1, 6, 1), torch.zeros(1, 6, 3),
        role != 0, torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 6, 3),
        torch.zeros(1, dtype=torch.bool), None, regions, role, limb_role_logits=joint)
    pred = plan.prediction(torch.zeros(1, 36))
    assert torch.equal(pred.role, role)
    assert pred.planned_regions[0, :2].all()  # Rear and front sole together.
    assert pred.planned_regions[0, 2:4, :2].all()
