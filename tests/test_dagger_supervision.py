import numpy as np
import torch
from generator.dagger_supervision import endpoint_gate, recovery_roles, verified_target


def test_gate_rejects_any_penetration_extra_contact_and_joint_limit():
    q = np.zeros(36); q[3] = 1
    active = np.array([1, 0, 0, 0, 0, 0], bool)
    surface = np.array([0, -1, -1, -1, -1, -1])
    actual = dict(contact_part_mask=active, contact_surface=surface)
    report = dict(schema='newton_full_robot_signed_separation_v1', terrain_rows=1,
                  self_rows=0, terrain_penetration_m=0., self_penetration_m=0.,
                  worst_terrain=None, worst_self=None)
    lo, hi = -np.ones(29), np.ones(29)
    assert endpoint_gate(q, active, surface, actual, report, lo, hi)
    bad = dict(report, terrain_penetration_m=1e-12, worst_terrain={'dist': -1e-12})
    assert not endpoint_gate(q, active, surface, actual, bad, lo, hi)
    assert not endpoint_gate(q, active, surface, actual, {}, lo, hi)
    extra = dict(actual, contact_part_mask=np.ones(6, bool))
    assert not endpoint_gate(q, active, surface, extra, report, lo, hi)
    q[7] = 1.000001
    assert not endpoint_gate(q, active, surface, actual, report, lo, hi)


def test_missing_support_is_recovery_touchdown_not_input_contact():
    active = torch.tensor([[False, True]])
    surface = torch.tensor([[-1, 0]])
    result = recovery_roles(active, surface, torch.ones_like(active),
                            torch.zeros_like(surface), torch.tensor([[2, 2]]))
    assert result.tolist() == [[1, 2]]
    assert active.tolist() == [[False, True]]


def test_verified_pose_replaces_old_label_and_material_anchor():
    from generator.next_interaction_joint import JointInteractionPredictor
    model = JointInteractionPredictor(48, 1)
    q = torch.zeros(1, 36); q[:, 3] = 1; q[:, 2] = .8
    old = q.clone(); old[:, 2] = .1
    anchors = torch.zeros(1, 6, 3)
    t = verified_target(model, {'q': old}, q, anchors)
    assert torch.equal(t['q'], q) and not torch.equal(t['q'], old)
    assert torch.equal(t['anchor'], anchors)
    assert torch.isfinite(t['material_local']).all()
