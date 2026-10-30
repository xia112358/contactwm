import torch

from contact_solver.newton_witness_loss import selected_contact_activation_deficit


def _pair(*, part=4, surface=0, dist=0.023, margin=0.020, active=False, allocated=False):
    metadata = {
        "part": part,
        "surface": surface,
        "dist": dist,
        "includemargin": margin,
        "constraint_active": active,
        "allocated": allocated,
    }
    return metadata


def test_activation_deficit_uses_actual_margin_and_backpropagates():
    q = torch.zeros((1, 2), requires_grad=True)
    distance = q[0, 0] + 0.023
    rows = [[(_pair(), distance)]]
    intended = torch.zeros((1, 6), dtype=torch.bool)
    intended[0, 4] = True
    surface = torch.zeros((1, 6), dtype=torch.long)

    loss, missing, realized = selected_contact_activation_deficit(
        q, rows, intended, surface, interior_fraction=0.05
    )

    # The target is 5% inside the real 20 mm activation interval: 19 mm.
    torch.testing.assert_close(loss[0, 4], torch.tensor((0.023 - 0.019) / 0.020))
    assert not missing.any()
    assert not realized.any()
    loss.sum().backward()
    assert q.grad[0, 0] > 0.0


def test_active_allocated_contact_has_zero_deficit():
    q = torch.zeros((1, 2), requires_grad=True)
    rows = [[(_pair(dist=0.018, active=True, allocated=True), q[0, 0] + 0.018)]]
    intended = torch.zeros((1, 6), dtype=torch.bool)
    intended[0, 4] = True
    surface = torch.zeros((1, 6), dtype=torch.long)

    loss, missing, realized = selected_contact_activation_deficit(
        q, rows, intended, surface
    )

    assert loss.sum() == 0.0
    assert not missing.any()
    assert realized[0, 4]
