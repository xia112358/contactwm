import pytest
import torch

from contact_solver.contact_priority import contact_priority_loss


def test_each_failed_output_keeps_primary_gradient_and_disables_secondary():
    primary = torch.tensor([9., 1., 4.], requires_grad=True)
    secondary = torch.tensor([100., 2., 30.], requires_grad=True)
    accepted = torch.tensor([False, True, False])
    loss = contact_priority_loss(primary, secondary, accepted)
    torch.testing.assert_close(loss, torch.tensor([9., 3., 4.]))
    loss.sum().backward()
    torch.testing.assert_close(primary.grad, torch.ones(3))
    torch.testing.assert_close(secondary.grad, torch.tensor([0., 1., 0.]))


def test_rejects_broadcast_acceptance_and_numeric_intent_as_truth():
    with pytest.raises(ValueError):
        contact_priority_loss(torch.ones(2), torch.ones(2), torch.tensor([True]))
    with pytest.raises(ValueError):
        contact_priority_loss(torch.ones(2), torch.ones(2), torch.ones(2))
