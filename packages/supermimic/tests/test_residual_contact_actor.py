import torch
import pytest
from supermimic.handoff.bottleneck_actor import BottleneckContactActor
from supermimic.handoff.residual_actor import ResidualContactActor


@pytest.mark.parametrize('mixture', [False, True])
def test_zero_residual_preserves_actions_and_bptt(mixture):
    torch.manual_seed(19)
    base = BottleneckContactActor().eval()
    actor = ResidualContactActor(base, mixture=mixture).eval()
    history = torch.randn(16, 5, 232, requires_grad=True)
    condition = torch.randn(16, 39, requires_grad=True)
    epsilon = torch.randn(16, 32)
    actor.initialize_features(history, condition)
    expected = base(history, condition, epsilon=epsilon)
    actual = actor(history, condition, epsilon=epsilon)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected.square().sum(), (history, condition))
    actual_grad = torch.autograd.grad(actual.square().sum(), (history, condition))
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert all(not p.requires_grad for p in actor.base.parameters())


@pytest.mark.parametrize('mixture', [False, True])
def test_residual_learns_without_updating_base(mixture):
    torch.manual_seed(31)
    actor = ResidualContactActor(BottleneckContactActor(), mixture=mixture)
    history, condition = torch.randn(32, 5, 232), torch.randn(32, 39)
    actor.initialize_features(history, condition)
    original = {k: v.clone() for k, v in actor.base.state_dict().items()}
    params = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    with torch.no_grad(): target = actor(history, condition) + 0.1
    before = (actor(history, condition)-target).square().mean().item()
    for _ in range(4):
        optimizer.zero_grad()
        loss = (actor(history, condition)-target).square().mean()
        loss.backward()
        optimizer.step()
    assert (actor(history, condition)-target).square().mean().item() < before
    assert all(torch.equal(v, original[k]) for k,v in actor.base.state_dict().items())
    if mixture:
        _, aux = actor(history, condition, return_aux=True)
        torch.testing.assert_close(aux['routing'].sum(-1), torch.ones(32))
        assert all(expert[-1].weight.norm()>0 for expert in actor.experts)


def test_grouped_residual_gradient_matches_weighted_backward():
    from supermimic.contact_grouped_gradient import GroupedActorGradients
    torch.manual_seed(44)
    actor = ResidualContactActor(BottleneckContactActor(), mixture=True)
    history, condition = torch.randn(12, 5, 232), torch.randn(12, 39)
    actor.initialize_features(history, condition)
    with torch.no_grad():
        for expert in actor.experts:
            expert[-1].weight.normal_(std=0.01)
    params = tuple(p for p in actor.parameters() if p.requires_grad)
    groups = torch.tensor([0]*2+[1]*4+[2]*6)
    with GroupedActorGradients(actor, groups) as grouped:
        per = actor(history, condition).square().mean(-1)
        torch.autograd.grad(per.mean(), params)
        actual = [(grouped.gradients[p]*(12/grouped.counts).reshape((-1,)+(1,)*(p.ndim))).mean(0).clone() for p in params]
    per = actor(history, condition).square().mean(-1)
    expected = torch.autograd.grad(torch.stack([per[groups==g].mean() for g in range(3)]).mean(), params)
    for got,want in zip(actual,expected):
        torch.testing.assert_close(got,want,rtol=2e-5,atol=2e-7)
    assert actor.router.weight.requires_grad
    assert expected[-2].norm()>0
