import torch
import pytest
from torch import nn
from supermimic.contact_case_gradient import cap_gradient_outliers
from supermimic.contact_grouped_gradient import GroupedActorGradients


def test_unexceptional_rows_keep_magnitudes_directions_and_sample_weights():
    g=torch.tensor([[1.,0.],[0.,2.],[3.,4.]])
    w=torch.tensor([.2,.3,.5])
    direction,audit=cap_gradient_outliers(g,w)
    torch.testing.assert_close(direction,(g*w[:,None]).sum(0))
    assert audit['clipped_groups']==0


def test_outlier_is_capped_before_it_rotates_the_batch_direction():
    g=torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1000.]])
    direction,audit=cap_gradient_outliers(g,torch.full((4,),.25))
    torch.testing.assert_close(direction,torch.tensor([.75,2.5]))
    assert audit['clipped_groups']==1 and audit['clipped_sample_fraction']==.25
    assert audit['cap']==10.


def test_zero_and_nonfinite_gradients_are_not_hidden():
    d,a=cap_gradient_outliers(torch.zeros(3,2),torch.full((3,),1/3))
    assert not d.any() and a['clipped_groups']==0
    with pytest.raises(FloatingPointError):
        cap_gradient_outliers(torch.tensor([[float('nan'),1.]]),torch.ones(1))


def test_grouped_bptt_matches_serial_mlp_and_preserves_weighting():
    torch.manual_seed(7)
    actor=nn.Sequential(nn.Linear(3,8),nn.Tanh(),nn.Linear(8,3)).double()
    x=torch.randn(7,3,dtype=torch.double)
    group=torch.tensor([0,0,0,1,1,2,2])
    def loss(x):
        out=0.
        for _ in range(4):
            x=.7*x+actor(x)
            out=out+x.square().mean(-1)
        return out
    expected=[]
    for i in group.unique():
        expected.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss(x[group==i]).mean(),tuple(actor.parameters()))]))
    with GroupedActorGradients(actor,group) as grouped:
        total=loss(x).mean()
        direct=torch.autograd.grad(total,tuple(actor.parameters()))
        rows=grouped.matrix()
    torch.testing.assert_close(rows,torch.stack(expected))
    direction,audit=cap_gradient_outliers(rows,grouped.counts.to(rows.dtype)/len(x),ratio=1e6)
    torch.testing.assert_close(direction,torch.cat([g.flatten() for g in direct]))
    assert all(not m._forward_hooks for m in actor.modules())
