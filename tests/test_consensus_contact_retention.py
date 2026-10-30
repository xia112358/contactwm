import torch
from contact_solver.research.source_contact_retention import ContactRetention


def test_normal_loss_preserves_tangential_freedom_and_inactive_frames():
    term=ContactRetention.__new__(ContactRetention)
    term.upper_only=False
    term.preserve_release=False
    term.indices=[0];term.local=torch.zeros((1,3));term.normal=torch.tensor([[0.,0.,1.]])
    term.target=torch.zeros((2,1));term.mask=torch.tensor([[1.],[0.]])
    r=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]])
    p=torch.tensor([[[2.,3.,.001]],[[0.,0.,100.]]],requires_grad=True)
    loss=term.loss(p,r)
    torch.testing.assert_close(loss,torch.tensor(1.))
    loss.backward()
    assert p.grad[0,0,2]>0
    assert not p.grad[..., :2].any() and not p.grad[1].any()


def test_upper_only_does_not_follow_lower_source_height_noise():
    term=ContactRetention.__new__(ContactRetention)
    term.upper_only=True;term.indices=[0]
    term.preserve_release=False
    term.local=torch.zeros((1,3));term.normal=torch.tensor([[0.,0.,1.]])
    term.target=torch.zeros((2,1));term.mask=torch.ones((2,1))
    r=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]])
    p=torch.tensor([[[0.,0.,-.001]],[[0.,0.,.001]]],requires_grad=True)
    loss=term.loss(p,r);loss.backward()
    assert not p.grad[0].any() and p.grad[1,0,2]>0


def test_release_loss_pushes_outward_only_at_selected_release_frames():
    term=ContactRetention.__new__(ContactRetention)
    term.upper_only=True;term.preserve_release=True;term.indices=[0]
    term.local=torch.zeros((1,3));term.normal=torch.tensor([[0.,0.,1.]])
    term.target=torch.zeros((2,1));term.mask=torch.zeros((2,1))
    term.release_target=torch.ones((2,1))*.001
    term.release_mask=torch.tensor([[1.],[0.]])
    r=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]])
    p=torch.zeros((2,1,3),requires_grad=True)
    loss=term.loss(p,r);loss.backward()
    assert p.grad[0,0,2]<0 and not p.grad[1].any() and not p.grad[..., :2].any()
