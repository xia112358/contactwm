import torch
import pytest
from generator.next_interaction_surface import region_cost


def example():
    faces=torch.zeros(1,2,15)
    faces[:,:,5]=1;faces[:,:,6]=1;faces[:,:,10]=1
    faces[:,:,12:14]=1;faces[:,0,14]=1;faces[:,1,2]=1
    points=torch.tensor([[[0.,0.,1.01]]*6],requires_grad=True)
    return points,torch.arange(6),faces,torch.ones(1,6,dtype=torch.long),torch.ones(1,6,dtype=torch.bool),torch.full((1,6,2),.02),torch.ones(1,6,2,dtype=torch.bool)


def test_touchdown_free_inside_actual_face():
    args=example();cost,_,_=region_cost(*args)
    moved=args[0].detach().clone();moved[:,:,0]=.8
    other,_,_=region_cost(moved,*args[1:])
    torch.testing.assert_close(cost,other);assert cost.item()==0


def test_outside_face_and_gap_have_gradient():
    args=list(example());args[0]=torch.tensor([[[1.3,0.,1.2]]*6],requires_grad=True)
    cost,_,_=region_cost(*args);cost.sum().backward()
    assert cost.item()>0 and torch.isfinite(args[0].grad).all()
    assert (args[0].grad[:,:,0]>0).all() and (args[0].grad[:,:,2]>0).all()


def test_no_invented_margin():
    args=list(example());args[-1][:]=False
    with pytest.raises(ValueError,match='unknown Newton margin'):region_cost(*args)
