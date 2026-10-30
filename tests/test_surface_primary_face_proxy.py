import torch
from contact_solver.surface_contact_proxy import primary_face_cost


def evaluate(point,infinite=False):
    p=torch.tensor(point,dtype=torch.float32).reshape(1,1,3).repeat(1,6,1).requires_grad_()
    face=torch.tensor([[[0,0,0,0,0,1,1,0,0,0,1,0,1,1,int(infinite)]]],dtype=torch.float32)
    cost,gap,excess,edge=primary_face_cost(p,torch.arange(6),face,torch.full((1,6,1),.02))
    return p,cost,gap,excess,edge


def test_approach_cost_is_not_a_top_contact_classifier():
    p,cost,gap,excess,edge=evaluate([1.005,0,.01])
    assert (gap<.02).all() and (excess==0).all()
    assert (cost==0).all() and (edge==0).all()
    cost.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_top_interior_still_allows_actual_solver_margin():
    _,cost,_,_,_=evaluate([.5,0,.01])
    assert (cost==0).all()


def test_ground_has_no_artificial_edges():
    _,cost,_,_,_=evaluate([100,100,.01],infinite=True)
    assert (cost==0).all()


def test_approach_does_not_invent_face_ownership():
    _,cost,_,_,_=evaluate([.999,0,.01])
    assert (cost==0).all()


def test_lower_limb_outside_support_does_not_invalidate_knee_approach():
    points=torch.tensor([[[1.005,0,0],[.5,0,.01]]]*6).reshape(1,12,3)
    parts=torch.arange(6).repeat_interleave(2)
    faces=torch.tensor([[[0,0,0,0,0,1,1,0,0,0,1,0,1,1,0]]],dtype=torch.float32)
    cost,_,_,_=primary_face_cost(points,parts,faces,torch.full((1,6,1),.02))
    assert (cost==0).all()
