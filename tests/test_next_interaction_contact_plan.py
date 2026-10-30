import dataclasses
import torch
from generator.next_interaction_contact_plan import ContactPlanPredictor,hard_choice


def fixture():
    torch.manual_seed(71)
    model=ContactPlanPredictor(48,1).eval()
    q=torch.zeros(2,36);q[:,3]=1
    faces=torch.tensor([[[0,0,0,0,0,1,1,0,0,0,1,0,1,1,0],
                         [0,0,2,0,0,1,1,0,0,0,1,0,.5,.5,0]]],dtype=torch.float).expand(2,-1,-1)
    part=torch.randn(2,6,48)
    roles=torch.zeros(2,6,4);roles[...,1]=3
    surfaces=torch.zeros(2,6,2);surfaces[...,1]=3
    return model,q,faces,model.make_plan(part,faces,roles,surfaces)


def test_selects_one_real_face_and_interior_point():
    _,_,_,plan=fixture()
    torch.testing.assert_close(plan.point_w[...,2],torch.full((2,6),2.))
    assert (plan.point_w[...,:2].abs()<=.5).all()
    assert ((plan.surface_weights==0)|(plan.surface_weights==1)).all()
    torch.testing.assert_close((plan.normal_w*plan.tangent_w).sum(-1),torch.zeros(2,6),atol=1e-6,rtol=0)


def test_contact_condition_causally_changes_pose_and_has_gradient():
    model,q,_,plan=fixture()
    point=plan.point_w.detach().requires_grad_()
    plan=dataclasses.replace(plan,point_w=point)
    first,_=model.decode_plan(q,plan)
    shifted=point+torch.tensor([.2,0,0])
    second,_=model.decode_plan(q,dataclasses.replace(plan,point_w=shifted))
    assert (first-second).abs().max()>1e-7
    gradient=torch.autograd.grad(first[:,:3].sum(),point)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_inactive_geometry_does_not_constrain_pose():
    model,q,_,plan=fixture()
    roles=torch.zeros_like(plan.role_weights);roles[...,0]=1
    plan=dataclasses.replace(plan,role_weights=roles)
    first,_=model.decode_plan(q,plan)
    second,_=model.decode_plan(q,dataclasses.replace(plan,point_w=plan.point_w+100,normal_w=-plan.normal_w))
    torch.testing.assert_close(first,second)


def test_hard_selection_keeps_learning_signal():
    logits=torch.tensor([[0.,2.]],requires_grad=True)
    selected=hard_choice(logits)
    torch.testing.assert_close(selected,torch.tensor([[0.,1.]]))
    selected[0,0].backward()
    assert logits.grad.abs().sum()>0


def test_observation_only_forward_contract():
    model,q,faces,_=fixture()
    result=model(current_q=q,current_contact=torch.zeros(2,6,dtype=torch.bool),
                 current_anchor=torch.zeros(2,6,3),current_surface=torch.zeros(2,6,dtype=torch.long),faces=faces)
    assert result.qpos.shape==(2,36) and torch.isfinite(result.qpos).all()
    assert result.plan.point_w.shape==(2,6,3)
    torch.testing.assert_close(result.qpos[:,3:7].norm(dim=-1),torch.ones(2),atol=1e-6,rtol=0)
