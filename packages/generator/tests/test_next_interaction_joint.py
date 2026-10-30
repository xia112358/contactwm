import inspect
import torch
from generator.next_interaction import NextInteraction
from generator.next_interaction_joint import JointInteractionPredictor,compatibility_cost


def test_pose_and_duration_depend_on_predicted_topology():
    torch.set_num_threads(1);torch.manual_seed(9)
    m=JointInteractionPredictor(48,1)
    assert list(inspect.signature(m.forward).parameters)==['current_q','current_contact','current_anchor','current_surface','faces']
    q=torch.zeros(2,36);q[:,3]=1.;q[:,2]=.8
    faces=torch.tensor([0,0,0,0,0,1,1,0,0,0,1,0,1,1,1.])[None,None].repeat(2,2,1)
    faces[:,1,2]=.7;faces[:,1,14]=0
    pred=m(q,torch.zeros(2,6,dtype=torch.bool),torch.zeros(2,6,3),torch.full((2,6),-1),faces)
    (pred.qpos.square().sum()+pred.duration.sum()).backward()
    assert m.role_head.weight.grad.abs().sum()>0
    assert m.surface_query.weight.grad.abs().sum()>0
    assert m.interaction_decoder.layers[0].linear1.weight.grad.abs().sum()>0


def test_consistency_moves_geometry_not_contact_flags():
    points=torch.zeros(1,6,3,requires_grad=True)
    with torch.no_grad():points[:,:,2]=.5
    role=torch.tensor([0.,10.,0.,0.]).expand(1,6,4).clone().requires_grad_()
    surface=torch.zeros(1,6,1,requires_grad=True)
    pred=NextInteraction(torch.empty(1,36),role,surface,torch.ones(1))
    faces=torch.tensor([0,0,0,0,0,1,1,0,0,0,1,0,1,1,1.])[None,None]
    cost,metric=compatibility_cost(points,torch.arange(6),pred,faces,torch.full((1,6,1),.02),torch.ones(1,6,1,dtype=torch.bool))
    cost.sum().backward()
    assert (points.grad[:,:,2]>0).all()
    assert role.grad is None and surface.grad is None
    torch.testing.assert_close(metric['intent_margin_excess_cm'],torch.tensor([48.]))


def test_unknown_margins_do_not_silently_use_zero():
    pred=NextInteraction(torch.empty(1,36),torch.tensor([0.,10.,0.,0.]).expand(1,6,4),torch.zeros(1,6,1),torch.ones(1))
    faces=torch.tensor([0,0,0,0,0,1,1,0,0,0,1,0,1,1,1.])[None,None]
    cost,metric=compatibility_cost(torch.ones(1,6,3),torch.arange(6),pred,faces,torch.zeros(1,6,1),torch.zeros(1,6,1,dtype=torch.bool))
    assert cost.item()==0 and metric['intent_margin_unknown'].item()==6
