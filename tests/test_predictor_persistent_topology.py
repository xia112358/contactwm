import types
import torch
from generator.next_interaction_surface import contact_terms
from generator.next_interaction import NextInteraction
import contact_solver.newton_witness_loss as witness


class FK:
    def __call__(self,q):
        positions=q[...,None,:3].expand(*q.shape[:-1],7,3)
        rotation=q.new_tensor([1,0,0,0,1,0]).expand(*q.shape[:-1],7,6)
        return positions,rotation


def setup(monkeypatch,missing=False):
    q=torch.zeros((1,36),requires_grad=True)
    role=torch.tensor([[2,2,0,0,0,0]]);surface=torch.zeros((1,6),dtype=torch.long)
    pred=NextInteraction(q,torch.nn.functional.one_hot(role,4).float()*20,
        torch.nn.functional.one_hot(surface,2).float()*20,torch.ones(1))
    target=dict(role=role,surface=surface,anchor=torch.ones((1,6,3)),
        material_local=torch.zeros((1,6,3)),start_material_local=torch.zeros((1,6,3)),
        start_anchor=torch.zeros((1,6,3)))
    pairs=[(dict(part=p,surface=0,includemargin=.02,dist=.01,position_w=[0,0,0],constraint_active=True,allocated=True),q.sum()*0+.01)
           for p in ([1] if missing else [0,1])]
    monkeypatch.setattr(witness,'query_local_distances',lambda *a,**kw:([pairs],dict(
        surface_attribution_schema='newton_source_triangle_witness_normal_fan_v2',
        full_robot_separation=[dict(worst_terrain=None,worst_self=None)],pairs=[[p for p,_ in pairs]],
        surface_catalog=[dict(surface=0,normal_w=[0,0,1])])) )
    return types.SimpleNamespace(fk=FK()),pred,target


def test_material_motion_is_diagnostic_not_penalty(monkeypatch):
    model,pred,target=setup(monkeypatch)
    cost,consistency,metrics=contact_terms(model,pred,target,None)
    moved=dict(target,start_anchor=torch.ones((1,6,3))*10)
    cost2,consistency2,metrics2=contact_terms(model,pred,moved,None)
    torch.testing.assert_close(cost,cost2);torch.testing.assert_close(consistency,consistency2)
    assert metrics2['start_material_motion_cm'].item()>metrics['start_material_motion_cm'].item()
    assert cost.item()==0 and metrics2['persistent_contact_loss'].item()==0


def test_missing_persistent_contact_still_has_approach_loss(monkeypatch):
    model,pred,target=setup(monkeypatch,missing=True)
    cost,consistency,metrics=contact_terms(model,pred,target,None)
    assert cost.item()>0 and consistency.item()==0
    assert metrics['persistent_contact_loss'].item()>0
    assert metrics['newton_unrealized_target_contacts'].item()==1


def test_missing_anchor_gradient_is_not_duplicated_in_consistency(monkeypatch):
    model,pred,target=setup(monkeypatch,missing=True)
    cost,consistency,_=contact_terms(model,pred,target,None)
    contact_grad=torch.autograd.grad(cost.sum(),pred.qpos,retain_graph=True)[0]
    consistency_grad=torch.autograd.grad(consistency.sum(),pred.qpos)[0]
    assert contact_grad.abs().sum()>0
    torch.testing.assert_close(consistency_grad,torch.zeros_like(consistency_grad))
    moved=dict(target,anchor=target['anchor']*2)
    cost2,consistency2,_=contact_terms(model,pred,moved,None)
    assert cost2.item()>cost.item()
    torch.testing.assert_close(consistency2,consistency)
