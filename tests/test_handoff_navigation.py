import importlib.util
from pathlib import Path
import torch
import pytest

PATH=Path(__file__).resolve().parents[1]/'packages/supermimic/supermimic/handoff/navigation_objective.py'
spec=importlib.util.spec_from_file_location('handoff_navigation_test',PATH)
nav=importlib.util.module_from_spec(spec);spec.loader.exec_module(nav)
BOX=[[0,0,.5,1,0,0,0,1,1,1]]

@pytest.mark.parametrize('goal',[[1.,0.,0.],[.5,0.,.3],[0.,0.,1.],[-1.,0.,.7]])
def test_original_targets_unchanged_and_zero_at_valid_goal(goal):
    target=torch.tensor([goal]);metric=nav.PointNavigation(BOX,target)
    assert torch.equal(metric.goals,target)
    assert torch.allclose(metric.distance(target),torch.zeros(1),atol=1e-6)

def test_direct_distance_and_gradient_without_obstacle():
    target=torch.tensor([[1.,2.,3.]])
    metric=nav.PointNavigation([],target)
    point=torch.zeros(1,1,3,requires_grad=True)
    loss=metric.distance(point).square().sum();loss.backward()
    assert torch.allclose(loss,torch.tensor(14.))
    assert torch.allclose(point.grad,-2*target[None])

def test_blocked_route_is_longer_and_has_finite_gradient():
    metric=nav.PointNavigation(BOX,torch.tensor([[1.,0.,.3]]))
    point=torch.tensor([[[-1.,.1,.3]]],requires_grad=True)
    distance=metric.distance(point)
    assert distance.item()>(point.detach()-metric.goals).norm().item()
    distance.sum().backward();assert torch.isfinite(point.grad).all() and point.grad.norm()>0

def test_penetrating_start_has_finite_escape_gradient():
    metric=nav.PointNavigation(BOX,torch.tensor([[1.,0.,.3]]))
    point=torch.tensor([[[.1,.1,.4]]],requires_grad=True)
    distance=metric.distance(point);distance.sum().backward()
    assert torch.isfinite(distance).all() and torch.isfinite(point.grad).all()

def test_ground_goal_is_not_raised_to_box_top():
    target=torch.tensor([[1.,0.,0.]])
    metric=nav.PointNavigation(BOX,target)
    point=torch.tensor([[[1.2,0.,0.]]])
    assert torch.allclose(metric.distance(point),torch.tensor([[.2]]),atol=1e-6)
    assert metric.goals[0,2]==0

@pytest.mark.parametrize('observed',[False,True])
def test_all_six_parts_use_original_paired_goals_and_backpropagate(observed):
    from types import SimpleNamespace as NS
    B=2;K=4
    local=torch.zeros(1,6,K,3)
    goals=torch.zeros_like(local);goals[...,0]=1.
    target=torch.zeros(1,6,3);target[...,0]=1.
    clock=NS(event_ends=torch.tensor([1]),segment_motion=torch.tensor([0]),
        target_masks=torch.ones(1,6,dtype=torch.bool),keep_masks=torch.zeros(1,6,dtype=torch.bool),
        data={'point':target},material_local=local,material_targets=goals,
        paired_material_local=local,paired_material_targets=goals,
        paired_material_valid=torch.ones(1,6,K,dtype=torch.bool))
    surface=NS(atlas=NS(points_local=torch.zeros(6,K,3),areas=torch.ones(6,K)),temperature=.02)
    objective=nav.ContactNavigationObjective(clock,surface,NS(tasks=[{'boxes':[]}]))
    p=torch.zeros(B,6,3,requires_grad=True);R=torch.eye(3).expand(B,6,3,3)
    stable=torch.full((B,6),observed);evaluate=torch.ones_like(stable)
    data={'contact_loss_point':torch.zeros(1,B,6,3),'contact_loss_position':torch.zeros(1,B,6,3),
        'contact_loss_rotation':R[None],'contact_loss_candidate':torch.zeros(1,B,6,dtype=torch.long),
        'contact_raw_stable':torch.zeros(1,B,6,dtype=torch.bool),
        'contact_raw_point':torch.zeros(1,B,6,3),'contact_raw_position':torch.zeros(1,B,6,3),'contact_raw_rotation':R[None]}
    swap=lambda x,y:x+(y-x).detach()
    cost=objective.cost(p,R,torch.zeros(B,dtype=torch.long),evaluate,stable,~stable,data,0,swap)
    assert torch.allclose(cost,torch.full((B,6),400.),atol=1e-3)
    cost.sum().backward()
    assert torch.isfinite(p.grad).all() and (p.grad[...,0]<0).all()
    assert torch.equal(clock.material_targets,goals)
    assert torch.equal(clock.data['point'],target)
