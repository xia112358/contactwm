import pytest
import torch
from contact_solver.strict_nonpenetration import generation_gate,priority_loss


def report(terrain=0.,self_depth=0.):
    return dict(schema='newton_full_robot_signed_separation_v1',terrain_rows=1,self_rows=1,
        terrain_penetration_m=terrain,self_penetration_m=self_depth,
        worst_terrain={'dist':-terrain} if terrain else None,
        worst_self={'dist':-self_depth} if self_depth else None)


def test_even_tiny_penetration_fails_and_unknown_never_passes():
    gate=generation_gate([report(),report(1e-12),report(self_depth=.01),{},report(float('nan'))])
    assert gate['statuses']==['clear','penetrating','penetrating','unknown','unknown']
    assert gate['collision_pass']==[True,False,False,False,False]
    assert not gate['all_frames_collision_pass']
    assert not generation_gate([])['all_frames_collision_pass']


def test_task_cannot_cancel_penetration_gradient():
    depth=torch.tensor([.001,0.],requires_grad=True)
    task=-1000000*depth
    loss,mode,_=priority_loss(task,depth,[report(.001),report()])
    loss.backward()
    assert mode=='feasibility' and depth.grad[0]>0


def test_clear_batch_can_optimize_task():
    task=torch.tensor([3.,4.],requires_grad=True)
    loss,mode,_=priority_loss(task,torch.zeros(2),[report(),report()])
    loss.backward()
    assert mode=='task'
    torch.testing.assert_close(task.grad,torch.tensor([.5,.5]))


def test_unknown_is_not_a_training_success():
    with pytest.raises(ValueError,match='Unknown'):
        priority_loss(torch.ones(1),torch.zeros(1),[{}])
    incomplete=report();incomplete.pop('worst_self')
    assert generation_gate([incomplete])['statuses']==['unknown']
    inconsistent=report(.01);inconsistent['worst_terrain']['dist']=-.02
    assert generation_gate([inconsistent])['statuses']==['unknown']
