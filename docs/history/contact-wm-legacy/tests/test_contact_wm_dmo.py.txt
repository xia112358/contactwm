"""The nine-target adapter keeps native values and model/actor derivatives."""
import torch
import pytest
from holosoma.contact_wm.dmo import native_policy_loss


class Learner:
    def __init__(self):
        self.weight=torch.tensor(0.,requires_grad=True)

    def _imag_step(self,state,contact,history,previous,force,*args,native):
        action=self.weight.expand_as(previous)
        native['action_error']=(action.detach()-native['action']).abs().max()
        prediction=state+action[:,:1]
        return prediction,contact,history,action,force,contact,action,torch.ones(len(state),dtype=torch.bool)

    def _objective(self,x,states,actions,forces,logits,priors,valid,native):
        self.states=states.detach().clone()
        self.live=native['live']
        return states[-1,:,0].mean()[None],valid.all()


def batch():
    T,B=3,9
    return dict(state=torch.zeros(T,B,71),next_state=torch.arange(T).float()[:,None,None].expand(T,B,71),
                observation=torch.zeros(T,B,567),previous_active=torch.zeros(T,B,6,dtype=torch.bool),
                previous_known=torch.ones(T,B,6,dtype=torch.bool),previous_force_w=torch.zeros(T,B,6,3),
                support=torch.zeros(T,B,6,3),age=torch.arange(5,5+T)[:,None].expand(T,B),
                ended=torch.zeros(T,B,dtype=torch.bool),action=torch.zeros(T,B,29),
                active=torch.zeros(T,B,6,dtype=torch.bool),force_w=torch.zeros(T,B,6,3),
                valid=torch.ones(T,B,dtype=torch.bool))


def test_native_values_model_gradients_and_terminal_mask():
    learner=Learner();data=batch();data['ended'][1,0]=True
    values,ok,metrics=native_policy_loss(learner,data,torch.zeros(9,6,3))
    assert ok and torch.equal(learner.states,data['next_state'])
    assert not learner.live[2,0] and learner.live[1,0]
    values[0].backward()
    assert torch.isclose(learner.weight.grad,torch.tensor(3.))
    assert metrics['dmo_attempts']==9 and metrics['dmo_complete_attempts']==8


def test_stale_actor_is_not_used_for_dmo():
    learner=Learner();data=batch();data['action']+=.2
    with pytest.raises(RuntimeError,match='action replay mismatch'):
        native_policy_loss(learner,data,torch.zeros(9,6,3))
