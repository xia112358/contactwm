import torch
from torch import nn
from supermimic import networks
from supermimic.shared_force_wm import shared_force_world_model


class ForceBase(networks.WorldModel):
    def __init__(self,hidden=(32,32)):
        super().__init__(hidden)
        self.contact_head=networks.mlp(self.in_dim,[16,16],18)
        nn.init.zeros_(self.contact_head[-1].weight);nn.init.zeros_(self.contact_head[-1].bias)

    def contact_loss(self,window,normalized_inputs=None):
        return (self.contact_head(normalized_inputs)-window['target']).square().mean()


class Context:
    contact_dim=18
    @staticmethod
    def world_forces(state,force):return force


def test_shared_trunk_preserves_original_initialization_rng_and_keys():
    Shared=shared_force_world_model(ForceBase,Context)
    torch.manual_seed(7);a=ForceBase();a_next=torch.rand(9)
    torch.manual_seed(7);b=Shared(hidden=(32,32));b_next=torch.rand(9)
    assert torch.equal(a_next,b_next)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items() if not k.startswith('contact_head.'))


def test_joint_step_executes_shared_backbone_once_and_preserves_state_integration():
    Shared=shared_force_world_model(ForceBase,Context);model=Shared(hidden=(32,32))
    nn.init.normal_(model.net[-1].weight,std=.01);nn.init.normal_(model.contact_head[-1].weight,std=.01)
    n=3;S=(torch.randn(n,3),torch.eye(3).expand(n,3,3),torch.randn(n,3),torch.randn(n,3),torch.randn(n,29),torch.randn(n,29))
    action=torch.randn(n,29);xi=torch.randn(n,52);normalized=xi[:,:33];calls=[]
    hook=model.net[0].register_forward_hook(lambda *args:calls.append(1))
    state,context=model.joint_step(S,action,xi,normalized);assert len(calls)==1
    separate=model.step(S,action,normalized);force=model.contact_logits(S,action,normalized)
    assert all(torch.equal(x,y) for x,y in zip(state,separate))
    assert torch.equal(context[:,34:52],force);hook.remove()


def test_force_supervision_reaches_shared_hidden_layers_not_state_output_projection():
    Shared=shared_force_world_model(ForceBase,Context);model=Shared(hidden=(32,32))
    nn.init.normal_(model.contact_head[-1].weight,std=.01)
    loss=model.contact_loss({'target':torch.ones(4,18)},normalized_inputs=torch.randn(4,model.in_dim));loss.backward()
    assert model.net[0].weight.grad.norm()>0
    assert model.net[-1].weight.grad is None
