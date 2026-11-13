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


def test_supervised_fit_reuses_trunk_and_matches_separate_losses_and_gradients():
    import copy
    Shared=shared_force_world_model(ForceBase,Context)
    model=Shared(hidden=(32,32))
    nn.init.normal_(model.net[-1].weight,std=.01)
    nn.init.normal_(model.contact_head[-1].weight,std=.01)
    reference=copy.deepcopy(model)
    inputs=torch.randn(5,model.in_dim)
    window={'target':torch.randn(5,1,18)}
    calls=[]
    hook=model.net[0].register_forward_hook(lambda *args:calls.append(1))
    state,contact=model.training_predictions(window,inputs)
    assert len(calls)==1
    hook.remove()
    expected_state=reference.net(inputs)
    expected_contact=reference.contact_loss(window,normalized_inputs=inputs[:,None])
    assert torch.equal(state,expected_state)
    assert torch.equal(contact,expected_contact)
    (state.square().mean()+.1*contact).backward()
    (expected_state.square().mean()+.1*expected_contact).backward()
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)


def test_dmo_one_step_fit_uses_shared_force_objective_and_one_trunk_pass():
    from supermimic.contact_control import wm_one_step_loss
    from supermimic.mdp import flatten_state,unflatten_state
    Shared=shared_force_world_model(ForceBase,Context)
    model=Shared(hidden=(32,32))
    nn.init.normal_(model.contact_head[-1].weight,std=.01)
    n=4
    S=(torch.randn(n,3),torch.eye(3).expand(n,3,3),torch.randn(n,3),torch.randn(n,3),torch.randn(n,29),torch.randn(n,29))
    s=flatten_state(S)[:,None]
    w=dict(s=s,s1=s.clone(),a=torch.randn(n,1,29),xi=torch.randn(n,1,52),
           next_xi=torch.randn(n,1,52),mask=torch.ones(n,1),target=torch.randn(n,1,18))
    normalize=lambda x:x[...,:33]
    calls=[]
    hook=model.net[0].register_forward_hook(lambda *args:calls.append(1))
    loss=wm_one_step_loss(model,w,normalize)
    assert len(calls)==1
    hook.remove()
    state=unflatten_state(s[:,0])
    inputs=model.in_norm(model.inputs(state,w['a'][:,0],normalize(w['xi'][:,0])))
    predicted=model.net(inputs)*model.out_norm.std()+model.out_norm.mean
    # Identical source/target state gives zero state increment; out_norm.std=1.
    expected=predicted.square().mean()+.1*model.contact_loss(w,normalized_inputs=inputs[:,None])
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert model.net[0].weight.grad.norm()>0
    assert model.contact_head[-1].weight.grad.norm()>0


def test_legacy_transfer_preserves_state_function_and_normalization():
    from supermimic.shared_force_wm import load_force_wm_state
    legacy=ForceBase()
    nn.init.normal_(legacy.net[-1].weight,std=.01)
    legacy.in_norm.update(torch.randn(12,legacy.in_dim)+3)
    legacy.out_norm.update(torch.randn(12,networks.WM_OUT)+2)
    source={k:v.clone() for k,v in legacy.state_dict().items()}
    shared=shared_force_world_model(ForceBase,Context)(hidden=(32,32))
    changed=load_force_wm_state(shared,source,source_architecture='independent',target_architecture='shared')
    assert changed
    assert all(torch.equal(v,shared.state_dict()[k]) for k,v in source.items() if not k.startswith('contact_head.'))
    assert all(torch.count_nonzero(p)==0 for p in shared.contact_head.parameters())
    inputs=torch.randn(7,legacy.in_dim)
    assert torch.equal(legacy.net(inputs),shared.net(inputs))
    assert all(torch.equal(v,source[k]) for k,v in legacy.state_dict().items())


def test_shared_resume_preserves_trained_force_weights():
    from supermimic.shared_force_wm import load_force_wm_state
    Shared=shared_force_world_model(ForceBase,Context)
    source=Shared(hidden=(32,32))
    nn.init.normal_(source.contact_head[-1].weight,std=.1)
    target=Shared(hidden=(32,32))
    assert not load_force_wm_state(target,source.state_dict(),source_architecture='shared',target_architecture='shared')
    assert all(torch.equal(v,target.state_dict()[k]) for k,v in source.state_dict().items())


def test_legacy_transfer_rejects_missing_trunk_and_reverse_architecture():
    import pytest
    from supermimic.shared_force_wm import load_force_wm_state
    legacy=ForceBase()
    shared=shared_force_world_model(ForceBase,Context)(hidden=(32,32))
    source=dict(legacy.state_dict());del source['net.0.weight']
    with pytest.raises(RuntimeError,match='Missing key'):
        load_force_wm_state(shared,source,source_architecture='independent',target_architecture='shared')
    with pytest.raises(ValueError,match='Unsupported WM architecture transfer'):
        load_force_wm_state(legacy,shared.state_dict(),source_architecture='shared',target_architecture='independent')
