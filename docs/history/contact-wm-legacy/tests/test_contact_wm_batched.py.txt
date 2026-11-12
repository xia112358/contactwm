import pytest
import numpy as np
import torch
from torch import nn
from holosoma.contact_wm.batched_learner import BatchedLearner,native_fit_tensor
from holosoma.contact_wm.core import ContactPolicy,ContactGoal,unroll,fit_loss,frozen_parameters
from holosoma.contact_wm.observations import PHPObservation,unpack_history
from holosoma.contact_wm.height_context import HeightRaster
from holosoma.contact_wm.train_policy import CalibratedDynamics
from holosoma.contact_wm.target_response import target_grid


class Geometry(nn.Module):
    def forward(self,s):return s[:,13:31].reshape(-1,6,3)

class Dynamics(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.001))
    def source_step(self,s,a,physics,height,force):
        state=torch.cat((s[:,:13],s[:,13:42]+a*self.scale,s[:,42:]),-1)
        logits=a[:,:6]
        return dict(state=state,contact_logits=logits,contact=logits.sigmoid(),force_w=force+a[:,:18].reshape(-1,6,3))


@pytest.mark.parametrize('target_scale',[.05,.02])
def test_mixed_deadlines_equal_serial_losses_and_gradients(tmp_path,target_scale):
    torch.manual_seed(13)
    prior=nn.Sequential(nn.Linear(567,8),nn.ELU(),nn.Linear(8,29));prior.requires_grad_(False)
    policy=ContactPolicy(prior).requires_grad_(True);wm=Dynamics();geom=Geometry()
    xyz=torch.zeros(1,6,3);normal=xyz.clone();normal[:,:,2]=1
    new=torch.zeros(1,6,dtype=torch.bool);new[:,2]=True
    maintain=torch.zeros_like(new);maintain[:,3:5]=True
    goal=ContactGoal(xyz,normal,new,maintain,xyz,torch.ones(1,6),26,10)
    states=np.zeros((300,71),np.float32);states[:,6]=1;states[:,2]=1
    height=torch.zeros(1,121,121);minimum=torch.tensor([[-3.,-3.]])
    atlas=tmp_path/'atlas.npz';np.savez(atlas,height=height.numpy(),minimum=minimum.numpy(),resolution=.05)
    task=dict(policy=policy,prior=prior,wm=wm,geometry=geom,meta=dict(default_joint_pos=[0.]*29),
        hint=torch.zeros(1,3),goal=goal,base_target=xyz,z={'state':states},target_tangent_scale=target_scale)
    b=BatchedLearner(task,atlas,compile_kernels=False)
    sample=dict(state=torch.tensor(states[266:268]),observation=torch.zeros(2,567),previous_known=torch.ones(2,6,dtype=torch.bool),
        previous_force_w=torch.full((2,6,3),10.),support=xyz.expand(2,-1,-1),age=torch.tensor([5,9]))
    serial=[]
    for i,age in enumerate([5,9]):
        expand=lambda x:x[i:i+1].expand(9,*x.shape[1:])
        g=ContactGoal(xyz.expand(9,-1,-1)+new[...,None]*target_grid(.04)[:,None],normal.expand(9,-1,-1),
            new.expand(9,-1),maintain.expand(9,-1),xyz.expand(9,-1,-1),torch.ones(9,6),26-(age-5),10)
        reader=HeightRaster(height.expand(9,-1,-1),minimum.expand(9,-1),.05,php=True)
        obs=PHPObservation(torch.zeros(29),reader,torch.zeros(9,3),torch.ones(9,6,dtype=torch.bool))
        result=unroll(policy,CalibratedDynamics(wm),expand(sample['state']),expand(sample['observation'])[:,:6],
            unpack_history(expand(sample['observation'])[:,102:]),torch.zeros(9,29),g,horizon=g.arrival_step+9,
            observation=obs,geometry=geom,prior=prior,target_tangent_scale=target_scale,stability=lambda s,p,g:s.sum()*0,
            context=lambda s,c,p:dict(physics=torch.zeros(9,33),height=torch.zeros(9,121),force=expand(sample['previous_force_w']) if p is None else p['force_w']))
        serial.append(result['losses']['loss']+(.01-.1)*result['losses']['motion'])
    old=torch.stack(serial).mean();old.backward();before=torch.cat([p.grad.flatten() for p in policy.parameters()]);policy.zero_grad()
    with frozen_parameters(wm):
        values,valid=b.policy_loss(sample);values[0].backward()
    after=torch.cat([p.grad.flatten() for p in policy.parameters()])
    assert valid
    torch.testing.assert_close(old,values[0],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(before,after,rtol=3e-4,atol=2e-5)
    assert wm.scale.grad is None

    # A migrating placement point must not redefine material slip or supports.
    class Surface(nn.Module):
        def forward(self,state,normal,plane):return state[:,:3]+normal*.01
    b.hand_surfaces['2']=Surface()
    with torch.no_grad():surface_values,surface_valid=b.policy_loss(sample)
    assert surface_valid
    torch.testing.assert_close(surface_values[2:8],values.detach()[2:8])
    torch.testing.assert_close(surface_values[9],values.detach()[9])
    assert not torch.allclose(surface_values[8],values.detach()[8])
    b.hand_surfaces.clear()

    # Explicit planning disables goal perturbations in batched BPTT too.
    original_offsets=b.offsets.clone();b.offsets.zero_()
    planned=b.prepare(sample)['target']
    torch.testing.assert_close(planned,b.target.expand_as(planned))
    b.offsets.copy_(original_offsets)

    # The same contact objective must train the depth encoder through frozen WM.
    from holosoma.contact_wm.panorama_policy import PanoramaContactPolicy
    policy.zero_grad();wm.zero_grad()
    b.policy=PanoramaContactPolicy(policy)
    b.use_panorama=True
    nn.init.normal_(b.policy.encoder[-1].weight,std=.02)
    sample['panorama']=torch.rand(2,1,64,128)
    sample['pano_view']=torch.tensor([[0.,0.,1.73,0.]]).expand(2,-1)
    with frozen_parameters(wm):
        values,valid,_=b.policy_loss(sample)
        values[0].backward()
    assert valid and len(values)==len(b.names)==10
    assert b.policy.encoder[0].conv.weight.grad.abs().sum()>0
    assert wm.scale.grad is None
    with torch.no_grad():
        altered={**sample,'panorama':torch.zeros_like(sample['panorama'])}
        changed,_,_=b.policy_loss(altered)
    assert not torch.allclose(values[0],changed[0],rtol=1e-7,atol=1e-7)


def test_tensor_fit_matches_native_loss():
    torch.manual_seed(3);state=torch.randn(7,71);target=torch.randn(7,71)
    p=dict(state=state,contact_logits=torch.randn(7,6),force_w=torch.randn(7,6,3))
    active=torch.rand(7,6)>.5;force=torch.randn(7,6,3);known=torch.rand(7,6)>.3;valid=torch.tensor([1,1,0,1,0,1,1],dtype=torch.bool)
    force[~known]=float('nan')
    torch.testing.assert_close(native_fit_tensor(p,target,active,force,known,valid),fit_loss(p,target,active,force,known,valid))
