import torch
from torch import nn
from holosoma.contact_wm.core import ContactPolicy
from holosoma.contact_wm.panorama_policy import PanoramaContactPolicy,PanoramaReplay,pack_panorama,relative_view


def test_actor_ignores_privileged_height_and_reads_only_depth():
    torch.manual_seed(5)
    base=ContactPolicy(nn.Sequential(nn.Linear(567,32),nn.ELU(),nn.Linear(32,29)))
    model=PanoramaContactPolicy(base)
    nn.init.normal_(model.encoder[-1].weight,std=.02)
    obs=torch.randn(2,567);extra=torch.randn(2,25);state=torch.zeros(2,71);state[:,6]=1
    view=torch.tensor([[0.,0.,.73,0.]]).expand(2,-1)
    depth=torch.rand(2,1,64,128)*.8+.1
    a=model(obs,extra,panorama=depth,state=state,view_pose=view)
    altered=obs.clone();altered[:,39:102]=float('nan')
    b=model(altered,extra,panorama=depth,state=state,view_pose=view)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    c=model(obs,extra,panorama=torch.zeros_like(depth),state=state,view_pose=view)
    assert not torch.allclose(a,c)
    a.square().mean().backward()
    assert model.encoder[0].conv.weight.grad.abs().sum()>0


def test_fusion_metadata_is_not_a_policy_channel():
    result=dict(depth=torch.ones(2,64,128),valid=torch.ones(2,64,128,dtype=torch.bool),age=torch.zeros(2,64,128),confidence=torch.ones(2,64,128))
    a=pack_panorama(result)
    result['age'].fill_(14);result['confidence'].fill_(.01)
    torch.testing.assert_close(a,pack_panorama(result))
    assert a.shape==(2,1,64,128)
    result['valid'].zero_();result['depth'].fill_(float('nan'))
    assert (pack_panorama(result)==0).all()


def test_panorama_replay_does_not_substitute_newer_images():
    r=PanoramaReplay(2,1,'cpu')
    image=torch.ones(1,1,64,128)
    r.add(image);r.add(image*2)
    assert (r.gather(torch.tensor([0]),torch.tensor([0]))==1).all()
    r.add(image*3)
    import pytest
    with pytest.raises(RuntimeError,match='expired'):r.gather(torch.tensor([0]),torch.tensor([0]))


def test_relative_reading_pose_tracks_view_yaw_and_position():
    state=torch.zeros(1,71);state[:,6]=1
    view=torch.tensor([[1.,2.,.73,torch.pi/2]])
    torch.testing.assert_close(relative_view(state,view),torch.tensor([[1.,2.,.73,1.,0.]]),atol=1e-6,rtol=0)
