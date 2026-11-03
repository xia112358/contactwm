"""Oracle actor uses current depth, never force or image replay."""
import sys
from pathlib import Path
import torch
from supermimic import networks
from supermimic.panorama import PanoramaContext,policy_class
c=PanoramaContext([{'boxes':[]}],10,oracle=True)
state=(torch.tensor([[0.,0.,1.]],device='cuda'),torch.eye(3,device='cuda')[None]);xi=torch.zeros(1,52,device='cuda')
f=c.policy_features(state,xi);xi[:,34:52]=1000
assert torch.equal(f,c.policy_features(state,xi))
pi=policy_class(networks,c)(104,730).cuda();obs=torch.zeros(1,10,104,device='cuda');obs[:,:,99:]=f[:,None]
z=pi.perception_features(obs);obs1=obs.clone();obs1[:,-1,101]+=.1
z1=pi.perception_features(obs1);assert not torch.allclose(z,z1,atol=1e-7,rtol=0)
assert c.bank is None and pi.obs_norm.mean.numel()==99
# Camera routing must not enter the action decoder directly.
g=torch.zeros(1,730,device='cuda');a=pi.forward_features(obs,g,z)
other=obs.clone();other[:,:,99:]=100
assert torch.equal(a,pi.forward_features(other,g,z))
z.square().sum().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in pi.depth_encoder.parameters())
print('PASS actor force isolation, current-pose depth regeneration, no image bank, camera routing excluded from decoder, trainable CNN')
