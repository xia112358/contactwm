"""GPU checks for full-depth gradients, causal replay and actor information isolation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys
from pathlib import Path
pass
import torch
from supermimic.panorama import PanoramaContext,OracleDepth,ImageBank,policy_class
from supermimic import networks

dev='cuda:0';context=PanoramaContext([{'boxes':[[1.,0.,1.,1.,0.,0.,0.,.3,.8,1.5]]}],10)
boxes,dirs=context.tensors(dev)
o=torch.tensor([[0.,0.,1.]],device=dev,requires_grad=True);a=torch.tensor([.08],device=dev,requires_grad=True);task=torch.zeros(1,device=dev)
# Smooth interior ray, away from a silhouette: compare position and yaw gradients.
image=OracleDepth.apply(o,a,task,dirs,boxes);iy,ix=31,63
loss=image[0,0,iy,ix];loss.backward();eps=1e-4
for axis in [0,1,2,3]:
 shift=torch.zeros_like(o);da=torch.zeros_like(a)
 if axis<3:shift[0,axis]=eps
 else:da[0]=eps
 hi=OracleDepth.apply(o.detach()+shift,a.detach()+da,task,dirs,boxes)[0,0,iy,ix]
 lo=OracleDepth.apply(o.detach()-shift,a.detach()-da,task,dirs,boxes)[0,0,iy,ix]
 numerical=((hi-lo)/(2*eps)).item();actual=(o.grad[0,axis] if axis<3 else a.grad[0]).item()
 assert abs(actual-numerical)<.002,(axis,actual,numerical)
# Fixed camera memory changes relative geometry as the imagined body moves.
S=(torch.tensor([[0.,0.,1.]],device=dev),torch.eye(3,device=dev)[None]);xi=torch.zeros(1,58,device=dev);xi[:,52:56]=torch.tensor([0,0,1.73,0.],device=dev);xi[:,56]=0
f=context.policy_features(S,xi);changed=xi.clone();changed[:,34:52]=100
assert torch.equal(f,context.policy_features(S,changed)), 'Force leaked into actor observation'
moved=(S[0]+torch.tensor([[.1,0,0]],device=dev),S[1]);assert not torch.equal(f,context.policy_features(moved,xi))
# A replay reference returns the historical image, never the current image.
bank=ImageBank(10,1,dev);context.bank=bank
first=bank.add(torch.zeros(1,1,64,128,device=dev));bank.add(torch.ones(1,1,64,128,device=dev))
assert bank.gather(torch.tensor([first],device=dev),torch.tensor([0],device=dev)).sum()==0
assert bank.gather(torch.tensor([-1],device=dev),torch.tensor([0],device=dev)).sum()==0
for _ in range(bank.capacity):bank.add(torch.ones(1,1,64,128,device=dev))
try:bank.gather(torch.tensor([first],device=dev),torch.tensor([0],device=dev))
except RuntimeError:pass
else:raise AssertionError('Expired replay image accepted')
# Routing IDs do not enter the decoder or normalizer; CNN receives image pixels.
pi=policy_class(networks,context)(106,730).to(dev)
obs=torch.zeros(1,10,106,device=dev);obs[:,:,104]=bank.generation
features=pi.perception_features(obs);goal=torch.zeros(1,730,device=dev)
y=pi.forward_features(obs,goal,features);other=obs.clone();other[:,:,104:]=123456
assert torch.equal(y,pi.forward_features(other,goal,features))
y.square().sum().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in pi.depth_encoder.parameters())
print('PASS box depth pose gradients, no actor force leak, moving relative view, immutable replay/expiry, routing isolation, trainable full-resolution actor CNN')
