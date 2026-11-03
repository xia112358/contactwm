"""Validate cropped ray geometry and compiled depth MLP value/gradient parity."""
import sys,copy,json,time
from pathlib import Path
import torch
from supermimic import networks
from supermimic.panorama import PanoramaContext,encoder,encode_depth
networks.AMP_DTYPE=torch.bfloat16
torch.backends.cudnn.benchmark=True
c=PanoramaContext([{'boxes':[]}],oracle=True,depth_layout='lower_band',depth_encoder='mlp')
p=torch.tensor([[0.,0.,1.]],device='cuda',requires_grad=True);r=torch.eye(3,device='cuda')[None];xi=torch.zeros(1,52,device='cuda')
im=c.oracle_image((p,r),xi)
assert im.shape==(1,1,40,128)
d=c.directions_cpu; distance=-1.73/d[:,2];expected=torch.where((distance>=.12)&(distance<=2.5),distance/2.5,0).to('cuda')
torch.testing.assert_close(im.flatten(),expected,rtol=1e-5,atol=1e-6)
im.sum().backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
# Preserve regular azimuth including circular seam; pixel centers above nadir.
assert c.resolution==(40,128) and d.shape==(5120,3)
assert torch.allclose(d.reshape(40,128,3)[:,:,2].std(1),torch.zeros(40),atol=1e-6)
a=encoder('mlp',(40,128)).cuda();b=copy.deepcopy(a);b.compile(dynamic=False)
x=torch.rand(16,1,40,128,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_()
o=encode_depth(a,x);q=encode_depth(b,y)
o.square().mean().backward();q.square().mean().backward()
torch.testing.assert_close(o,q,rtol=.03,atol=.003)
torch.testing.assert_close(x.grad,y.grad,rtol=.1,atol=2e-6)
for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=.1,atol=2e-5)
print('PASS regular 40x128 lower-band floor geometry, pose gradients, compiled depth MLP output/input/parameter gradient parity',flush=True)

from supermimic.panorama import policy_class
pi=policy_class(networks,c)(104,730).cuda()
f=c.policy_features((p.detach(),r),xi);xi[:,34:52]=500
assert torch.equal(f,c.policy_features((p.detach(),r),xi))
obs=torch.zeros(1,10,104,device='cuda');obs[:,:,99:]=f[:,None]
z=pi.perception_features(obs);obs1=obs.clone();obs1[:,-1,101]+=.1
assert not torch.allclose(z,pi.perception_features(obs1),atol=1e-7,rtol=0)
assert c.bank is None and pi.obs_norm.mean.numel()==99
z.square().sum().backward();assert any(v.grad is not None and v.grad.abs().sum()>0 for v in pi.depth_encoder.parameters())
print('PASS MLP actor force isolation, pose-dependent depth and trainable encoder')
