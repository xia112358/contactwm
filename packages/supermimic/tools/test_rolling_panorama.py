"""Rolling geometry, bounded history and CPU-image replay checks on GPU."""
import sys
from pathlib import Path
import torch
import warp as wp
from supermimic.rolling_panorama import RollingPanorama
from supermimic.panorama import ImageBank
wp.init();device='cuda:0'
m=RollingPanorama(2,device)
o=torch.tensor([[0.,0.,1.],[0.,0.,1.]],device=device);yaw=torch.zeros(2,device=device)
p=torch.tensor([[[1.,0.,1.],[2.,0.,1.]],[[0.,0.,0.],[0.,0.,0.]]],device=device)
v=torch.tensor([[True,True],[False,False]],device=device)
r=m.update(p,v,o,yaw,0.)
assert r['valid'][0].sum()==1 and r['valid'][1].sum()==0
assert abs(r['depth'][0,32,64].item()-1)<1e-6
# A pure viewpoint change must reproject old *world points*, not shift pixels.
yaw[0]=torch.pi/2;o[0,0]=.2
r=m.update(p,torch.zeros_like(v),o,yaw,.1)
assert abs(r['depth'][0,32,32].item()-.8)<1e-6
assert abs(r['age'][0,32,32].item()-.1)<1e-6
# Keep the exact world point after repeated reprojection (no pixel quantization drift).
for i in range(2,12):
 yaw[0]=i*.1;r=m.update(p,torch.zeros_like(v),o,yaw,i*.1)
 assert r['valid'][0].sum()==1
 torch.testing.assert_close(m.points[0][m.stamps[0]>-1e10][0],p[0,0])
r=m.update(p,torch.zeros_like(v),o,yaw,16.)
assert not r['valid'].any()
m.reset(torch.tensor([0,1],device=device));r=m.update(p,v,o,yaw,0.)
assert r['valid'][0].sum()==1
m.reset(torch.tensor([0],device=device));r=m.update(p,torch.zeros_like(v),o,yaw,.1)
assert not r['valid'].any()
# CPU replay stores historical images; GPU keeps only the newest frame.
b=ImageBank(10,2,device,cpu_replay=True)
old=b.add(torch.zeros(2,1,64,128,device=device));b.add(torch.ones(2,1,64,128,device=device))
assert b.images.device.type=='cpu' and b.images.is_pinned()
g=torch.tensor([old,b.generation,-1],device=device);e=torch.tensor([0,1,0],device=device)
x=b.gather(g,e);assert x[0].sum()==0 and (x[1]==1).all() and x[2].sum()==0
for _ in range(b.capacity):b.add(torch.ones(2,1,64,128,device=device))
try:b.gather(torch.tensor([old],device=device),torch.tensor([0],device=device))
except RuntimeError:pass
else:raise AssertionError('Expired CPU replay image accepted')
print('PASS reprojection, nearest-surface occlusion, exact-point retention, timestamps/expiry, reset isolation, pinned CPU historical replay and expiry guard')
print('Rolling map GiB at 4531 worlds:',m.gpu_bytes/2*4531/2**30)
