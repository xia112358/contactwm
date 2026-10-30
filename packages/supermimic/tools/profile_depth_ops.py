"""Isolated full-resolution depth timing; no simulation or replay mutation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,time,json
from pathlib import Path
pass
import torch
from supermimic import networks
from supermimic.panorama import PanoramaContext,encoder,encode_depth
networks.AMP_DTYPE=torch.bfloat16
torch.backends.cudnn.benchmark=True
c=PanoramaContext([{'boxes':[]}],oracle=True)
p=torch.zeros(2048,3,device='cuda');p[:,2]=1;p.requires_grad_();R=torch.eye(3,device='cuda').expand(2048,3,3);xi=torch.zeros(2048,52,device='cuda')
def bench(name,fn,n=10):
 for _ in range(3):fn()
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(n):fn()
 torch.cuda.synchronize();print(name,round((time.perf_counter()-t)*1000/n,3),'ms',flush=True)
bench('render_forward_backward',lambda:c.oracle_image((p,R),xi).sum().backward())
x=c.oracle_image((p,R),xi).detach().requires_grad_()
for layout in ['contiguous','channels_last']:
 m=encoder().cuda()
 if layout=='channels_last':m=m.to(memory_format=torch.channels_last);xx=x.detach().to(memory_format=torch.channels_last).requires_grad_()
 else:xx=x
 bench('cnn_'+layout,lambda:encode_depth(m,xx).square().mean().backward())
m=encoder().cuda(); m.compile();bench('cnn_compiled',lambda:encode_depth(m,x).square().mean().backward())
