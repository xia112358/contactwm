"""Measure the GPU geometry component only, separate from WM/BPTT and simulation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--collision-model',type=Path,required=True)
    p.add_argument('--samples',type=int,nargs='+',default=[128,512])
    p.add_argument('--batch',type=int,nargs='+',default=[256,1024,4096])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    from supermimic.contact_control import ContactGeometry
    from supermimic.contact_surface import SurfaceRegionObjective
    from supermimic.mdp import unflatten_state
    recording=torch.load(a.recording,map_location='cuda:0',weights_only=False)
    data=recording['data'];profile=recording['metadata'].get('collision_profile','legacy_v1')
    geometry=ContactGeometry(data,'cuda:0');results=[]
    for samples in a.samples:
        surface=SurfaceRegionObjective(a.collision_model,geometry,'cuda:0',samples_per_part=samples,
            target_normal=[0.,0.,1.],collision_profile=profile)
        for batch in a.batch:
            index=torch.arange(batch,device='cuda:0')%len(data['s1'])
            state=data['s1'][index].detach().clone().requires_grad_(True)
            target=data['point'][index].detach()
            def measure():
                cost=surface.region_loss(unflatten_state(state),target,list(range(6)))['loss'].mean()
                return torch.autograd.grad(cost,state)[0]
            for _ in range(2):measure()
            torch.cuda.synchronize();before=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for _ in range(a.repeats):gradient=measure()
            torch.cuda.synchronize()
            results.append(dict(samples=samples,batch=batch,
                forward_backward_ms=(time.perf_counter()-start)*1000/a.repeats,
                extra_peak_mib=(torch.cuda.max_memory_allocated()-before)/2**20,
                gradients_finite=bool(torch.isfinite(gradient).all())))
        del surface
    report=dict(schema='surface_objective_gpu_component_benchmark_v1',gpu=torch.cuda.get_device_name(0),
        scope='FP32 geometry forward/backward to all 71 state components, all six parts; no WM/BPTT/simulator memory included',
        collision_profile=profile,results=results)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
