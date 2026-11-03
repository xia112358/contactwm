"""Fixed-state collision/loss ablation; never new contact labels or execution.

The surface-v2 model is compiled in RAM from the authoritative USD by Newton.
An in-memory file fixture exposes those exact bytes and their real fingerprint
to the production objective loader; no runtime artifact or alternate URDF is
created. The fixture changes I/O only, not geometry, losses or derivatives.
"""
from __future__ import annotations
from supermimic.paths import runtime_path

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def build_surface_model():
    import newton
    from newton.solvers import SolverMuJoCo
    from pxr import Usd,UsdPhysics
    from somaforge_core.robot_assets import canonical_g1_asset_metadata
    from somaforge_core.collision_profiles import audit_surface_stage,audit_surface_mujoco
    asset=canonical_g1_asset_metadata(collision_profile='surface_v2')
    stage=Usd.Stage.Open(asset['usd_path']);stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection('mujoco')
    audit_surface_stage(stage,stage.GetDefaultPrim())
    UsdPhysics.Joint(stage.GetPrimAtPath('/g1_29dof/Physics/root_joint')).GetJointEnabledAttr().Set(False)
    builder=newton.ModelBuilder();builder.add_ground_plane();builder.begin_world()
    builder.add_usd(stage,floating=True,load_visual_shapes=False,collapse_fixed_joints=False)
    builder.end_world()
    solver=SolverMuJoCo(builder.finalize('cpu'),use_mujoco_cpu=True)
    return solver.mj_model,asset,audit_surface_mujoco(solver.mj_model)


@contextmanager
def memory_model_fixture(model,asset):
    """Use the production loader and all validation without writing runtime files."""
    import mujoco
    path=runtime_path('tmp/__in_memory_surface_v2.mjb')
    sidecar=path.with_suffix('.mjb.json')
    binary=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,None,binary)
    raw=binary.tobytes()
    meta=json.dumps(dict(schema='somaforge_solver_collision_model_v1',robot_asset=asset,
        model_sha256=hashlib.sha256(raw).hexdigest(),source='cpu_newton_authority_geometry_check'))
    original_bytes,original_text,original_file=Path.read_bytes,Path.read_text,Path.is_file
    original_model=mujoco.MjModel
    def read_bytes(p,*args,**kw):return raw if p==path else original_bytes(p,*args,**kw)
    def read_text(p,*args,**kw):return meta if p==sidecar else original_text(p,*args,**kw)
    def is_file(p,*args,**kw):return True if p in (path,sidecar) else original_file(p,*args,**kw)
    def from_binary(p):return model if str(p)==str(path) else original_model.from_binary_path(str(p))
    with patch.object(Path,'read_bytes',read_bytes),patch.object(Path,'read_text',read_text),\
         patch.object(Path,'is_file',is_file),patch.object(mujoco,'MjModel',SimpleNamespace(from_binary_path=from_binary)):
        yield path


def metrics(values,gradient,valid):
    pair=valid[1:]&valid[:-1]
    def stats(v):
        if not v.numel():return None
        return dict(mean=float(v.mean()),p95=float(torch.quantile(v,.95)),max=float(v.max()))
    dg=(gradient[1:]-gradient[:-1]).norm(dim=-1)
    answer=dict(samples=int(valid.sum()),pairs=int(pair.sum()),mean_loss=float(values[valid].mean()),
        adjacent_loss_delta=stats((values[1:]-values[:-1]).abs()[pair]),
        adjacent_root_gradient_delta=stats(dg[pair]))
    return answer


def score(surface,geometry,data,column,segment,clock,demo,normal,mode):
    from supermimic.mdp import unflatten_state
    from somaforge_core.contact_system.objectives import region_cost
    endpoint=int(clock.event_ends[segment-1])-1
    selected=clock.target_masks[endpoint]&~clock.keep_masks[endpoint]
    if not selected.any():selected=clock.target_masks[endpoint]
    parts=selected.nonzero().flatten().tolist()
    raw=data['s1'][:,column].double()
    valid=torch.isfinite(raw).all(-1)&(data['goal_index'][:,column]==segment-1)
    done=data['done'][:,column].bool()
    valid &= done.long().cumsum(0)-done.long()==0
    valid[1:] &= data['ep_step'][1:,column]>data['ep_step'][:-1,column]
    placeholder=torch.zeros_like(raw);placeholder[:,3]=1
    state=unflatten_state(torch.where(valid[:,None],raw,placeholder))
    root=state[0].detach().clone().requires_grad_()
    state=(root,)+state[1:]
    target=demo['point'][endpoint].double()[None].expand(len(raw),-1,-1)
    normals=torch.as_tensor(normal,dtype=torch.float64)[None].expand_as(target)
    if mode=='euclidean_witness':
        point=surface.contact_points(state,target,parts)
        cost=region_cost(point[:,:,None],target,normals,normal_weight=4.,tangent_weight=1.)[0].squeeze(-1)
    else:
        result=surface.region_loss(state,target,parts,normals=normals,normal_weight=4.,tangent_weight=1.)
        cost=result['loss' if mode=='smooth' else 'hard_loss']
    value=cost[:,parts].mean(-1)
    gradient=torch.autograd.grad(value.sum(),root)[0]
    answer=metrics(value.detach(),gradient.detach(),valid)
    ids=valid.nonzero().flatten()[:8]
    if mode=='smooth' and len(ids):
        fixed=tuple(v[ids].detach() for v in state)
        def objective(position):
            return surface.region_loss((position,)+fixed[1:],target[ids],parts,normals=normals[ids],
                normal_weight=4.,tangent_weight=1.)['loss'][:,parts].mean(-1).sum()
        finite=[]
        for axis in range(3):
            delta=torch.zeros_like(fixed[0]);delta[:,axis]=1e-5
            finite.append((objective(fixed[0]+delta)-objective(fixed[0]-delta))/(2e-5))
        answer['gradient_finite_difference_max_abs_error']=float((torch.stack(finite)-gradient[ids].sum(0)).abs().max())
    return answer,value.detach(),gradient.detach(),valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,nargs='+',required=True)
    p.add_argument('--legacy-model',type=Path,required=True)
    p.add_argument('--samples',type=int,nargs='+',default=[128,512,2048])
    p.add_argument('--temperature',type=float,default=.02)
    p.add_argument('--output',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    from supermimic.contact_control import ContactClock,ContactGeometry
    from supermimic.contact_surface import SurfaceRegionObjective
    demo=torch.load(a.recording,map_location='cpu',weights_only=False)['data']
    clock=ContactClock(demo,hold=1,role_aware=True)
    geometry=ContactGeometry(demo,'cpu').double()
    with redirect_stdout(sys.stderr):model,asset,counts=build_surface_model()
    surfaces={}
    for samples in a.samples:
        surfaces['legacy_'+str(samples)]=SurfaceRegionObjective(a.legacy_model,geometry,
            samples_per_part=samples,temperature=a.temperature,target_normal=[0.,0.,1.]).double()
        with memory_model_fixture(model,asset) as path:
            surfaces['surface_'+str(samples)]=SurfaceRegionObjective(path,geometry,
                samples_per_part=samples,temperature=a.temperature,target_normal=[0.,0.,1.],collision_profile='surface_v2').double()
    first=surfaces['legacy_'+str(a.samples[0])];second=surfaces['surface_'+str(a.samples[0])]
    mesh_difference=[]
    for part in range(6):
        old=getattr(first,'triangles_'+str(part)).reshape(-1,3).unique(dim=0)
        new=getattr(second,'triangles_'+str(part)).reshape(-1,3).unique(dim=0)
        distances=torch.cdist(old,new)
        error=max(float(distances.amin(0).max()),float(distances.amin(1).max()))
        mesh_difference.append(error)
        if error>3e-6:raise ValueError('Part mesh differs beyond helper removal; comparison would be confounded')
    report=dict(schema='somaforge_fixed_collision_profile_ablation_v1',
        scope='same historical states and fixed targets; geometry only, no new contact truth or policy success',
        model_source='canonical surface_v2 USD -> CPU Newton -> actual solver geometry, in RAM',
        robot_asset=asset,colliders=counts,mesh_hausdorff_error_m=mesh_difference,
        settings=dict(normal_weight=4.,tangent_weight=1.,temperature=a.temperature,radius_m=0.),
        catalogs={key:surface.catalog for key,surface in surfaces.items()},trajectories=[])
    for trajectory in a.trajectory:
        saved=torch.load(trajectory,map_location='cpu',weights_only=False)
        data=saved['data'];examples=saved['config']['examples'];cases=[]
        pooled={}
        for column,example in enumerate(examples):
            segment=example['segment'];normal=saved['config']['normals'][column]
            modes={}
            for key,surface in surfaces.items():
                for mode in (['euclidean_witness','smooth'] if key.endswith('_'+str(a.samples[0])) else ['smooth']):
                    name=key+'_'+mode
                    result,value,grad,valid=score(surface,geometry,data,column,segment,clock,demo,normal,mode)
                    modes[name]=result
                    pair=valid[1:]&valid[:-1]
                    v=pooled.setdefault(name,dict(delta=[],gradient=[]))
                    v['delta'].append((value[1:]-value[:-1]).abs()[pair]);v['gradient'].append((grad[1:]-grad[:-1]).norm(dim=-1)[pair])
            cases.append(dict(segment=segment,entry=example['entry'],results=modes))
        summary={}
        for key,fields in pooled.items():
            summary[key]={}
            for field,values in fields.items():
                value=torch.cat(values)
                summary[key][field]=dict(pairs=len(value),mean=float(value.mean()),p95=float(torch.quantile(value,.95)),max=float(value.max()))
        report['trajectories'].append(dict(path=str(trajectory.resolve()),cases=cases,pooled=summary))
    text=json.dumps(report,indent=2,allow_nan=False)
    if a.output:
        with a.output.open('x') as f:f.write(text+'\n')
    print(text)


if __name__=='__main__':main()
