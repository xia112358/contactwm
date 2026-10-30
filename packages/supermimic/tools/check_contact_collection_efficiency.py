"""Frozen native collector check; compare witness reads on the same solver solve.

Pass the normal contact_train arguments with both --freeze flags. This harness
exits after --efficiency_steps (default 32), before any training or checkpoint.
"""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import inspect
import json
from pathlib import Path
import runpy
import statistics
import sys
import textwrap
import time

pass
import torch
from supermimic.contact_native import NativeContacts
from supermimic.contact_task import ContactTask
from supermimic.contact_control import assess_region_arrival,goal_contact_mask,new_part_contact
from supermimic.mdp import unflatten_state
from supermimic.rotations import rot_angle_sq

assert '--freeze_actor' in sys.argv and '--freeze_wm' in sys.argv
steps=32
if '--efficiency_steps' in sys.argv:
    i=sys.argv.index('--efficiency_steps');steps=int(sys.argv[i+1]);del sys.argv[i:i+2]
output=Path(sys.argv[sys.argv.index('--output')+1])
original=NativeContacts._read_direct
source=textwrap.dedent(inspect.getsource(original))
new='points=torch.cat((point,point.new_zeros((1,3))),0)[winner]'
old='points=point.new_zeros((total,3))\n    points[present]=point[winner[present]]'
assert new in source
source=source.replace(new,old)
source=source.replace('positions[:,self.part_body_ids],rotations[:,self.part_body_ids]',
                      'positions[:,self.part_body_ids].clone(),rotations[:,self.part_body_ids].clone()')
scope=dict(original.__globals__);exec(source,scope);reference=scope['_read_direct']
timings={'before':[],'after':[]};frames=[];reader_checks=0;force_delta=[];state_checks=0

def checked_read(self):
    global reader_checks
    if len(frames)%4:
        return original(self)
    results={}
    # Alternate order to avoid consistently charging one side for cache warming.
    order=[('before',reference),('after',original)]
    if reader_checks%2:order.reverse()
    for label,fn in order:
        torch.cuda.synchronize();t=time.perf_counter();results[label]=fn(self)
        torch.cuda.synchronize();timings[label].append(time.perf_counter()-t)
    for key in results['before']:
        if key=='paired_force_w':
            # Unchanged CUDA index_add uses atomic summation; separate identical
            # reads can differ at roundoff scale even before this optimization.
            force_delta.append(float((results['after'][key]-results['before'][key]).abs().max()))
            torch.testing.assert_close(results['after'][key],results['before'][key],rtol=2e-5,atol=2e-4,msg=key)
        else:
            torch.testing.assert_close(results['after'][key],results['before'][key],rtol=0,atol=0,equal_nan=True,msg=key)
    reader_checks+=1
    # Retained evidence/poses refer to this exact solve whichever method ran last.
    return results['after']

NativeContacts._read_direct=checked_read
step=ContactTask.step

def measured(self):
    global state_checks
    assert not self.sequential,'This check uses independent episodes'
    if not hasattr(self,'_efficiency_checked_env'):
        ordinary_step=self.env.step
        def checked_env_step(action,*,precomputed_state=None):
            global state_checks
            if len(frames)%4==0 and precomputed_state is not None:
                saved_state=tuple(x.clone() for x in precomputed_state)
                saved_xi=self.env.xi.clone();fresh=self.env.state()
                for a,b in zip(saved_state,fresh):assert torch.equal(a,b)
                assert torch.equal(saved_xi,self.env.xi)
                out=ordinary_step(action,precomputed_state=precomputed_state)
                for a,b in zip(saved_state,precomputed_state):assert torch.equal(a,b),'Precomputed state aliases live simulator storage'
                state_checks+=1
                return out
            return ordinary_step(action,precomputed_state=precomputed_state)
        self.env.step=checked_env_step;self._efficiency_checked_env=True
    saved={key:getattr(self,key).clone() for key in ('previous_contact','recontact','valid_contacts','onset_point','onset_face','ends','budgets')}
    torch.cuda.synchronize();started=time.perf_counter()
    result=step(self);torch.cuda.synchronize();frames.append(time.perf_counter()-started)
    # Original completion formula, evaluated on the very same native transition.
    # Includes short segments that reset during the 32-frame probe.
    target=self.clock.data['point'][saved['ends']-1]
    keep=self.clock.keep_masks[saved['ends']-1]
    new=self.clock.target_masks[saved['ends']-1]&~keep
    support=None if self.reader.direct else self.clock.support_masks[saved['ends']-1]
    arrived,kept=assess_region_arrival(result,target,self.target_faces,new,keep,support,tolerance=self.contact_tolerance)
    state=unflatten_state(result['s1']);reference_state=unflatten_state(self.clock.data['s1'][saved['ends']-1])
    root=(state[0]-reference_state[0]).norm(dim=-1);angle=rot_angle_sq(state[1],reference_state[1]).sqrt()
    failed=(~torch.isfinite(result['s1']).all(-1))|(state[0][:,2]<.15)
    observed=result['known']&result['active']
    first=observed&~saved['previous_contact']&~saved['recontact']&new
    onset_point=torch.where(first[...,None],result['point'],saved['onset_point'])
    onset_face=torch.where(first,result['surface_local'],saved['onset_face'])
    contact_end,recontact=new_part_contact(observed,saved['previous_contact'],saved['recontact'],new)
    valid=recontact&(torch.ones_like(observed) if self.target_faces is None else onset_face==self.target_faces)&((onset_point-target).norm(dim=-1)<=self.contact_tolerance)
    arrived=torch.where(new.any(-1),((~new)|valid).all(-1),arrived)
    completed=arrived&(root<self.root_tolerance)&(angle<.4)&torch.where(new.any(-1),contact_end,kept)&~failed
    valid_first=torch.zeros_like(new)
    if self.completion_mode=='accumulated_target':
        valid=observed if self.contact_only else goal_contact_mask(result,target,self.target_faces,self.contact_tolerance)
        valid_first=valid&new&~saved['valid_contacts']
        accumulated=saved['valid_contacts']|(valid&new)
        arrived=((~new)|accumulated).all(-1)
        completed=arrived&(root<self.root_tolerance)&(angle<.4)&torch.where(new.any(-1),True,kept)&~failed
        if self.contact_only:completed=arrived&torch.where(new.any(-1),True,kept)&~failed
        contact_end=torch.zeros_like(completed)
    expired=result['ep_step']+1>=saved['budgets']
    expected=dict(contact_first=first,contact_goal=valid_first,
        done=contact_end|completed|failed|expired,
        term=failed|(expired&~completed)|(contact_end&~completed))
    for key,value in expected.items():
        assert torch.equal(result[key],value),(len(frames),key)
    if len(frames)==steps:
        # Ordinary frames invoke exactly one normal reader; comparison frames
        # intentionally perform two reads and tensor checks and are excluded.
        ordinary=[value for i,value in enumerate(frames) if i%4]
        report=dict(scope='Frozen native collection; no optimizer, no checkpoint write',
            worlds=self.env.N,frames=steps,matched_solver_read_checks=reader_checks,
            native_decisions_matched=steps*self.N,
            precomputed_state_and_sensor_context_checks=state_checks,
            nearest_witness_and_contact_masks_bitwise_equal=True,
            unchanged_atomic_force_max_abs_delta=max(force_delta),
            actor_updates=0,wm_updates=0,
            ordinary_frame_mean_ms=1000*statistics.mean(ordinary),
            ordinary_frame_median_ms=1000*statistics.median(ordinary),
            native_reader={k:dict(median_ms=1000*statistics.median(v[1:]),
                mean_ms=1000*statistics.mean(v[1:]),samples_ms=[1000*x for x in v]) for k,v in timings.items()})
        (output/'collection-efficiency.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'status.json').write_text(json.dumps(dict(phase='complete',**report),indent=2)+'\n')
        print(json.dumps(report),flush=True)
        raise SystemExit(0)
    return result

ContactTask.step=measured
runpy.run_module('supermimic.contact_train',run_name='__main__')
