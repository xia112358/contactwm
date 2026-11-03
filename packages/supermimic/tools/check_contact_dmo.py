"""Check exact simulator forward values and chained surrogate derivatives."""
from pathlib import Path
import sys
import torch
from supermimic.contact_dmo import swap

x=torch.tensor([2.],requires_grad=True)
a=torch.tensor([3.],requires_grad=True)
actual1=torch.tensor([10.],requires_grad=True)
y=swap(x.square()+3*a,actual1)
assert torch.equal(y,actual1)
z=swap(y.square()+a,torch.tensor([7.]))
assert z.item()==7
z.square().sum().backward()
# 2*z * d(y^2)/dy at real y=10 * d(x^2)/dx at x=2.
assert x.grad.item()==14*20*4
assert a.grad.item()==14*(20*3+1)
assert actual1.grad is None
print('PASS exact real forward; Jacobians evaluated at real states; early-action chain; no simulator derivative')

from types import SimpleNamespace
from supermimic.contact_dmo import native_attempt_metrics
s1=torch.zeros(3,2,71);s1[...,2]=1;s1[...,3]=1
mask=torch.tensor([[True,False,False,False,False,False]])
clock=SimpleNamespace(event_ends=torch.tensor([1]),target_masks=mask,keep_masks=torch.zeros_like(mask),data={'point':torch.zeros(1,6,3),'s1':s1[0,:1]})
active=torch.zeros(3,2,6,dtype=torch.bool);active[:2,0,0]=True;active[2,:,0]=True
point=torch.zeros(3,2,6,3);point[0,0,0,0]=.1;point[1,0,0,0]=.06
batch=dict(s1=s1,done=torch.tensor([[False,False],[True,True],[True,True]]),term=torch.tensor([[False,False],[False,True],[False,False]]),goal_index=torch.zeros(3,2,dtype=torch.long),active=active,known=torch.ones_like(active),surface_local=torch.zeros(3,2,6,dtype=torch.long),point=point)
m=native_attempt_metrics(batch,clock,torch.zeros(1,6,dtype=torch.long))
assert m['attempts']==2 and m['success_rate']==.5 and m['target_face_contact_rate']==.5
assert m['contact_error_samples']==1 and abs(m['best_contact_error_cm']-6)<1e-5
print('PASS contact error excludes missing observations and post-reset frames')

# Direct contact mode does not require surface IDs, but still excludes absent contacts.
batch['surface_local'].fill_(-1)
m=native_attempt_metrics(batch,clock,None)
assert m['face_matching'] is False and m['contact_error_samples']==1
assert abs(m['best_contact_error_cm']-6)<1e-5
from supermimic.contact_control import assess_region_arrival
evidence={k:batch[k][1,0].clone() for k in ('active','known','point','surface_local')}
evidence['point'][0,0]=.04
assert assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]))[0]
evidence['point'][0,0]=.06
assert not assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]))[0]
evidence['point'][0,0]=0;evidence['active'][0]=False
assert not assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]))[0]
print('PASS direct contact needs actual observation within 5 cm, without face matching')

# A relaxed landing radius changes completion only and remains per environment.
from supermimic.contact_control import goal_contact_mask,load_contact_tolerances
evidence['active'][0]=True;evidence['point'][0,0]=.07
assert not assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]))[0]
assert assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]),tolerance=.08)[0]
evidence['point'][0,0]=.081
assert not assess_region_arrival(evidence,clock.data['point'][0],None,mask[0],torch.zeros_like(mask[0]),tolerance=.08)[0]
evidence['point'][0,0]=.07
pair={k:torch.stack([v,v]) for k,v in evidence.items()}
valid=goal_contact_mask(pair,clock.data['point'][0],tolerance=torch.tensor([[.08],[.05]]))
assert valid[:,0].tolist()==[True,False]
pair['active'][:,0]=False
assert not goal_contact_mask(pair,clock.data['point'][0],tolerance=.08).any()

# Two landing contacts can be achieved at different frames, but both need evidence.
sequence=dict(point=torch.zeros(2,6,3),active=torch.zeros(2,6,dtype=torch.bool),known=torch.ones(2,6,dtype=torch.bool))
sequence['point'][:,:2,0]=.07
sequence['active'][0,0]=True;sequence['active'][1,1]=True
valid=goal_contact_mask(sequence,torch.zeros(6,3),tolerance=.08)
assert not valid[:,:2].all(-1).any()
assert valid.long().cummax(0).values[-1,:2].all()

# The source profile must not loosen another recording or another contact type.
import json
profile=Path(__file__).resolve().parents[1]/'experiments/contact_completion.json'
rule=json.loads(profile.read_text())['overrides'][0]
target_masks=torch.zeros(799,6,dtype=torch.bool);target_masks[512,:2]=True
scoped_clock=SimpleNamespace(bounds=[0,478,513,799],event_ends=torch.tensor([478,513,799]),target_masks=target_masks,keep_masks=torch.zeros_like(target_masks),data={'s':torch.zeros(799,71)})
metadata={k:rule[k] for k in ('task_manifest_sha256','motion')}
assert load_contact_tolerances(scoped_clock,metadata,profile)==[.05,.08,.05]
assert load_contact_tolerances(scoped_clock,dict(metadata,motion=1),profile)==[.05,.05,.05]
target_masks[512,1]=False
assert load_contact_tolerances(scoped_clock,metadata,profile)==[.05,.05,.05]
print('PASS scoped 8 cm landing criterion; other tasks remain 5 cm; actual contacts required; asynchronous contacts preserved')
