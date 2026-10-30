"""Check command handoff, failure isolation, and preservation of live history."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys
from pathlib import Path
from types import SimpleNamespace
pass
import torch
from supermimic.contact_task import ContactTask

t=ContactTask.__new__(ContactTask)
t.count=2;t.segment=torch.tensor([0,0,1]);t.sequence_live=torch.ones(3,dtype=torch.bool)
t.sequence_completed=torch.zeros(3,dtype=torch.bool);t.sequence_progress=torch.tensor([0,0,1])
t.clock=SimpleNamespace(bounds=[0,2,5],event_ends=torch.tensor([2,5]))
t.starts=torch.tensor([0,0,2]);t.ends=torch.tensor([2,2,5]);t.grace_steps=10
t.budgets=torch.tensor([12,12,13]);t.contact_tolerances_by_segment=torch.tensor([.05,.08])
t.contact_tolerance=torch.tensor([[.05],[.05],[.08]]);t.stage_step=torch.tensor([2,12,3])
t.valid_contacts=torch.ones(3,6,dtype=torch.bool);t.recontact=t.valid_contacts.clone()
t.onset_point=torch.ones(3,6,3);t.onset_face=torch.zeros(3,6,dtype=torch.long)
t.previous_contact=torch.ones(3,6,dtype=torch.bool)
t.history=torch.randn(3,5,232);history=t.history.clone()
t.env=SimpleNamespace(a_prev=torch.randn(3,29),xi=torch.randn(3,53),ep_step=torch.tensor([2,12,8]))
physical={k:v.clone() for k,v in vars(t.env).items()}
t.demo={'point':torch.randn(5,6,3)};t.ids=torch.arange(3)
calls=[];t.reader=SimpleNamespace(update_targets=lambda points,ids:calls.append((points.clone(),ids.clone())))
transition={};t.finish_sequence(torch.ones(3,dtype=torch.bool),torch.tensor([True,False,True]),transition)
assert t.segment.tolist()==[1,0,1]
assert t.sequence_progress.tolist()==[1,0,2]
assert t.sequence_live.tolist()==[True,False,False]
assert t.sequence_completed.tolist()==[False,False,True]
assert transition['done'].tolist()==[False,True,True]
assert t.stage_step.tolist()==[0,12,3] and t.budgets.tolist()==[13,12,13]
assert not t.valid_contacts[0].any() and t.previous_contact[0].all()
assert torch.equal(t.history,history)
assert all(torch.equal(v,getattr(t.env,k)) for k,v in physical.items())
assert len(calls)==1 and torch.equal(calls[0][0][0],t.demo['point'][4])
assert calls[0][1].tolist()==[0], 'Only the handed-off world changes its target'
assert abs(float(t.contact_tolerance[0])-.08)<1e-7
print('PASS: success advances only the command; physics, history and previous action preserved; local deadline resets; failures and final successes stop independently')
