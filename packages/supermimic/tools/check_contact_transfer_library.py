"""Synthetic regression of whole-transfer clocks and matching-terrain routing."""
import sys
from pathlib import Path
import torch

from supermimic.contact_control import ContactClock
from supermimic.contact_library import build_contact_clock,case_world_assignment,dagger_case_worlds,initial_contact_indices


def data(motion):
    n=90
    s=torch.zeros(n,71);s[:,3]=1
    active=torch.zeros(n,6,dtype=torch.bool);active[:,0]=True
    active[25:,1]=True;active[55:,2]=True
    xi=torch.zeros(n,52);xi[:,33]=motion
    return dict(s=s,s1=s.clone(),point=torch.zeros(n,6,3),active=active,known=torch.ones(n,6,dtype=torch.bool),
        load=torch.ones(n,6),slip=torch.zeros(n,6),xi=xi,f=torch.arange(n)+motion*n)


first,second=data(0),data(1)
recording=dict(data={k:torch.cat((first[k],second[k])) for k in first},metadata=dict(recording_spans=[
    dict(begin=0,end=90,motion=0),dict(begin=90,end=180,motion=1)]))
clock=build_contact_clock(recording,hold=1,role_aware=True)
a,b=ContactClock(first,hold=1,role_aware=True),ContactClock(second,hold=1,role_aware=True)
assert clock.bounds==a.bounds+[v+90 for v in b.bounds[1:]]
assert len(clock.event_ends)==len(a.event_ends)+len(b.event_ends)
starts=torch.tensor(clock.bounds[:-1])
initial=initial_contact_indices(clock,starts)
assert initial[starts==90].item()==90
assert torch.equal(initial,torch.maximum(starts-1,(starts//90)*90))
for i,(begin,end) in enumerate(zip(clock.bounds,clock.bounds[1:])):
    assert begin//90==(end-1)//90==int(clock.segment_motion[i])
    expected=a if int(clock.segment_motion[i])==0 else b
    local=clock.segment_local_bounds[i]
    torch.testing.assert_close(clock.target_masks[end-1],expected.target_masks[local[1]-1])
    torch.testing.assert_close(clock.keep_masks[end-1],expected.keep_masks[local[1]-1])
tasks=torch.tensor([0]*8+[1]*8+[2]*3)
ids,segments,background=case_world_assignment(clock,tasks)
assert torch.equal(tasks[ids],clock.segment_motion[segments])
assert set(segments.tolist())==set(range(len(clock.event_ends)))
assert (tasks[background]==2).all()
chosen=ids
cases=torch.arange(len(clock.event_ends))
worlds=dagger_case_worlds(clock,tasks,chosen,cases,repeat=2)
assert len(worlds.unique())==len(worlds)
assert torch.equal(tasks[worlds],clock.segment_motion[cases])
single=build_contact_clock(dict(data=first),hold=1,role_aware=True)
assert single.bounds==a.bounds and not hasattr(single,'segment_motion')
assert initial_contact_indices(single,torch.tensor([0,26])).tolist()==[0,25]
print('PASS complete transfer boundaries, no synthetic motion-boundary event, matching native terrain, full case coverage and single-motion compatibility')
