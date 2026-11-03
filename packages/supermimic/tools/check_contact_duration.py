"""Regression checks for dynamic duration selection without a simulator."""
import sys
from pathlib import Path
import torch
from supermimic.contact_duration import select_duration_batch


def batch(frames, success, segments):
    frames = torch.tensor(frames)
    done = torch.arange(int(frames.max()))[:, None] == frames[None] - 1
    return dict(done=done, term=done & ~torch.tensor(success)[None],
                goal_index=torch.tensor(segments)[None].expand_as(done))


d = batch([10,10,11,20,10,20,20,21,30],
          [True,True,True,True,False,True,True,True,False], [0]*5+[1]*4)
before = {k:v.clone() for k,v in d.items()}
ids, stops, info = select_duration_batch(d, [10,20], 2, 3)
assert ids.tolist() == list(range(9))
assert stops.tolist() == [10,10,11,12,10,20,20,21,22]
assert info["truncated"] == 2
assert info['segments'][0]['retained_failures'] == 1
assert [s['center'] for s in info['segments']] == [10,20]
assert all(torch.equal(d[k],v) for k,v in before.items())
# A new batch selects its own mode, with no persistent center.
_, _, info = select_duration_batch(batch([30,30,31], [True]*3, [0]*3), [10], 2, 3)
assert info['segments'][0]['center'] == 30
# No reliable successful mode: preserve failures rather than starve the task.
ids, stops, info = select_duration_batch(batch([10,20], [False,False], [0,0]), [10], 2, 3)
assert ids.tolist() == [0,1] and info['segments'][0]['fallback']
# Distinct equal peaks must select a peak, never their midpoint.
_, _, info = select_duration_batch(batch([10,10,30,30], [True]*4, [0]*4), [30], 2, 3)
assert info['segments'][0]['center'] == 30 and info['segments'][0]['fallback']
assert info['retained'] == 2


ids, stops, info = select_duration_batch(batch([40,40], [False,False], [0,0]), [10], 2, 3)
assert ids.tolist()==[0,1] and stops.tolist()==[12,12]
assert info["truncated"]==2

# Too-short attempts are excluded; native records remain unchanged.
ids, stops, info=select_duration_batch(batch([5,10,20],[False,False,False],[0,0,0]),[10],2,3)
assert ids.tolist()==[1,2] and stops.tolist()==[5,10,12]
assert info['discarded']==1 and info['truncated']==1
print('PASS duration filtering: independent segments, dynamic centers, failure retention, fallback, multimodal tie, immutable endpoints')

# Exercise actual replay_loss: a long rollout's earlier goal hit must not
# replace its cutoff-frame contact loss; states after cutoff cannot contribute.
from types import SimpleNamespace
from unittest.mock import patch
from supermimic.contact_dmo import replay_loss
class Actor(torch.nn.Module):
    hist=1
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(3));self.calls=0
    def forward(self,h,c):
        a=self.weight[self.calls].expand(len(h),29);self.calls+=1;return a
class WM(torch.nn.Module):
    def step(self,S,a,xi):return (S[0]+a[:,:3],)+S[1:]
    def next_context(self,S,a,xi,xin):return xi
class Geometry:
    def poses(self,S):return S[0][:,None].expand(-1,6,-1),S[1][:,None].expand(-1,6,-1,-1)
    def __call__(self,S,frame):return self.poses(S)[0]
s1=torch.zeros(3,1,71);s1[:,:,2]=1;s1[:,:,3]=1;s1[:,0,0]=torch.tensor([.1,.2,.3])
s0=s1.clone();s0[0,0,0]=0;s0[1:]=s1[:-1]
new=torch.tensor([[True,False,False,False,False,False]])
ref=s0[:1,0].clone();target=torch.zeros(1,6,3);target[:,:,2]=1
clock=SimpleNamespace(bounds=[0,1],event_ends=torch.tensor([1]),target_masks=new,keep_masks=torch.zeros_like(new),data={'point':target,'s1':ref},condition=lambda S,*a,**kw:torch.zeros(len(S[0]),1))
clock.data['s']=ref
d=dict(s=s0,s1=s1,xi=torch.zeros(3,1,1),next_xi=torch.zeros(3,1,1),a=torch.zeros(3,1,29),a_prev=torch.zeros(3,1,29),ep_step=torch.arange(3)[:,None],goal_index=torch.zeros(3,1,dtype=torch.long),done=torch.tensor([[False],[False],[True]]),contact_goal=torch.zeros(3,1,6,dtype=torch.bool),active=torch.ones(3,1,6,dtype=torch.bool),known=torch.ones(3,1,6,dtype=torch.bool))
d['contact_goal'][0,0,0]=True
d['point']=s1[:,:,:3][:,:,None,:].expand(-1,-1,6,-1).clone();d['point'][0,0,0,0]=.01
with patch('supermimic.contact_dmo.observation',lambda S,*a:S[0][:,:1]):
    actor=Actor();loss,details=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,d,completion_mode='accumulated_target',stop_steps=torch.tensor([2]))
    assert abs(details['terminal_contact']-16.)<1e-4 and details['truncated_trajectories']==1
    assert actor.calls==2
    loss.backward();assert actor.weight.grad[:2].abs().min()>0 and actor.weight.grad[2]==0
    actor=Actor();_,details=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,d,completion_mode='accumulated_target')
    assert abs(details['terminal_contact']-.04)<1e-5 and details['truncated_trajectories']==0
print('PASS actual replay_loss: cutoff contact/root, earlier goal hit ignored for truncated attempts, early gradient preserved, no post-cutoff gradient, ordinary behavior preserved')

from supermimic.contact_duration import DurationBandHistory
history=DurationBandHistory([84],bootstrap_batches=2)
reliable=batch([111]*8,[True]*8,[0]*8)
unreliable=batch([112]*8,[False]*8,[0]*8)
history.select(reliable)
_,stops,r=history.select(unreliable)
assert stops.tolist()==[86]*8 and r['segments'][0]['source']=='initial_prior'
# Once bootstrapped, failure cannot reset a learned 111-frame center to 84.
_,stops,r=history.select(unreliable)
assert stops.tolist()==[112]*8 and r['segments'][0]['center']==111
assert r['segments'][0]['source']=='last_reliable' and r['retained']==8
restored=DurationBandHistory.from_state_dict(history.state_dict())
_,stops,r=restored.select(unreliable)
assert r['segments'][0]['center']==111 and restored.batches==4
restored.select(batch([100]*8,[True]*8,[0]*8))
_,stops,r=restored.select(unreliable)
assert stops.tolist()==[102]*8 and r['segments'][0]['center']==100
empty=DurationBandHistory([84],bootstrap_batches=0)
_,stops,r=empty.select(unreliable)
assert r['segments'][0]['source']=='initial_no_reliable_history'
print('PASS rolling prior: bootstrap, last reliable fallback, checkpoint restore, reliable updates, no-history fallback')
