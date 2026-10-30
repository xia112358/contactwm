"""Continuous stage extraction: preserve actual history but isolate gradients."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import runpy
from pathlib import Path
x=runpy.run_path(str(Path(__file__).with_name('check_contact_pairs.py')))
globals().update({k:v for k,v in x.items() if not k.startswith('__')})
from supermimic.contact_sequence import stage_batches
from supermimic.buffer import ReplayBuffer
# Use actual fixture transitions with a handoff after two frames.
class TimedActor(torch.nn.Module):
    hist=3
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(3))
    def forward(self,h,c):
        index=(h[:,-1,0].detach()*10).round().long().clamp(0,2)
        return self.weight[index,None].expand(-1,29)
actor=TimedActor()
buf=SimpleNamespace(data=d)
plan=dict(length=2,starts=torch.tensor([[0],[2]]),native_ends=torch.tensor([[2],[3]]),ends=torch.tensor([[2],[3]]),enabled=torch.tensor([[False],[True]]))
with patch('supermimic.contact_control.observation',lambda S,*a:S[0][:,:1]),patch('supermimic.contact_dmo.observation',lambda S,*a:S[0][:,:1]):
    batches=list(stage_batches(buf,torch.arange(3),plan,actor,None,None,4096))
    assert len(batches)==1
    batch,cutoffs,history,weight,total=batches[0]
    torch.testing.assert_close(history[0,:,0],torch.tensor([0.,.1,.2]))
    assert batch['ep_step'][0,0]==2 and weight==1 and total==1
    loss,detail=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,batch,completion_mode='accumulated_target',stop_steps=cutoffs,initial_history=history,check=True)
    loss.backward()
    assert actor.weight.grad[:2].eq(0).all() and actor.weight.grad[2]!=0
    assert abs(detail['terminal_contact']-36)<1e-4
print('PASS continuous: true prior-stage history retained, later loss has no earlier-action gradient, local endpoint loss and action replay exact')
# Different starts can run all remaining stages, not only a fixed pair.
from supermimic.contact_task import ContactTask
t=ContactTask.__new__(ContactTask)
t.count=19;t.N=4096;t.continuous_random=True;t.segment_index=None;t.initial_segment=torch.zeros(t.N,dtype=torch.long);t.segment=t.initial_segment.clone()
t.clock=SimpleNamespace(bounds=list(range(20)),event_ends=torch.arange(1,20));t.grace_steps=30
t.contact_tolerances_by_segment=torch.full((19,),.05);t.sequence_live=torch.zeros(t.N,dtype=torch.bool);t.sequence_completed=t.sequence_live.clone();t.sequence_progress=torch.zeros(t.N,dtype=torch.long);t.stage_step=t.sequence_progress.clone()
t.demo={'point':torch.zeros(19,6,3)};t.ids=torch.arange(t.N);t.reader=SimpleNamespace(set_targets=lambda *a:None);t.reset=lambda slots:None
t.reset_sequence();before=t.initial_segment.clone();t.reset_sequence()
assert t.segment.unique().numel()==19 and not torch.equal(before,t.initial_segment)
assert t.sequence_live.all() and t.stage_step.eq(0).all()
print('PASS random resets: all 19 segment starts covered and resampled between attempts')
# Length sorting preserves the weighted objective and parameter gradient.
wide=SimpleNamespace(data={k:v.expand((v.shape[0],3)+v.shape[2:]).clone() for k,v in d.items()})
plan=dict(length=2,starts=torch.tensor([[0]*3,[2]*3]),native_ends=torch.tensor([[2]*3,[3]*3]),ends=torch.tensor([[2,1,2],[3]*3]),enabled=torch.ones(2,3,dtype=torch.bool))
results=[]
with patch('supermimic.contact_control.observation',lambda S,*a:S[0][:,:1]),patch('supermimic.contact_dmo.observation',lambda S,*a:S[0][:,:1]):
    for sort in (False,True):
        actor=TimedActor();total_loss=0.
        for batch,cutoffs,history,weight,total in stage_batches(wide,torch.arange(3),plan,actor,None,None,2,sort_by_duration=sort):
            loss,_=replay_loss(actor,WM(),clock,Geometry(),None,lambda x:x,None,batch,completion_mode='accumulated_target',stop_steps=cutoffs,initial_history=history)
            (loss*weight).backward();total_loss+=float(loss.detach())*weight
        results.append((total_loss,actor.weight.grad.clone()))
assert abs(results[0][0]-results[1][0])<1e-5
torch.testing.assert_close(results[0][1],results[1][1])
print('PASS duration-sorted batches preserve weighted loss and parameter gradients')
