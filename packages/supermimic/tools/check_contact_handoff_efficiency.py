"""Same-state command handoff benchmark; no physics or optimizer updates."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

pass
import torch
from supermimic.contact_task import ContactTask
from supermimic.contact_native import NativeContacts

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline-source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
namespace=dict(ContactTask.finish_sequence.__globals__)
exec(compile(a.baseline_source.read_text(),str(a.baseline_source),'exec'),namespace)
before=namespace['ContactTask'].finish_sequence

def dense(self,stage_done,success,transition):
    last=self.sequence_last
    final=stage_done&success&(self.segment==last)
    terminal=stage_done&(~success|(self.segment==last))
    advance=stage_done&success&~final
    self.sequence_progress+=success.long();self.sequence_completed|=final;self.sequence_live&=~terminal
    transition['done']=terminal;self.segment+=advance.long()
    self.starts.copy_(torch.where(advance,self._sequence_starts[self.segment],self.starts))
    self.ends.copy_(torch.where(advance,self.clock.event_ends[self.segment],self.ends))
    self.budgets.copy_(torch.where(advance,self.fixed_budgets[self.segment],self.budgets))
    self.contact_tolerance.copy_(torch.where(advance[:,None],self.contact_tolerances_by_segment[self.segment,None],self.contact_tolerance))
    self.stage_step.masked_fill_(advance,0);self.valid_contacts.masked_fill_(advance[:,None],False)
    self.recontact.masked_fill_(advance[:,None],False);self.onset_point.masked_fill_(advance[:,None,None],0)
    self.onset_face.masked_fill_(advance[:,None],-1)
    values=self.demo['point'][self.ends-1]+self.env.scene.env_origins[self.ids,None]
    self.reader.targets[self.ids]=torch.where(advance[:,None,None],values,self.reader.targets[self.ids])

def fixture():
    t=ContactTask.__new__(ContactTask);t.count=19;t.N=4057;t.ids=torch.arange(t.N,device=a.device)
    bounds=[0,82,110,132,202,223,261,305,349,392,441,478,513,541,588,617,643,669,690,799]
    t.clock=SimpleNamespace(bounds=bounds,event_ends=torch.tensor(bounds[1:],device=a.device))
    t._sequence_starts=torch.tensor(bounds[:-1],device=a.device)
    t.segment=torch.arange(t.N,device=a.device)%19;t.sequence_last=torch.full_like(t.segment,18)
    t.starts=t._sequence_starts[t.segment];t.ends=t.clock.event_ends[t.segment]
    t.fixed_budgets=torch.tensor([95,31,22,74,23,19,44,21,42,42,41,38,27,49,30,28,26,23,17],device=a.device)
    t.budgets=t.fixed_budgets[t.segment];t.grace_steps=30
    t.contact_tolerances_by_segment=torch.full((19,),.05,device=a.device)
    t.contact_tolerance=t.contact_tolerances_by_segment[t.segment,None]
    t.sequence_live=torch.ones(t.N,device=a.device,dtype=torch.bool);t.sequence_completed=torch.zeros_like(t.sequence_live)
    t.sequence_progress=torch.zeros_like(t.segment);t.stage_step=torch.ones_like(t.segment)*20
    t.valid_contacts=torch.ones(t.N,6,device=a.device,dtype=torch.bool);t.recontact=t.valid_contacts.clone()
    t.onset_point=torch.ones(t.N,6,3,device=a.device);t.onset_face=torch.ones(t.N,6,device=a.device,dtype=torch.long)
    t.previous_contact=t.valid_contacts.clone();t.history=torch.randn(t.N,5,232,device=a.device)
    t.env=SimpleNamespace(N=4096,device=a.device,scene=SimpleNamespace(env_origins=torch.randn(4096,3,device=a.device)))
    t.demo={'point':torch.randn(799,6,3,device=a.device)}
    t.reader=NativeContacts.__new__(NativeContacts);t.reader.direct=True;t.reader.env=t.env;t.reader.targets=None
    t.reader.set_targets(t.demo['point'][t.ends-1],t.ids)
    return t

torch.manual_seed(47);t=fixture()
saved={k:v.clone() for k,v in vars(t).items() if isinstance(v,torch.Tensor)}
targets=t.reader.targets.clone()
def restore():
    for k,v in saved.items():getattr(t,k).copy_(v)
    t.reader.targets.copy_(targets)
def sync():
    if a.device.startswith('cuda'):torch.cuda.synchronize()

reports=[]
for fraction in (0.,.01,.1,.5):
    done=torch.arange(t.N,device=a.device)<int(fraction*t.N)
    success=done&(torch.arange(t.N,device=a.device)%7!=0)
    ended=done.nonzero().flatten()
    restore();reference={};before(t,done,success,reference)
    expected={k:v.clone() for k,v in vars(t).items() if isinstance(v,torch.Tensor)}
    expected_target=t.reader.targets.clone()
    samples={k:[] for k in ('before','partial_targets','reused_ended','dense_candidate')}
    variants=[('before',before),('partial_targets',ContactTask.finish_sequence),
        ('reused_ended',lambda task,done,success,out:ContactTask.finish_sequence(task,done,success,out,ended=ended)),
        ('dense_candidate',dense)]
    for repeat in range(24):
        for label,fn in (variants if repeat%2 else list(reversed(variants))):
            restore();out={};sync();start=time.perf_counter();fn(t,done,success,out);sync()
            samples[label].append(1000*(time.perf_counter()-start))
            for k,v in expected.items():assert torch.equal(getattr(t,k),v),(fraction,label,k)
            assert torch.equal(t.reader.targets,expected_target)
            assert torch.equal(out['done'],reference['done'])
    reports.append(dict(fraction_done=fraction,equivalent=True,
        timings={k:dict(median_ms=statistics.median(v[2:]),mean_ms=statistics.mean(v[2:])) for k,v in samples.items()}))
report=dict(scope='Same-state synthetic handoff benchmark; no simulator or training',
            worlds=t.N,device=a.device,reports=reports,targets_and_all_task_state_exact=True)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
