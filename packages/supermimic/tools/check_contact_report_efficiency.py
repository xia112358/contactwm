"""Compare packed reporting with the former scalar readbacks, without simulation."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import inspect
import json
from pathlib import Path
import statistics
import sys
import textwrap
import time
from types import SimpleNamespace

pass
import torch
from supermimic.contact_task import ContactTask

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path)
a=p.parse_args();torch.set_num_threads(4)
source=textwrap.dedent(inspect.getsource(ContactTask.report))
begin=source.index('    # Reporting runs repeatedly')
end=source.index('    total=sum(n);wins=sum(s)',begin)
old='''    n=self.attempts.cpu().tolist();s=self.successes.cpu().tolist()
    root=self.root_sum.cpu().tolist();point=self.point_sum.cpu().tolist()
'''
source=source[:begin]+old+source[end:]
source=source.replace('current_root_error_m=values[-2],current_contact_error_m=values[-1]',
    'current_root_error_m=float(self.current_root),current_contact_error_m=float(self.current_contact)')
source=source.replace('contact_tolerance_m=tolerances[i]','contact_tolerance_m=float(self.contact_tolerances_by_segment[i])')
source=source.replace('terminal_steps=int(terminals[i])','terminal_steps=int(self.last_terminal_steps[i])')
scope=dict(ContactTask.report.__globals__);exec(source,scope);old_report=scope['report']
t=ContactTask.__new__(ContactTask);t.count=19;t.N=4057;t.env=SimpleNamespace(N=4096)
t.sequential=False;t.segment_index=None;t.grace_steps=10
t._report_ranges=[(20*i,20*(i+1)) for i in range(t.count)]
t.attempts=torch.arange(1,20,device=a.device,dtype=torch.float32)*13
t.successes=(t.attempts*.6).floor();t.root_sum=t.attempts*.13;t.point_sum=t.attempts*.03
t.contact_tolerances_by_segment=torch.full((t.count,),.05,device=a.device)
t.last_terminal_steps=torch.arange(19,38,device=a.device)
t.current_root=torch.tensor(.21,device=a.device);t.current_contact=torch.tensor(.04,device=a.device)
timings={'before':[],'after':[]}
def sync():
    if a.device.startswith('cuda'):torch.cuda.synchronize()
reference=old_report(t,3,40)
for _ in range(4):assert ContactTask.report(t,3,40)==reference
for i in range(60):
    order=[('before',old_report),('after',ContactTask.report)]
    if i%2:order.reverse()
    for label,fn in order:
        sync();start=time.perf_counter();result=fn(t,3,40);sync()
        timings[label].append(1000*(time.perf_counter()-start));assert result==reference
report=dict(scope='Report-only benchmark; no simulator, optimizer or checkpoints',
    device=a.device,segments=t.count,all_fields_exact=True,
    timings={k:dict(median_ms=statistics.median(v),mean_ms=statistics.mean(v)) for k,v in timings.items()})
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
