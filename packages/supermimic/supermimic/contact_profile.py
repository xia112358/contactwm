"""Synchronized, nested wall-clock attribution for the native collector."""
import functools
import json
import time
from pathlib import Path
import torch


class CollectionProfiler:
    def __init__(self):
        self.stack=[];self.samples={}

    def wrap(self,obj,attr,label):
        original=getattr(obj,attr)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            torch.cuda.synchronize();entry=[time.perf_counter(),0.];self.stack.append(entry)
            try:return original(*args,**kwargs)
            finally:
                torch.cuda.synchronize();elapsed=time.perf_counter()-entry[0];self.stack.pop()
                self.samples.setdefault(label,[]).append((elapsed,max(0.,elapsed-entry[1])))
                if self.stack:self.stack[-1][1]+=elapsed
        setattr(obj,attr,measured)

    def write(self,path):
        report={k:dict(calls=len(v),inclusive_seconds=sum(x[0] for x in v),exclusive_seconds=sum(x[1] for x in v),mean_ms=1000*sum(x[0] for x in v)/len(v),first_call_ms=1000*v[0][0],mean_after_first_ms=1000*sum(x[0] for x in v[1:])/max(1,len(v)-1)) for k,v in self.samples.items()}
        Path(path).write_text(json.dumps(dict(method='CUDA synchronize at nested timing boundaries; inclusive timings overlap, exclusive timings do not; first call may include capture/initialization',timings=report),indent=2))
        return report
