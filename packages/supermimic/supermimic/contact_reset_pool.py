"""Bounded, successor-keyed physical entry states for independent segment training."""
from __future__ import annotations
import torch


def adjacent_successors(starts, stops, motions):
    """Require an exact shared boundary in the same original motion/terrain."""
    entries={(int(m),int(s)):i for i,(m,s) in enumerate(zip(motions,starts))}
    return torch.tensor([entries.get((int(m),int(e)),-1) for m,e in zip(motions,stops)],
                        device=starts.device,dtype=torch.long)


class ResetStatePool:
    def __init__(self, capacity=256, max_age=10, seed=1042):
        self.capacity=capacity;self.max_age=max_age;self.entries={}
        self.generator=torch.Generator(device='cpu').manual_seed(seed)
        self.inserted=0;self.sampled=0

    def add(self, successor, payload, iteration):
        if not len(payload['s']):return
        # A fixed number from each fresh batch, not the last/highest world IDs.
        n=len(payload['s']);indices=torch.randperm(n,generator=self.generator)[:self.capacity].to(payload['s'].device)
        new={k:v[indices].detach().clone() for k,v in payload.items()}
        new['iteration']=torch.full((len(indices),),iteration,device=indices.device,dtype=torch.long)
        previous=self.entries.get(successor)
        if previous is not None:
            live=iteration-previous['iteration']<=self.max_age
            new={k:torch.cat((previous[k][live],v),0)[-self.capacity:] for k,v in new.items()}
        self.entries[successor]=new;self.inserted+=n

    def sample(self, successor, count, iteration):
        item=self.entries.get(successor)
        if item is None:return None
        live=(iteration-item['iteration']<=self.max_age).nonzero().flatten()
        if not len(live):return None
        idx=live[torch.randint(len(live),(count,),generator=self.generator).to(live.device)]
        self.sampled+=count
        return {k:v[idx].clone() for k,v in item.items() if k!='iteration'}

    def counts(self, iteration):
        return {int(k):int((iteration-v['iteration']<=self.max_age).sum()) for k,v in self.entries.items()}
