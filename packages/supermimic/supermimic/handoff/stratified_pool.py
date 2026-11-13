import torch
class StratifiedPool:
 def __init__(self,capacity=256,max_age=None,seed=1042):
  self.capacity=capacity;self.entries={};self.generator=torch.Generator().manual_seed(seed)
 def add(self,successor,payload,iteration):
  keys=torch.stack((payload['origin'],payload['depth']),1)
  for pair in keys.unique(dim=0).tolist():
   origin,depth=pair;assert successor-origin==depth and depth>0
   ids=(keys==keys.new_tensor(pair)).all(1).nonzero().flatten();ids=ids[torch.randperm(len(ids),generator=self.generator).to(ids.device)[:self.capacity]]
   key=(int(successor),origin,depth);new={k:v[ids].detach().clone() for k,v in payload.items()};previous=self.entries.get(key)
   if previous is not None:
    merged={k:torch.cat((previous[k],v),0) for k,v in new.items()}
    keep=torch.randperm(len(merged['s']),generator=self.generator)[:self.capacity].to(merged['s'].device)
    new={k:v[keep] for k,v in merged.items()}
   self.entries[key]=new
 def sample(self,successor,count,iteration):
  keys=sorted(k for k in self.entries if k[0]==successor)
  if not keys:return None
  sizes=torch.tensor([len(self.entries[k]['s']) for k in keys])
  allocation=capacity_balanced_counts(sizes,count,self.generator)
  chunks=[]
  for key,n in zip(keys,allocation.tolist()):
   item=self.entries[key];size=len(item['s'])
   # Exhaust each random permutation before any stored entry is reused.
   cycles,tail=divmod(n,size)
   draws=[torch.randperm(size,generator=self.generator) for _ in range(cycles)]
   draws.append(torch.randperm(size,generator=self.generator)[:tail])
   idx=torch.cat(draws).to(item['s'].device)
   chunks.append({k:v[idx].clone() for k,v in item.items()})
  result={k:torch.cat([x[k] for x in chunks]) for k in chunks[0]};shuffle=torch.randperm(count,generator=self.generator).to(result['s'].device)
  return {k:v[shuffle] for k,v in result.items()}
 def counts(self,iteration):
  result={}
  for (dest,origin,depth),v in self.entries.items():result[dest]=result.get(dest,0)+len(v['s'])
  return result
 def composition(self):
  return [dict(case=d+82,origin=o+82,depth=h,count=len(v['s'])) for (d,o,h),v in sorted(self.entries.items())]
 def state_dict(self):return dict(entries=self.entries,generator=self.generator.get_state(),capacity=self.capacity)
 def load_state_dict(self,state):
  self.entries=state['entries'];self.capacity=state['capacity'];self.generator.set_state(state['generator'].cpu())


def capacity_balanced_counts(sizes,count,generator):
 """Max-min source balance capped by available entries, without a maturity knob.

 When the whole pool is exhausted, every stored entry gets one more reuse before
 any gets another. The remainder is balanced across sources up to their capacity.
 """
 total=int(sizes.sum());cycles,remainder=divmod(count,total)
 order=torch.randperm(len(sizes),generator=generator)
 levels=torch.arange(int(sizes.max()))[:,None]
 tickets=order.expand(len(levels),-1)[levels<sizes[order][None,:]]
 return cycles*sizes+torch.bincount(tickets[:remainder],minlength=len(sizes))
