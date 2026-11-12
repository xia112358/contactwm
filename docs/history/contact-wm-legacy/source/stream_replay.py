"""Per-environment ring replay; windows cannot cross resets or overwritten rows."""
import torch

class Replay:
    def __init__(self,capacity,num_envs,device):
        self.capacity=capacity;self.num_envs=num_envs;self.device=device
        self.data={};self.steps=0
    def add(self,row):
        if 'age' in row:
            # The first post-reset pose has no corresponding previous native solve.
            row={**row,'valid':row['valid']&(row['age']>0)}
        if not self.data:
            self.data={k:torch.empty((self.capacity,*v.shape),dtype=v.dtype,device=self.device) for k,v in row.items()}
        for k,v in row.items():self.data[k][self.steps%self.capacity].copy_(v.detach())
        self.steps+=1
    def windows(self,batch,length,event_fraction=0.,motion_replay=None):
        if not 0<=event_fraction<=1:raise ValueError('Invalid event sampling fraction')
        low=max(0,self.steps-self.capacity);high=self.steps-length+1
        if high<=low:raise ValueError('Replay too short')
        selected=[];envs=[];remaining=batch
        pool=None
        if motion_replay is not None:
            pool=(self.data['motion_replay'][(self.steps-1)%self.capacity]==motion_replay).nonzero().flatten()
            if not len(pool):raise ValueError('No environments in requested WM controller group')
        for _ in range(100):
            starts=torch.randint(low,high,(remaining*2,),device=self.device)
            ids=torch.randint(self.num_envs if pool is None else len(pool),(remaining*2,),device=self.device)
            if pool is not None:ids=pool[ids]
            times=(starts[None]+torch.arange(length,device=self.device)[:,None])%self.capacity
            valid=self.data['valid'][times,ids].all(0)&self.data['previous_known'][times,ids].all((0,2))
            if 'age' in self.data:
                ages=self.data['age'][times,ids]
                valid&=(ages[1:]==ages[:-1]+1).all(0)
            if 'ended' in self.data:
                valid&=~self.data['ended'][times[:-1],ids].any(0)
            candidates=valid.nonzero().flatten()
            if event_fraction and len(candidates) and 'event' in self.data:
                event=self.data['event'][times,ids].any(0)[candidates].float()
                weights=torch.full_like(event,(1-event_fraction)/len(event))
                weights+=event_fraction*event/event.sum().clamp_min(1)
                if not weights.sum():weights.fill_(1.)
                chosen=torch.multinomial(weights,remaining,replacement=True)
                candidates=candidates[chosen]
            chosen=candidates[:remaining]
            selected.append(starts[chosen]);envs.append(ids[chosen])
            remaining-=len(selected[-1])
            if remaining==0:break
        if remaining:raise RuntimeError('No sufficient valid native replay windows')
        starts=torch.cat(selected);ids=torch.cat(envs)
        times=(starts[None]+torch.arange(length,device=self.device)[:,None])%self.capacity
        return {k:v[times,ids] for k,v in self.data.items() if k not in ('observation','support','age')}
    def start(self):
        # Real early contact-approach states, with complete history and maintained supports.
        count=min(self.steps,self.capacity)
        for _ in range(100):
            t=torch.randint(count,(256,),device=self.device);e=torch.randint(self.num_envs,(256,),device=self.device)
            age=self.data['age'][t,e];obs=self.data['observation'][t,e]
            mask=(age>=5)&(age<=15)&self.data['valid'][t,e]&self.data['previous_known'][t,e].all(-1)&obs[:,3].bool()&obs[:,4].bool()
            if mask.any():
                i=mask.nonzero()[0,0];return {k:v[t[i:i+1],e[i:i+1]] for k,v in self.data.items()}
        raise RuntimeError('No valid early approach starts with required supports')

    def starts(self,batch):
        count=min(self.steps,self.capacity);selected=[];envs=[];remaining=batch
        for _ in range(100):
            size=max(256,remaining*4)
            t=torch.randint(count,(size,),device=self.device);e=torch.randint(self.num_envs,(size,),device=self.device)
            age=self.data['age'][t,e];obs=self.data['observation'][t,e]
            mask=(age>=5)&(age<=15)&self.data['valid'][t,e]&self.data['previous_known'][t,e].all(-1)&obs[:,3].bool()&obs[:,4].bool()
            ids=mask.nonzero().flatten()[:remaining]
            selected.append(t[ids]);envs.append(e[ids]);remaining-=len(ids)
            if remaining==0:break
        if remaining:raise RuntimeError('Insufficient valid native policy starts')
        t=torch.cat(selected);e=torch.cat(envs)
        keys=['state','observation','previous_known','previous_force_w','support','age']
        keys += [k for k in ('pano_frame','pano_view','env_id') if k in self.data]
        return {k:self.data[k][t,e] for k in keys}
