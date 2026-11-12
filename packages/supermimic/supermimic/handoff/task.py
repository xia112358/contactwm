"""Bootstrap reference entries; subsequent training successes refill the pools."""
import torch
from pure_task import PurePoolTask


class ReferenceProducerTask(PurePoolTask):
    @torch.no_grad()
    def populate_reference(self):
        stats={k:getattr(self,k).clone() for k in ('attempts','successes','root_sum','point_sum','point_count','last_terminal_steps')}
        self.evaluating=True
        try:
            self._assign(self.original_segments);self.reset(torch.arange(self.N,device=self.env.device))
            live=torch.ones(self.N,device=self.env.device,dtype=torch.bool)
            attempts=torch.zeros(self.count,device=self.env.device,dtype=torch.long);successes=attempts.clone()
            for _ in range(int(self.budgets.max())):
                tr=self.step();ended=live&tr['done'];passed=ended&~tr['term'];segment=tr['goal_index'];dest=self.successor[segment]
                attempts.index_add_(0,segment[ended],torch.ones_like(segment[ended]))
                successes.index_add_(0,segment[passed],torch.ones_like(segment[passed]))
                for destination in dest[passed&(dest>=0)].unique().tolist():
                    slots=(passed&(dest==destination)).nonzero().flatten()
                    self.pool.add(destination,dict(s=tr['s1'][slots],a_prev=tr['a'][slots],xi=tr['next_xi'][slots],history=self.history[slots],active=self.previous_contact[slots],f=self.env.f[self.ids[slots]],origin=segment[slots],depth=torch.ones_like(segment[slots])),self.iteration)
                live &= ~ended
                if not live.any():break
                dead=(~live).nonzero().flatten()
                if len(dead):self.reset(dead)
            self.producer_stats=dict(attempts=attempts.cpu().tolist(),successes=successes.cpu().tolist(),iteration=self.iteration)
        finally:
            self._assign(self.original_segments);self.evaluating=False
            for key,value in stats.items():getattr(self,key).copy_(value)
        return self.producer_stats


    @torch.no_grad()
    def validate_pool(self,output):
        import json
        from pathlib import Path
        rounds=[]
        for cycle in range(6):
            deep={k:{name:v.clone() for name,v in value.items()} for k,value in self.pool.entries.items() if k[2]>1}
            self.populate_reference()
            for key,value in deep.items():
                assert all(torch.equal(value[name],self.pool.entries[key][name]) for name in value),'Reference refill overwrote deep entries'
            self.reset(torch.arange(self.N,device=self.env.device))
            seen=torch.zeros(self.N,dtype=torch.bool,device=self.env.device)
            for _ in range(int(self.budgets.max())):
                tr=self.step();seen |= tr['done']
                if seen.all():break
            assert seen.all()
            rounds.append(dict(cycle=cycle,pool=self.pool.composition()))
        assert any(k[2]>1 for k in self.pool.entries),'No genuine multi-stage entry produced'
        sampling=[]
        for dest in self.pool.counts(self.iteration):
            payload=self.pool.sample(dest,306,self.iteration);keys=torch.stack((payload['origin'],payload['depth']),1);pairs,counts=keys.unique(dim=0,return_counts=True)
            available=torch.tensor([len(self.pool.entries[(dest,int(k[0]),int(k[1]))]['s']) for k in pairs],device=counts.device)
            total_available=self.pool.counts(self.iteration)[dest]
            if total_available>=306:
                assert bool((counts<=available).all())
                unsaturated=counts[counts<available]
                if len(unsaturated)>1:assert int(unsaturated.max()-unsaturated.min())<=1
            sampling.append(dict(case=dest+82,groups=[dict(origin=int(k[0])+82,depth=int(k[1]),samples=int(n)) for k,n in zip(pairs,counts)]))
        Path(output,'pool-validation.json').write_text(json.dumps(dict(rounds=rounds,sampling=sampling,deep_preserved=True,sampling_contract="capacity_capped_source_balance_without_replacement"),indent=2))
        torch.save(self.pool.state_dict(),Path(output,'initial-pool.pt'))
        print('PASS_POOL_VALIDATION',json.dumps(sampling),flush=True)
