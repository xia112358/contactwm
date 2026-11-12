"""Whole-motion successor pools, with uninterrupted full-chain evaluation."""
import json,time,torch
from pathlib import Path
from supermimic.contact_task import ContactTask
from supermimic.contact_reset_pool import adjacent_successors
from stratified_pool import StratifiedPool as ResetStatePool

class PoolTask(ContactTask):
    def __init__(self,*args,**kwargs):
        self.ready=False;self.in_step=False;self.evaluating=False;self.captured=None
        super().__init__(*args,**kwargs)
        self.pool=ResetStatePool(capacity=256,max_age=10);self.pool_ratio=.5;self.iteration=0
        self.successor=adjacent_successors(self.clock.segment_starts,self.clock.segment_stops,self.clock.segment_motion)
        assert self.successor.tolist()==list(range(1,self.count))+[-1]
        self.original_segments=self.segment.clone();self.pool_source=torch.zeros(self.N,device=self.env.device,dtype=torch.bool)
        self.entry_origin=self.segment.clone();self.entry_depth=torch.zeros_like(self.segment)
        self.reset_counts=torch.zeros(self.count,2,device=self.env.device,dtype=torch.long);self.ready=True

    def _assign(self,segments):
        self.segment=segments.clone();self.starts=self.clock.segment_starts[self.segment];self.ends=self.clock.event_ends[self.segment]
        self.budgets=(self.clock.segment_stops-self.clock.segment_starts)[self.segment]+self.grace_steps
        self.contact_tolerance=self.contact_tolerances_by_segment[self.segment,None]
        self.reader.set_targets(self.clock.data['point'][self.ends-1],self.ids)

    def reset(self,slots,entry=None):
        if self.ready and self.in_step:
            if self.evaluating:return
            self.captured=(slots.clone(),self.history[slots].clone(),self.previous_contact[slots].clone(),self.env.f[self.ids[slots]].clone(),self.entry_origin[slots].clone(),self.entry_depth[slots].clone())
        super().reset(slots,entry)
        if not self.ready:return
        self.entry_origin[slots]=self.segment[slots] if entry is None else entry['origin']
        self.entry_depth[slots]=0 if entry is None else entry['depth']
        self.pool_source[slots]=False
        if self.evaluating:return
        for segment in self.segment[slots].unique().tolist():
            candidates=slots[self.segment[slots]==segment]
            selected=candidates[(torch.rand(len(candidates),generator=self.pool.generator)<self.pool_ratio).to(candidates.device)]
            if len(selected):
                payload=self.pool.sample(segment,len(selected),self.iteration)
                if payload is not None:
                    super().reset(selected,entry=payload);self.pool_source[selected]=True
                    # Validate the public physical state restoration, not solver caches.
                    from supermimic.mdp import flatten_state
                    actual=flatten_state(self.env.state())[self.ids[selected]]
                    error=(actual-payload['s']).abs().max()
                    if error>2e-4:raise AssertionError(f'Pool state restoration mismatch: {float(error)}')
            self.reset_counts[segment,0]+=(~self.pool_source[candidates]).sum();self.reset_counts[segment,1]+=self.pool_source[candidates].sum()

    def step(self):
        self.captured=None;self.in_step=True
        try:tr=super().step()
        finally:self.in_step=False
        if self.captured is not None and not self.evaluating:
            slots,history,active,f,origin,depth=self.captured;dest=self.successor[tr['goal_index'][slots]]
            good=tr['done'][slots]&~tr['term'][slots]&(dest>=0)
            for destination in dest[good].unique().tolist():
                use=good&(dest==destination);s=slots[use]
                self.pool.add(destination,dict(s=tr['s1'][s],a_prev=tr['a'][s],xi=tr['next_xi'][s],history=history[use],active=active[use],f=f[use],origin=origin[use],depth=depth[use]+1),self.iteration)
        return tr

    @torch.no_grad()
    def _evaluate(self,chain):
        slots=torch.arange(self.N,device=self.env.device);self._assign(torch.zeros_like(self.original_segments) if chain else self.original_segments);self.reset(slots)
        live=torch.ones(self.N,device=self.env.device,dtype=torch.bool);passed=torch.zeros_like(live);progress=torch.zeros(self.N,device=self.env.device,dtype=torch.long)
        limit=int(((self.clock.segment_stops-self.clock.segment_starts)+self.grace_steps).sum()) if chain else int(self.budgets.max())
        stages=torch.zeros(self.count,3,device=self.env.device,dtype=torch.long)
        for step in range(limit):
            tr=self.step();ended=live&tr['done'];success=ended&~tr['term'];seg=tr['goal_index']
            for s in seg[ended].unique().tolist():
                at=ended&(seg==s);stages[s,0]+=at.sum();stages[s,1]+=(at&success).sum();stages[s,2]+=(at&tr['term']).sum()
            advance=success&(seg<self.count-1) if chain else torch.zeros_like(live)
            progress+=success.long();terminal=ended&~advance;passed|=terminal&success;live&=~terminal
            changing=advance.nonzero().flatten()
            if len(changing):
                before=tr['s1'][changing].clone();new=self.segment.clone();new[changing]+=1;self._assign(new)
                self.env.ep_step[self.ids[changing]]=0
                initial=self.previous_contact[changing]
                self.entry_contact[changing]=initial
                required=self.clock.target_masks[self.ends[changing]-1]&~self.clock.keep_masks[self.ends[changing]-1]
                self.contact_monitor.begin_stage(self.ids[changing],initial,required,torch.zeros_like(initial,dtype=torch.int32))
                self.contact_confirming[changing]=False;self.contact_hold_action[changing]=0
                self.arrival.reset(changing)
                self.material_error_sq[changing]=float('inf')
                self.recontact[changing]=False;self.valid_contacts[changing]=False;self.onset_point[changing]=0;self.onset_face[changing]=-1
                # Handoff changes the command and its local deadline only.
                from supermimic.mdp import flatten_state
                error=(flatten_state(self.env.state())[self.ids[changing]]-before).abs().max()
                if error>2e-4:raise AssertionError(f'Chain handoff altered physical state: {float(error)}')
            if not live.any():break
            dead=(~live).nonzero().flatten()
            if len(dead):self.reset(dead)
        self.last_eval_passed=passed.clone()
        return dict(attempts=self.N,successes=int(passed.sum()),success_rate=float(passed.float().mean()),stages_completed_mean=float(progress.float().mean()),stages=stages.cpu().tolist(),steps=step+1,scope='full chain without handoff reset' if chain else 'each stage from its recorded reference start')

    def evaluate(self,output,iteration):
        stats={k:getattr(self,k).clone() for k in ('attempts','successes','root_sum','point_sum','point_count','last_terminal_steps')};self.evaluating=True
        try:
            result=dict(iteration=iteration,original=self._evaluate(False),chain=self._evaluate(True),pool_counts=self.pool.counts(iteration),reset_counts=self.reset_counts.cpu().tolist(),updated_at=time.time())
            with (Path(output)/'evaluation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print('FULL_CLIMB58_EVALUATION '+json.dumps(result),flush=True)
        finally:
            self._assign(self.original_segments);self.reset(torch.arange(self.N,device=self.env.device));self.evaluating=False
            for k,v in stats.items():getattr(self,k).copy_(v)
        return result
