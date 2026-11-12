"""Source/depth sampling unchanged; apply all independent resets in one native batch."""
import torch
from supermimic.contact_task import ContactTask
from supermimic.mdp import flatten_state
from task_base import PoolTask

class PurePoolTask(PoolTask):
    def reset(self,slots,entry=None):
        if not self.ready or self.evaluating:
            return PoolTask.reset(self,slots,entry)
        if not len(slots):return
        if self.in_step:
            self.captured=(slots.clone(),self.history[slots].clone(),self.previous_contact[slots].clone(),self.env.f[self.ids[slots]].clone(),self.entry_origin[slots].clone(),self.entry_depth[slots].clone())
        assigned=self.original_segments[slots]
        self.segment[slots]=assigned
        self.starts[slots]=self.clock.segment_starts[assigned];self.ends[slots]=self.clock.event_ends[assigned]
        self.budgets[slots]=(self.clock.segment_stops-self.clock.segment_starts)[assigned]+self.grace_steps
        self.contact_tolerance[slots]=self.contact_tolerances_by_segment[assigned,None]
        self.reader.update_targets(self.clock.data['point'][self.ends[slots]-1],self.ids[slots])
        self.pool_source[slots]=False
        available=self.pool.counts(self.iteration);entries=[];handoffs=[];states=[]
        for segment in range(self.count):
            selected=slots[assigned==segment]
            if not len(selected):continue
            if segment and available.get(segment,0)>0:
                payload=self.pool.sample(segment,len(selected),self.iteration)
                entries.append((selected,payload));handoffs.append(selected);states.append(payload['s'])
                self.pool_source[selected]=True
                self.entry_origin[selected]=payload['origin'];self.entry_depth[selected]=payload['depth']
                self.reset_counts[segment,1]+=len(selected)
            else:
                entries.append((selected,None))
                self.entry_origin[selected]=segment;self.entry_depth[selected]=0
                self.reset_counts[segment,0]+=len(selected)
        ContactTask.reset_batch(self,entries)
        if handoffs:
            use=torch.cat(handoffs);expected=torch.cat(states)
            error=(flatten_state(self.env.state())[self.ids[use]]-expected).abs().max()
            if error>2e-4:raise AssertionError(f'Pool state restoration mismatch: {float(error)}')
