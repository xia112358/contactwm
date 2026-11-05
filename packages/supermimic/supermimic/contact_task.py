"""Single-segment task adapter for SuperMimic's existing parallel simulator."""
import torch
from .contact_control import observation, new_part_contact
from .mdp import unflatten_state
from .rotations import rot_angle_sq


def has_fallen(state):
    """Physical fall: root near the floor or pelvis up axis points downward.

    This permits the deep forward lean in the low-speed references. It rejects
    inversion before the already-fallen robot reaches a singular floor impact.
    Numerical invalidity remains a separate hard failure in the collector.
    """
    return (state[0][...,2]<.15) | (state[1][...,2,2]<=0.)


class ContactTask:
    def __init__(self,env,actor,teacher,motion,mdp,clock,demo,context,default_q,reader,teacher_hist,goal_frames,grace_steps=10,segment_index=None,root_tolerance=.25,completion_mode='onset',contact_tolerances=None,sequential=False,sequence_length=None,contact_only=False,continuous_random=False,root_angle_tolerance=.4,first_contact_only=False,contact_window_frames=1,cache_reset_contacts=True,stable_contact_frames=0):
        if grace_steps < 0:
            raise ValueError('Segment grace must be nonnegative')
        if contact_only and (not sequential or completion_mode!='accumulated_target'):
            raise ValueError('Contact-only diagnostic requires continuous accumulated contact execution')
        self.__dict__.update(locals());self.__dict__.pop('self')
        self.fixed_goal_mode=getattr(clock,'fixed_goal_mode',False)
        self.ids=(env.task_ids==int(demo['xi'][0,33])).nonzero().flatten()
        self.background=(env.task_ids!=int(demo['xi'][0,33])).nonzero().flatten()
        self.N=len(self.ids);self.count=len(clock.event_ends);dev=env.device
        if segment_index is not None and not 0<=segment_index<self.count:
            raise ValueError('Segment index out of range')
        self.segment=torch.arange(self.N,device=dev)%self.count
        if segment_index is not None:self.segment.fill_(segment_index)
        if hasattr(clock,'segment_motion'):
            if sequential:raise ValueError('Whole-transfer libraries use independent episodes; motion boundaries cannot be chained')
            from .contact_library import case_world_assignment
            self.ids,self.segment,self.background=case_world_assignment(clock,env.task_ids,segment_index)
            self.N=len(self.ids)
        if sequential:
            if completion_mode!='accumulated_target' or not reader.direct:
                raise ValueError('Continuous execution uses the current accumulated native-contact criterion')
            self.segment.fill_(0 if segment_index is None else segment_index)
            if sequence_length is not None:
                if not 1<=sequence_length<=self.count:raise ValueError('Invalid sequence window')
                windows=self.count if continuous_random else self.count-sequence_length+1
                self.segment=torch.randint(windows,(self.N,),device=dev) if continuous_random else torch.arange(self.N,device=dev)%windows
                if segment_index is not None:
                    if segment_index>=windows:raise ValueError('Sequence start out of range')
                    self.segment.fill_(segment_index)
                self.initial_segment=self.segment.clone()
                self.sequence_last=torch.full_like(self.initial_segment,self.count-1) if continuous_random else self.initial_segment+sequence_length-1
                self.pair_attempts=torch.zeros(windows,device=dev)
                self.pair_successes=self.pair_attempts.clone();self.pair_root_sum=self.pair_attempts.clone();self.pair_point_sum=self.pair_attempts.clone()
            self.sequence_live=torch.ones(self.N,device=dev,dtype=torch.bool)
            self.sequence_completed=torch.zeros_like(self.sequence_live)
            self.sequence_progress=torch.zeros(self.N,device=dev,dtype=torch.long)
            self.stage_step=torch.zeros_like(self.sequence_progress)
        from .contact_library import segment_ranges
        all_starts,all_stops=segment_ranges(clock)
        self._sequence_starts=all_starts
        self._report_ranges=list(zip(all_starts.tolist(),all_stops.tolist()))
        self.starts=all_starts[self.segment]
        self.ends=clock.event_ends[self.segment]
        self.contact_tolerances_by_segment=torch.as_tensor(
            [.05]*self.count if contact_tolerances is None else contact_tolerances,device=dev)
        if self.contact_tolerances_by_segment.shape!=(self.count,) or not bool(
                (torch.isfinite(self.contact_tolerances_by_segment)&(self.contact_tolerances_by_segment>0)).all()):
            raise ValueError('Expected one positive contact tolerance per segment')
        self.contact_tolerance=self.contact_tolerances_by_segment[self.segment,None]
        self.budgets=(all_stops-all_starts)[self.segment]+grace_steps
        self.last_terminal_steps=all_stops-all_starts+grace_steps
        self.target_faces_by_segment=None if reader.direct else reader.target_faces(clock.data['point'][clock.event_ends-1],self.ids)
        self.target_faces=None if reader.direct else self.target_faces_by_segment[self.segment]
        reader.set_targets(clock.data['point'][self.ends-1],self.ids)
        self.attempts=torch.zeros(self.count,device=dev);self.successes=self.attempts.clone()
        self.root_sum=self.attempts.clone();self.point_sum=self.attempts.clone();self.point_count=self.attempts.clone()
        if stable_contact_frames:
            from .contact_phase import reference_release_starts,LandingPhaseTracker
            from .contact_stability import NativeContactStability
            from .motion import CONTROL_DT
            if not (reader.direct and reader.primary_faces):
                raise ValueError("Stable support requires the native primary-surface row view")
            from .contact_stability import CONTRACT
            if getattr(clock,'reference_support_contract',None)!=CONTRACT:
                raise ValueError('Stable support training requires rematerialized stable reference slices; old onset goals are incompatible')
            if self.fixed_goal_mode:
                from .contact_gpu_stability import GPUContactLossMonitor
                self.contact_monitor=GPUContactLossMonitor(env,reader)
            else:
                self.contact_stability=NativeContactStability(CONTROL_DT)
            if not self.fixed_goal_mode:
                clock.contact_phase_release_starts=reference_release_starts(clock,stable_contact_frames)
                required=clock.target_masks[self.ends-1]&~clock.keep_masks[self.ends-1]
                self.landing_phase=LandingPhaseTracker(clock.contact_phase_release_starts[self.segment],required,stable_contact_frames)
        self.previous_contact=torch.zeros((self.N,6),device=dev,dtype=torch.bool)
        self.recontact=torch.zeros_like(self.previous_contact)
        self.valid_contacts=torch.zeros_like(self.previous_contact)
        from .contact_arrival import ArrivalWindow
        self.arrival=ArrivalWindow(self.previous_contact.shape,dev,contact_window_frames)
        self.onset_point=torch.zeros((self.N,6,3),device=dev)
        self.onset_face=torch.full((self.N,6),-1,device=dev,dtype=torch.long)
        if hasattr(clock,'target_grid'):
            self.grid_last_points=torch.zeros_like(self.onset_point)
            self.grid_last_observed=torch.zeros_like(self.recontact)
            self.grid_last_completed=torch.zeros(self.N,device=dev,dtype=torch.bool)
        self.history=self.teacher_history=None
        self.pending_history_reset=torch.zeros(self.N,device=dev,dtype=torch.bool)
        self.entry_xi=self.demo['xi'][self.starts].clone()
        self._reset_history=None
        if hasattr(clock,'segment_starts'):
            from .contact_library import history_indices
            indices=history_indices(clock,all_starts,actor.hist)
            self._reset_history=observation(unflatten_state(demo['s'][indices].flatten(0,1)),
                demo['a_prev'][indices].flatten(0,1),demo['xi'][indices].flatten(0,1),
                default_q,context).reshape(self.count,actor.hist,-1).detach()
        # Runtime-only evidence for immutable reference entries on static paired terrain.
        # Keyed by physical world and segment; dynamic entries always bypass this cache.
        self.cache_reset_contacts=cache_reset_contacts and env.eval_mode and hasattr(clock,'segment_starts') and not sequential
        self._reset_contact_segment=torch.full((self.N,),-1,device=dev,dtype=torch.long)
        self._reset_contact_active=torch.zeros_like(self.previous_contact)
        self.initial_contact_probe=None
        if self.first_contact_only:
            from .contact_initial import InitialContactProbe
            self.initial_contact_probe=InitialContactProbe(reader)
        env.reset(torch.arange(env.N,device=dev),motion.clip_start[env.task_ids])
        self.reset(torch.arange(self.N,device=dev))

    def invalidate_reset_contacts(self):
        """Call after changing terrain or collision parameters; references are immutable."""
        self._reset_contact_segment.fill_(-1)

    def reset(self,slots,entry=None):
        """Reset once to an exact reference or a detached physical handoff entry.

        Handoff entries carry s, a_prev, xi, history and previous native active
        contacts. They never use the fixed-reference evidence/history caches.
        """
        if not len(slots):return
        env=self.env;ids=self.ids[slots];starts=self.starts[slots]
        state=self.demo['s'][starts] if entry is None else entry['s']
        frames=self.demo['f'][starts].long() if entry is None else entry['f']
        env.reset(ids,frames,state=unflatten_state(state))
        env.a_prev[ids]=self.demo['a_prev'][starts] if entry is None else entry['a_prev']
        env.xi[ids]=self.demo['xi'][starts] if entry is None else entry['xi']
        if entry is not None:
            env.robot.set_joint_position_target_index(
                target=env.default_q+entry['xi'][:,1:30]+env.alpha*entry['a_prev'],env_ids=ids)
        self.entry_xi[slots]=env.xi[ids]
        env.refresh()
        if hasattr(self.clock,'segment_starts'):
            hist=self._reset_history[self.segment[slots]] if entry is None else entry['history']
            if self.history is None:self.history=hist.new_empty((self.N,*hist.shape[1:]))
            self.history[slots]=hist;self.pending_history_reset[slots]=True
        if entry is not None:
            # Same last-solve evidence used by an uninterrupted command handoff.
            self.previous_contact[slots]=entry['active']
        elif self.initial_contact_probe is not None:
            missing=(self._reset_contact_segment[slots]!=self.segment[slots]) if self.cache_reset_contacts else torch.ones_like(slots,dtype=torch.bool)
            if bool(missing.any()):
                fresh=self.initial_contact_probe.read()
                update=slots[missing]
                self._reset_contact_active[update]=fresh[self.ids[update]]
                self._reset_contact_segment[update]=self.segment[update]
            self.previous_contact[slots]=self._reset_contact_active[slots]
        else:
            from .contact_library import initial_contact_indices
            initial=initial_contact_indices(self.clock,starts)
            self.previous_contact[slots]=self.demo['active'][initial]&self.demo['known'][initial]
        if self.stable_contact_frames and not self.fixed_goal_mode:self.landing_phase.reset(slots)
        self.arrival.reset(slots)
        self.recontact[slots]=False
        self.valid_contacts[slots]=False
        self.onset_point[slots]=0
        self.onset_face[slots]=-1

    def reset_sequence(self):
        """Reset only between full window attempts, never at the handoff."""
        if self.continuous_random and self.segment_index is None:
            self.initial_segment.random_(0,self.count)
        self.segment.copy_(self.initial_segment)
        if not hasattr(self,'_sequence_starts'):
            self._sequence_starts=torch.as_tensor(self.clock.bounds[:-1],device=self.segment.device)
        bounds=self._sequence_starts
        self.starts=bounds[self.segment];self.ends=self.clock.event_ends[self.segment]
        self.budgets=(self.fixed_budgets[self.segment] if hasattr(self,'fixed_budgets') else self.ends-self.starts+self.grace_steps)
        self.contact_tolerance=self.contact_tolerances_by_segment[self.segment,None]
        self.sequence_live.fill_(True);self.sequence_completed.zero_();self.sequence_progress.zero_();self.stage_step.zero_()
        self.reader.set_targets(self.clock.data['point'][self.ends-1],self.ids)
        self.reset(torch.arange(self.N,device=self.segment.device))

    @torch.no_grad()
    def step(self):
        env=self.env;ids=self.ids;S=env.state();current=tuple(x[ids] for x in S)
        if hasattr(self.clock,'segment_starts'):
            reset_slots=self.pending_history_reset.nonzero().flatten()
            env.xi[ids[reset_slots]]=self.entry_xi[reset_slots]
            self.pending_history_reset.zero_()
        policy_xi=env.xi[ids].clone() if hasattr(self.clock,'segment_starts') else None
        obs=observation(current,env.a_prev[ids],env.xi[ids],self.default_q,self.context)
        history=obs[:,None].expand(-1,self.actor.hist,-1)
        self.history=(torch.cat((self.history[:,1:],obs[:,None]),1) if hasattr(self.clock,'segment_starts') else
            history.clone() if self.history is None else torch.where(
            (env.ep_step[ids]==0)[:,None,None],history,torch.cat((self.history[:,1:],obs[:,None]),1)))
        entry_history=self.history.clone() if hasattr(self.clock,'segment_starts') else None
        actions=torch.zeros((env.N,29),device=env.device)
        actions[ids]=self.actor(self.history,self.clock.condition(current,self.starts,ends=self.ends))
        if self.sequential:actions[ids]=torch.where(self.sequence_live[:,None],actions[ids],0.)
        bg=self.background
        if len(bg):
            teacher_obs=self.mdp.obs(tuple(x[bg] for x in S),env.a_prev[bg],env.xi[bg],env.f[bg])
            history=teacher_obs[:,None].expand(-1,self.teacher_hist,-1)
            self.teacher_history=(history.clone() if self.teacher_history is None else torch.where(
                (env.ep_step[bg]==0)[:,None,None],history,torch.cat((self.teacher_history[:,1:],teacher_obs[:,None]),1)))
            actions[bg]=self.teacher(self.teacher_history,self.motion.goal(env.f[bg],self.goal_frames))
        # Library-entry restoration can overwrite sensor xi after the first
        # state read. Keep its original refresh path; ordinary episodes can reuse
        # the physical state and sensor context already read for the policy.
        step_kwargs={} if hasattr(self.clock,'segment_starts') else dict(precomputed_state=S)
        if getattr(self,'skip_student_tracking',False):step_kwargs['tracking_env_ids']=self.background
        if self.stable_contact_frames and self.fixed_goal_mode:
            initial=self.demo['active'][self.starts]&self.demo['known'][self.starts]
            self.contact_monitor.begin_control(ids,self.contact_tolerance,initial)
        out=env.step(actions,**step_kwargs)
        evidence={k:v[ids] for k,v in self.reader.read_device().items()}
        if self.stable_contact_frames:
            support=(self.contact_monitor.summary() if self.fixed_goal_mode else
                self.contact_stability.update(self.reader.last_view,self.reader.last_part_poses,env.ep_step))
            evidence.update({k:v[ids] for k,v in support.items()})
        state=unflatten_state(out['s1'][ids]);clock=self.clock
        target=clock.data['point'][self.ends-1];keep=clock.keep_masks[self.ends-1]
        new=clock.target_masks[self.ends-1]&~keep;support=None if self.reader.direct else clock.support_masks[self.ends-1]
        reference=unflatten_state(clock.data['s1'][self.ends-1])
        root=(state[0]-reference[0]).norm(dim=-1);angle=rot_angle_sq(state[1],reference[1]).sqrt()
        angle_ok=torch.ones_like(root,dtype=torch.bool) if self.root_angle_tolerance is None else angle<self.root_angle_tolerance
        failed=(~torch.isfinite(out['s1'][ids]).all(-1))|has_fallen(state)
        observed=evidence['known']&evidence['active']
        # This distance is also the native metric. Compute it once; accumulated
        # completion does not need the discarded onset-mode arrival calculation.
        contact_distance=(evidence['point']-target).norm(dim=-1)
        in_goal=observed&(contact_distance<=self.contact_tolerance)
        if self.target_faces is not None:
            in_goal&=(evidence['surface_local']==self.target_faces)&(self.target_faces>=0)
        kept=((~keep)|observed).all(-1)
        if support is not None:
            kept&=((~support)|(observed&(evidence['load']>=1))).all(-1)
        has_new=new.any(-1)
        first=observed&~self.previous_contact&~self.recontact&new
        self.onset_point=torch.where(first[...,None],evidence['point'],self.onset_point)
        self.onset_face=torch.where(first,evidence['surface_local'],self.onset_face)
        contact_end,self.recontact=new_part_contact(observed,self.previous_contact,self.recontact,new)
        self.previous_contact=observed.clone()
        valid_first=torch.zeros_like(new)
        if self.completion_mode=='accumulated_target':
            valid=observed if self.contact_only else in_goal
            if self.first_contact_only:
                eligible=torch.ones_like(observed) if self.target_faces is None else (evidence['surface_local']==self.target_faces)&(self.target_faces>=0)
                step=self.stage_step if self.sequential else out['ep_step'][ids]
                improved=self.arrival.update(first,observed,eligible,contact_distance,step[:,None],failed|(step+1>=self.budgets))
                self.onset_point=torch.where(improved[...,None],evidence['point'],self.onset_point)
                self.onset_face=torch.where(improved,evidence['surface_local'],self.onset_face)
                valid=self.arrival.closed&(self.arrival.distance<=self.contact_tolerance)&new
                valid_first=valid&~self.valid_contacts
                self.valid_contacts|=valid
            else:
                valid_first=valid&new&~self.valid_contacts
                self.valid_contacts|=valid&new
            arrived=((~new)|self.valid_contacts).all(-1)
            completed=arrived&(root<self.root_tolerance)&angle_ok&torch.where(has_new,True,kept)&~failed
            if self.contact_only:
                completed=arrived&torch.where(has_new,True,kept)&~failed
            contact_end=torch.zeros_like(completed)
        else:
            valid=self.recontact&(torch.ones_like(observed) if self.target_faces is None else self.onset_face==self.target_faces)&((self.onset_point-target).norm(dim=-1)<=self.contact_tolerance)
            arrived=torch.where(has_new,((~new)|valid).all(-1),((~new)|in_goal).all(-1))
            completed=arrived&(root<self.root_tolerance)&angle_ok&torch.where(has_new,contact_end,kept)&~failed
        if self.stable_contact_frames and not self.fixed_goal_mode:
            # Confirm qualified native support after the demonstrated release.
            # Scraping/row changes cannot latch a target or end the attempt.
            first=self.landing_phase.update(evidence['active'],evidence['known'],evidence['contact_stable'])
            self.onset_point=torch.where(first[...,None],evidence['point'],self.onset_point)
            self.onset_face=torch.where(first,evidence['surface_local'],self.onset_face)
            valid_first=first&in_goal
            self.valid_contacts=self.landing_phase.confirmed&((self.onset_point-target).norm(dim=-1)<=self.contact_tolerance)
            phase_ready=((~new)|self.landing_phase.confirmed).all(-1)&has_new
            phase_arrived=((~new)|self.valid_contacts).all(-1)
            completed=phase_ready&phase_arrived&(root<self.root_tolerance)&angle_ok&~failed
            completed=torch.where(has_new,completed,kept&(root<self.root_tolerance)&angle_ok&~failed)
            contact_end=phase_ready
        attempt_step=self.stage_step.clone() if self.sequential else out['ep_step'][ids]
        if getattr(self,'native_region_mode',False):
            # Region windows are evaluated from complete same-solve evidence
            # after collection; point-onset completion must not end them early.
            completed=torch.zeros_like(completed)
            contact_end=torch.zeros_like(contact_end)
        success=completed if not self.sequential else completed&self.sequence_live
        expired=attempt_step+1>=self.budgets
        # Accept success on the last allowed transition; otherwise end as failure.
        deadline_failed=expired&~completed
        done=contact_end|completed|failed|expired
        if self.sequential:done &= self.sequence_live
        # Replay insertion uses the original buffer's episode masks/history rules.
        transition={k:v[ids] for k,v in out.items() if isinstance(v,torch.Tensor) and v.shape[:1]==(env.N,)}
        if entry_history is not None:transition.update(actor_history=entry_history,xi=policy_xi)
        transition.update(evidence,contact_first=first.clone(),contact_goal=valid_first.clone(),next_xi=env.xi[ids].clone(),goal_index=self.segment.clone(),
                          term=failed|deadline_failed|(contact_end&~completed),done=done)
        if getattr(self,'native_region_mode',False):transition['term']=failed
        if self.sequential:
            transition.update(sequence_live=self.sequence_live.clone(),stage_step=attempt_step,
                              stage_done=done.clone(),stage_success=success.clone(),root_error=root,angle_error=angle)
            transition['term'] &= self.sequence_live
        ended=done.nonzero().flatten()
        # One fixed representative per segment supplies an actually observed
        # stopping time; do not infer contacts from imagined geometry/forces.
        if len(ended):
            segments=self.segment[ended]
            representative=ended<self.count if self.segment_index is None else ended==0
            slots=ended[representative]
            self.last_terminal_steps[self.segment[slots]]=attempt_step[slots]+1
            self.attempts.index_add_(0,segments,torch.ones(len(ended),device=env.device))
            self.successes.index_add_(0,segments,success[ended].float())
            self.root_sum.index_add_(0,segments,root[ended])
        metric_distance=contact_distance if self.completion_mode=='accumulated_target' else (self.onset_point-target).norm(dim=-1)
        if self.first_contact_only:metric_distance=self.arrival.distance
        error=metric_distance.masked_fill(~new,0).max(-1).values
        finite=torch.isfinite(error)
        if len(ended):
            self.point_sum.index_add_(0,segments,error[ended].masked_fill(~finite[ended],0))
            self.point_count.index_add_(0,segments,finite[ended].float())
        if len(ended) and hasattr(self.clock,'target_grid'):
            self.grid_last_points[ended]=self.onset_point[ended]
            self.grid_last_observed[ended]=self.recontact[ended]
            self.grid_last_completed[ended]=True
        self.current_root=root.mean();self.current_contact=error.masked_fill(~finite,0).sum()/finite.sum().clamp_min(1)
        self.current_contact_count=finite.sum()
        if self.sequential:
            self.stage_step+=self.sequence_live.long()
            self.finish_sequence(done,success,transition,ended=ended)
            if self.sequence_length is not None and len(ended):
                ended=transition['done'].nonzero().flatten();groups=self.initial_segment[ended]
                self.pair_attempts.index_add_(0,groups,torch.ones(len(ended),device=env.device))
                self.pair_successes.index_add_(0,groups,self.sequence_completed[ended].float())
                self.pair_root_sum.index_add_(0,groups,root[ended]);self.pair_point_sum.index_add_(0,groups,error[ended])
        elif len(ended):self.reset(ended)
        reset=bg[out['done'][bg]]
        if len(reset):
            env.reset(reset,self.motion.clip_start[env.task_ids[reset]]);env.refresh()
        env.f=torch.minimum(env.f,self.motion.clip_end[env.task_ids]-1)
        return transition

    def finish_sequence(self,stage_done,success,transition,*,ended=None):
        """Change only the command after success; never reset physics or history."""
        # step already compacted the stage-ending slots for its counters. On
        # quiet frames success is necessarily false too; avoid compacting again.
        if ended is not None and not len(ended):
            transition['done']=stage_done
            return
        last=getattr(self,'sequence_last',self.count-1)
        final=stage_done&success&(self.segment==last)
        terminal=stage_done&(~success|(self.segment==last))
        advance=(stage_done&success&~final).nonzero().flatten()
        self.sequence_progress+=success.long()
        self.sequence_completed|=final
        self.sequence_live&=~terminal
        transition['done']=terminal
        if not len(advance):return
        self.segment[advance]+=1
        bounds=getattr(self,'_sequence_starts',None)
        if bounds is None:bounds=torch.as_tensor(self.clock.bounds,device=self.segment.device)
        self.starts[advance]=bounds[self.segment[advance]]
        self.ends[advance]=self.clock.event_ends[self.segment[advance]]
        self.budgets[advance]=(self.fixed_budgets[self.segment[advance]] if hasattr(self,'fixed_budgets') else self.ends[advance]-self.starts[advance]+self.grace_steps)
        self.contact_tolerance[advance]=self.contact_tolerances_by_segment[self.segment[advance],None]
        self.stage_step[advance]=0
        self.arrival.reset(advance)
        self.valid_contacts[advance]=False
        self.recontact[advance]=False
        self.onset_point[advance]=0
        self.onset_face[advance]=-1
        self.reader.update_targets(self.clock.data['point'][self.ends[advance]-1],self.ids[advance])

    def grid_report(self):
        if not hasattr(self.clock,'target_grid'):return {}
        rows=[]
        for index in range(self.count):
            slots=((self.segment==index)&self.grid_last_completed).nonzero().flatten()[:64]
            rows.append(dict(target_index=index,points_w=self.grid_last_points[slots].cpu().tolist(),
                observed=self.grid_last_observed[slots].cpu().tolist()))
        return dict(target_grid=self.clock.target_grid,landing_distribution=rows,
            landing_scope=f'last completed attempt; first contiguous native contact window ({self.contact_window_frames} frames); missing explicitly masked')

    def report(self,iteration,updates):
        if self.sequential and self.sequence_length is not None:
            rows=[]
            # One device read instead of four scalar synchronizations per group.
            counts=torch.stack((self.pair_attempts,self.pair_successes,self.pair_root_sum,self.pair_point_sum),-1).cpu().tolist()
            for i,(attempts,wins,root_sum,point_sum) in enumerate(counts):
                n=int(attempts);wins=int(wins)
                rows.append(dict(segment=i,label=f'{i+1}→{self.count if self.continuous_random else i+self.sequence_length}',attempts=n,successful_attempts=wins,
                    success_rate=wins/n if n else None,final_root_error_m=root_sum/n if n else None,
                    final_contact_error_m=point_sum/n if n else None,
                    budget_steps=self.clock.bounds[self.count if self.continuous_random else i+self.sequence_length]-self.clock.bounds[i]+self.grace_steps*(self.count-i if self.continuous_random else self.sequence_length),terminal_steps=None))
            total=sum(r['attempts'] for r in rows);wins=sum(r['successful_attempts'] for r in rows)
            covered=[r for r in rows if r['attempts']]
            return dict(iteration=iteration,policy_updates=updates,independent_segments=True,sequence_length=self.sequence_length,
                training_collection=True,separate_evaluation=False,total_worlds=self.env.N,parallel_student_worlds=self.N,
                total_goals=len(rows),attempts=total,successful_attempts=wins,success=False,success_rate=wins/total if total else None,
                macro_success_rate=sum(r['success_rate'] for r in covered)/len(covered) if covered else None,covered_goals=len(covered),
                mean_root_error_m=sum(v[2] for v in counts)/total if total else None,segments=rows,
                metric_scope='continuous adjacent windows; resets only between windows; both stages required')
        # Reporting runs repeatedly during collection. Read all scalar counters
        # together instead of synchronizing separately for every segment/field.
        packed=torch.cat((torch.stack((self.attempts,self.successes,self.root_sum,self.point_sum,
            self.contact_tolerances_by_segment,self.last_terminal_steps,self.point_count),0).flatten(),
            torch.stack((self.current_root,self.current_contact,self.current_contact_count))))
        values=packed.cpu().tolist();n,s,root,point,tolerances,terminals,point_count=[values[i*self.count:(i+1)*self.count] for i in range(7)]
        total=sum(n);wins=sum(s)
        covered=[i for i in range(self.count) if n[i]>0]
        from .contact_library import segment_ranges
        if not hasattr(self,'_report_ranges'):
            report_starts,report_stops=segment_ranges(self.clock)
            self._report_ranges=list(zip(report_starts.tolist(),report_stops.tolist()))
        return dict(iteration=iteration,policy_updates=updates,independent_segments=not self.sequential,
            training_collection=not self.sequential,separate_evaluation=self.sequential,total_worlds=self.env.N,parallel_student_worlds=self.N,
            total_goals=self.count if self.segment_index is None else 1,attempts=int(total),successful_attempts=int(wins),
            success_rate=wins/total if total else None,success=False,
            macro_success_rate=sum(s[i]/n[i] for i in covered)/len(covered) if covered else None,
            covered_goals=len(covered),
            mean_root_error_m=sum(root)/total if total else None,
            current_root_error_m=values[-3],current_contact_error_m=values[-2] if values[-1] else None,contact_error_samples=int(values[-1]),
            segments=[dict(segment=i,attempts=int(n[i]),successful_attempts=int(s[i]),
                contact_tolerance_m=tolerances[i],
                budget_steps=clock_end-clock_start+self.grace_steps,
                terminal_steps=int(terminals[i]),
                success_rate=s[i]/n[i] if n[i] else None,
                final_root_error_m=root[i]/n[i] if n[i] else None,
                final_contact_error_m=point[i]/point_count[i] if point_count[i] else None,contact_error_samples=int(point_count[i]))
                for i,(clock_start,clock_end) in enumerate(self._report_ranges)
                if self.segment_index is None or i==self.segment_index],
            metric_scope='completed training attempts since this run started; exact recorded segment starts',**self.grid_report())
