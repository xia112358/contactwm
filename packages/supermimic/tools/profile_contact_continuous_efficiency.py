"""Frozen continuous collection and paired stage extraction/BPTT profiling.

Use contact_train's --continuous_random --fixed_inherited_duration options and
both freeze flags with --profile_collection. No optimizer updates or replay saves.
"""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
import torch
from supermimic import contact_profile

assert all(flag in sys.argv for flag in ('--continuous_random','--fixed_inherited_duration',
    '--profile_collection','--freeze_actor','--freeze_wm'))
reference_dir=Path('/tmp/continuous-efficiency-before')
if '--efficiency_reference_dir' in sys.argv:
    i=sys.argv.index('--efficiency_reference_dir');reference_dir=Path(sys.argv[i+1]);del sys.argv[i:i+2]
started=time.perf_counter()


def reference(name):
    spec=importlib.util.spec_from_file_location('supermimic._continuous_reference_'+name,reference_dir/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def forbidden(*args,**kwargs):raise AssertionError('Optimizer updates are disabled in this efficiency probe')
torch.optim.Adam.step=forbidden


class ContinuousProfiler:
    def wrap(self,obj,attr,label):pass

    def write(self,path):
        frame=inspect.currentframe().f_back;scope=frame.f_locals
        args=scope['args'];task=scope['task'];buffer=scope['buffer'];actor=scope['actor'];wm=scope['wm']
        assert args.continuous_random and args.fixed_inherited_duration
        from supermimic import contact_sequence as sequence
        from supermimic.contact_dmo import replay_loss
        from supermimic.contact_duration import FixedDurationBand
        old_sequence=reference('contact_sequence');old_duration=reference('contact_duration')
        initial_seconds=time.perf_counter()-started
        # Readiness includes one warm-up of simulation/kernel paths; no weights changed.
        rng=torch.cuda.get_rng_state();collections=[]
        for repeat,skip in enumerate((False,True,True,False)):
            torch.cuda.set_rng_state(rng);task.skip_student_tracking=skip
            torch.cuda.synchronize();begin=time.perf_counter();task.reset_sequence()
            count=scope['collect'](scope['config']['horizon'],scope['start'],until_first_done=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            live=buffer.recent(count,('sequence_live',))['sequence_live']
            active=live.sum(1).cpu().tolist()
            collections.append(dict(skip_unused_student_tracking=skip,seconds=elapsed,frames=count,
                active_transitions=sum(active),allocated_transitions=count*task.N,
                active_fraction=sum(active)/(count*task.N),active_by_frame=active))
            print(json.dumps(dict(phase='collection',**{k:v for k,v in collections[-1].items() if k!='active_by_frame'})),flush=True)
        rows=(buffer.ptr-count+torch.arange(count,device=scope['device']))%buffer.T
        fields=('done','term','s1','goal_index','active','known','point','stage_step','stage_done','stage_success')
        # Both implementations below use this exact same native collection.
        raw=buffer.recent(count,fields)
        halfwidth=args.duration_halfwidth or 0;centers=scope['duration_history'].centers
        plans=[];planning=[]
        for label,module,history in [('before',old_sequence,old_duration.FixedDurationBand(centers)),
                                    ('after',sequence,FixedDurationBand(centers))]:
            times=[]
            for _ in range(4):
                torch.cuda.synchronize();begin=time.perf_counter()
                selected,plan,audit=module.sequence_plan(raw,args.sequence_length,history,halfwidth,args.duration_min_success,args.duration_min_mass)
                torch.cuda.synchronize();times.append(time.perf_counter()-begin)
            plans.append((selected,plan,audit));planning.append(dict(revision=label,seconds=times,median_seconds=statistics.median(times[1:])))
        assert torch.equal(plans[0][0],plans[1][0])
        for key,value in plans[0][1].items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,plans[1][1][key]),key
            else:assert value==plans[1][1][key],key
        assert plans[0][2]==plans[1][2],(plans[0][2],plans[1][2])
        plan=plans[1][1];data_reads=[]
        for label,fn in [('before',lambda:{k:buffer.data[k][rows] for k in fields}),('after',lambda:buffer.recent(count,fields))]:
            times=[]
            for _ in range(5):
                torch.cuda.synchronize();begin=time.perf_counter();value=fn();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
            for key in fields:assert torch.equal(value[key],raw[key])
            data_reads.append(dict(revision=label,median_seconds=statistics.median(times),seconds=times))
        del value
        actor.requires_grad_(True);backward=[];expected_grad=None;expected_loss=None
        for repeat,label in enumerate(('before','after','after','before')):
            module=old_sequence if label=='before' else sequence
            actor.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            shapes=[];objective=0.;prep_seconds=0.
            batches=iter(module.stage_batches(buffer,rows,plan,actor,scope['context'],scope['default_q'],args.dmo_microbatch))
            while True:
                torch.cuda.synchronize();prep_begin=time.perf_counter()
                try:data,stops,history,weight,total=next(batches)
                except StopIteration:break
                torch.cuda.synchronize();prep_seconds+=time.perf_counter()-prep_begin
                value,details=replay_loss(actor,wm,scope['clock'],scope['geometry'],scope['context'],scope['normalize_xi'],scope['default_q'],data,
                    prior_actor=scope['prior'],prior_weight=args.prior_weight,pose_weight=args.pose_weight,
                    completion_mode=args.completion_mode,stop_steps=stops,initial_history=history,contact_surface=scope['surface'])
                (value*weight).backward();objective+=float(value.detach())*weight
                shapes.append([len(stops),int(stops.max())])
                del value,data,history
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            gradient=torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
            if expected_grad is None:expected_grad=gradient.clone();expected_loss=objective
            relative=float((gradient-expected_grad).norm()/expected_grad.norm().clamp_min(1e-20))
            assert abs(objective-expected_loss)<1e-5 and relative<1e-4,(objective,expected_loss,relative)
            backward.append(dict(revision=label,seconds=elapsed,stage_prepare_seconds=prep_seconds,loss=objective,
                gradient_relative_error=relative,batches=shapes,retained_stages=total,padded_steps=sum(b*t for b,t in shapes),
                peak_extra_gib=(torch.cuda.max_memory_allocated()-baseline)/2**30))
            print(json.dumps(dict(phase='stage_bptt',**backward[-1])),flush=True)
        actor.zero_grad(set_to_none=True);actor.requires_grad_(False)
        native=sequence.sequence_attempt_metrics(raw,scope['clock'],args.sequence_length,continuous=True)
        report=dict(scope='Frozen continuous handoff collection; paired stages and gradients from identical native data; shared GPU',
            actor_updates=0,wm_updates=0,worlds=args.num_envs,student_worlds=task.N,startup_and_warmup_seconds=initial_seconds,
            fixed_centers=centers,fixed_budgets=task.fixed_budgets.cpu().tolist(),halfwidth=halfwidth,
            collection=collections,planning=planning,replay_reads=data_reads,stage_bptt=backward,
            duration_filter=plans[1][2],native=native,
            median_bptt_seconds={label:statistics.median(v['seconds'] for v in backward if v['revision']==label) for label in ('before','after')})
        Path(path).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(phase='complete',median_bptt_seconds=report['median_bptt_seconds'],planning=planning)),flush=True)


contact_profile.CollectionProfiler=ContinuousProfiler
runpy.run_module('supermimic.contact_train',run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1]);state=json.loads((output/'status.json').read_text())
assert state['actor_frozen_unchanged'] and state['wm_frozen_unchanged']
path=output/'collection-profile.json';report=json.loads(path.read_text());report['weights_verified_unchanged']=True
path.write_text(json.dumps(report,indent=2)+'\n')
