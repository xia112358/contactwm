"""Frozen real collection plus WM/DMO backward timings; never take an optimizer step.

Use contact_train arguments with --profile_collection --freeze_actor --freeze_wm.
Only diagnostic JSON is written; existing checkpoints are read-only inputs.
"""
import inspect
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
import torch
from supermimic import contact_profile

assert all(flag in sys.argv for flag in ('--profile_collection','--freeze_actor','--freeze_wm'))
started=time.perf_counter()


def no_optimizer_step(*args,**kwargs):
    raise AssertionError('This efficiency probe forbids optimizer updates')


torch.optim.Adam.step=no_optimizer_step


class PipelineProfiler:
    def wrap(self,obj,attr,label):
        # Preserve normal scheduling: nested synchronization is deliberately absent.
        pass

    def write(self,path):
        frame=inspect.currentframe().f_back
        scope=frame.f_locals
        args=scope['args'];task=scope['task'];buffer=scope['buffer']
        actor=scope['actor'];wm=scope['wm'];collect=scope['collect'];device=scope['device']
        assert args.policy_gradient=='dmo' and args.sequence_length==1
        horizon=scope['config']['horizon']
        assert buffer.T>=horizon
        initial_seconds=time.perf_counter()-started
        collections=[]
        for repeat in range(2):
            torch.cuda.synchronize();begin=time.perf_counter()
            task.reset(torch.arange(task.N,device=device))
            count=collect(horizon,scope['start'],until_first_done=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            collections.append(dict(frames=count,seconds=elapsed,worlds=task.N))
        rows=(buffer.ptr-count+torch.arange(count,device=device))%buffer.T
        needed=('s','s1','xi','next_xi','a','a_prev','ep_step','goal_index','done',
                'contact_goal','contact_first','active','known','point')
        torch.cuda.synchronize();begin=time.perf_counter()
        batch={k:buffer.data[k][rows] for k in needed}
        torch.cuda.synchronize();prepare_seconds=time.perf_counter()-begin
        history=frame.f_locals['collection_initial_history']
        from supermimic.contact_dmo import replay_loss,native_attempt_metrics
        torch.cuda.synchronize();begin=time.perf_counter()
        native=native_attempt_metrics(dict(batch,term=buffer.data['term'][rows]),scope['clock'],task.target_faces_by_segment)
        torch.cuda.synchronize();metrics_seconds=time.perf_counter()-begin

        wm.requires_grad_(True)
        w=buffer.sample_windows(args.wm_batch,1,fields=('s','s1','a','xi','next_xi'))
        wm_times=[]
        for repeat in range(3):
            wm.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
            loss=scope['wm_one_step_loss'](wm,w,scope['normalize_xi']);loss.backward()
            with torch.no_grad():scope['wm_one_step_loss'](wm,w,scope['normalize_xi'])
            torch.cuda.synchronize();wm_times.append(time.perf_counter()-begin)
        wm.zero_grad(set_to_none=True);wm.requires_grad_(False)
        actor.requires_grad_(True);backward=[]
        for repeat in range(2):
            actor.zero_grad(set_to_none=True);torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            loss,details=replay_loss(actor,wm,scope['clock'],scope['geometry'],scope['context'],
                scope['normalize_xi'],scope['default_q'],batch,prior_actor=scope['prior'],
                prior_weight=args.prior_weight,pose_weight=args.pose_weight,completion_mode=args.completion_mode,
                initial_history=history,contact_surface=scope['surface'])
            loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            finite=all(bool(torch.isfinite(p.grad).all()) for p in actor.parameters() if p.grad is not None)
            backward.append(dict(seconds=elapsed,loss=float(loss.detach()),finite_gradient=finite,
                peak_extra_gib=(torch.cuda.max_memory_allocated()-baseline)/2**30,details=details))
            assert finite
        actor.zero_grad(set_to_none=True);actor.requires_grad_(False)
        report=dict(scope='Frozen real simulator collection and gradient computation only; no optimizer or checkpoint writes',
            num_envs=args.num_envs,student_worlds=task.N,actor_updates=0,wm_updates=0,
            startup_and_warmup_seconds=initial_seconds,collection=collections,
            replay_prepare_seconds=prepare_seconds,metrics_seconds=metrics_seconds,
            wm_fit_forward_backward_and_postcheck_seconds=wm_times,dmo=backward,
            native=native,contact_position_loss=args.contact_position_loss)
        Path(path).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


contact_profile.CollectionProfiler=PipelineProfiler
runpy.run_module('supermimic.contact_train',run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1])
status=json.loads((output/'status.json').read_text())
assert status['actor_frozen_unchanged'] and status['wm_frozen_unchanged']
path=output/'collection-profile.json';report=json.loads(path.read_text())
report['weights_verified_unchanged_by_entrypoint']=True
path.write_text(json.dumps(report,indent=2)+'\n')
