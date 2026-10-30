"""Matched old/new BPTT batching on exactly the same simulator data; no updates."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,json,time,runpy
from pathlib import Path
pass
import torch
import supermimic.contact_sequence as sequence
from supermimic.contact_dmo import replay_loss
original=sequence.stage_batches
output=Path(sys.argv[sys.argv.index('--output')+1])
recording=Path(sys.argv[sys.argv.index('--recording')+1])
prior_path=Path(sys.argv[sys.argv.index('--prior_checkpoint')+1])

def benchmark(buffer,rows,plan,actor,context,default_q,batch_size):
    import inspect,copy
    caller=inspect.currentframe().f_back
    # Called directly while creating the generator in contact_train.main.
    scope=caller.f_locals
    normalize_xi=scope['normalize_xi'];wm=scope['wm'];clock=scope['clock'];geometry=scope['geometry'];prior=scope['prior'];args=scope['args']
    reports=[];reference=None
    import supermimic.networks as networks
    original_amp=networks.AMP_DTYPE
    for repeat,sorted_batches in enumerate([False,True,False,True,False,True]):
        if repeat==4:networks.AMP_DTYPE=None;reference=None
        actor.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.perf_counter()
        loss_value=0.;shapes=[];peak_before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        for data,stops,history,weight,total in original(buffer,rows,plan,actor,context,default_q,batch_size,sort_by_duration=sorted_batches):
            value,details=replay_loss(actor,wm,clock,geometry,context,normalize_xi,default_q,data,
                prior_actor=prior,prior_weight=args.prior_weight,pose_weight=args.pose_weight,
                completion_mode=args.completion_mode,stop_steps=stops,initial_history=history)
            (value*weight).backward();loss_value+=float(value.detach())*weight
            shapes.append([len(stops),int(stops.max())])
        torch.cuda.synchronize();seconds=time.perf_counter()-t0
        grad=torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
        if reference is None:reference=(loss_value,grad.clone())
        delta=(grad-reference[1]).norm()/reference[1].norm().clamp_min(1e-20)
        assert abs(loss_value-reference[0])<1e-4 and float(delta)<(1e-3 if networks.AMP_DTYPE is not None else 1e-4),(loss_value,float(delta))
        report=dict(sorted=sorted_batches,wm_precision=str(networks.AMP_DTYPE),seconds=seconds,loss=loss_value,gradient_relative_difference=float(delta),batches=shapes,
            padded_steps=sum(b*t for b,t in shapes),peak_extra_gb=(torch.cuda.max_memory_allocated()-peak_before)/1e9)
        reports.append(report);print(json.dumps(report),flush=True)
    actor.zero_grad(set_to_none=True)
    (output/'batching-benchmark.json').write_text(json.dumps(reports,indent=2))
    state=json.loads((output/'status.json').read_text());state.update(phase='complete',benchmark_only=True,actor_updates_performed=0);(output/'status.json').write_text(json.dumps(state,indent=2))
    raise SystemExit(0)
sequence.stage_batches=benchmark
runpy.run_module('supermimic.contact_train',run_name='__main__')
