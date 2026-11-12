"""Fit a small diagnostic WM to recorded native transitions; never certify BPTT.

Single-trajectory train/validation is only an integration diagnostic. Contact
switching perturbation data and simulator gradient probes remain required.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from holosoma.contact_wm.core import ContactWorldModel,fit_loss

@dataclass(frozen=True)
class Options:
    recording:Path
    output_dir:Path
    steps:int=100
    batch:int=32
    seed:int=7


def main():
    import tyro
    o=tyro.cli(Options)
    if min(o.steps,o.batch)<1:raise ValueError('Positive training sizes required')
    out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(o.seed);torch.set_num_threads(1)
    z=np.load(o.recording/'transitions.npz',allow_pickle=False)
    meta=json.loads((o.recording/'report.json').read_text())
    if meta['schema']!='contact_wm_native_transitions_v1':raise ValueError('Native recording required')
    x={k:torch.from_numpy(z[k].copy()) for k in z.files}
    n=len(x['state']);cut=int(n*.7)
    train=torch.arange(1,cut);val=torch.arange(cut+1,n)
    train=train[x['transition_valid'][train]];val=val[x['transition_valid'][val]]
    if len(train)<16 or len(val)<8:raise ValueError('Too few real transitions for held-out diagnostic')
    m=ContactWorldModel(63);opt=torch.optim.Adam(m.parameters(),lr=2e-4)
    def prediction(ids):return m(x['state'][ids],x['action'][ids],x['observation'][ids,:6],x['observation'][ids,39:102])
    def loss(ids):return fit_loss(prediction(ids),x['next_state'][ids],x['active'][ids],x['force_w'][ids],x['known'][ids],x['transition_valid'][ids])
    with torch.no_grad():initial=float(loss(train))
    for step in range(o.steps):
        ids=train[torch.randint(len(train),(o.batch,))];opt.zero_grad(set_to_none=True);value=loss(ids)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite native WM loss')
        value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    with torch.no_grad():
        p=prediction(val);switch=(x['active'][val]!=x['active'][val-1]).any(-1)
        error=(p['state']-x['next_state'][val]).square().mean(-1).sqrt()
        report=dict(training_initial_loss=initial,training_final_loss=float(loss(train)),validation_loss=float(loss(val)),
            validation_state_rmse=float(error.mean()),switch_samples=int(switch.sum()),
            switch_state_rmse=float(error[switch].mean()) if switch.any() else None,
            steady_state_rmse=float(error[~switch].mean()) if (~switch).any() else None,
            train_samples=len(train),validation_samples=len(val),steps=o.steps,
            gradient_alignment=None,optimization_allowed=False,
            limitation='one trajectory, one-step validation, no perturbed-action or held-out-terrain coverage; no simulator gradient validation',
            policy_updated=False,recording=str(o.recording))
    torch.save(dict(schema='contact_wm_diagnostic_v1',wm=m.state_dict(),config=m.config,optimizer=opt.state_dict(),
        iteration=o.steps,seed=o.seed,torch_rng_state=torch.get_rng_state(),validation=report,robot_asset=meta['robot_asset']),out/'model.pt')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
