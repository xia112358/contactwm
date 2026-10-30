"""Fixed-data segment-one regression probe; keeps accepted policies untouched."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,json,sys,time
from pathlib import Path
pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args()
    import torch
    from supermimic.contact_control import ContactActor
    from supermimic.robot import G1_JOINT_NAMES
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    dev='cuda:0';a.output.mkdir(exist_ok=False,parents=True)
    cfg=json.loads((a.run/'config.json').read_text())
    paths={}
    parent=Path(cfg['dagger_resume']).parent
    for directory in [parent,a.run/'dagger']:
        for path in sorted(directory.glob('rollout_*.pt')):paths[int(path.stem.split('_')[-1])]=path
    rows={i:torch.load(path,map_location=dev,weights_only=False)['data'] for i,path in paths.items()}
    accepted=torch.load(a.run/'checkpoint_dagger.pt',map_location='cpu',weights_only=False)
    keys=('history','condition','action');fresh=rows[28][0]
    aggregate={k:torch.cat([rows[i][0][k] for i in sorted(rows) if i<28]) for k in keys}
    other={k:torch.cat([rows[28][j][k] for j in range(1,19)]) for k in keys}
    results=[]
    for label,train in [('fixed_round28',fresh),('aggregate_before28',aggregate)]:
        actor=ContactActor().to(dev);actor.load_state_dict(accepted['actor'])
        opt=torch.optim.Adam(actor.parameters(),lr=a.lr)
        out=a.output/label;out.mkdir()
        @torch.no_grad()
        def evaluate(data):
            errors=[]
            for start in range(0,len(data['action']),1024):
                sl=slice(start,start+1024);errors.append((actor(data['history'][sl],data['condition'][sl])-data['action'][sl]).square())
            return torch.cat(errors)
        started=time.perf_counter();metrics=[];best=float('inf');stale=0
        for step in range(a.steps+1):
            if step%50==0:
                loss=float(evaluate(train).mean());fresh_error=evaluate(fresh)
                row=dict(step=step,train_mse=loss,round28_mse=float(fresh_error.mean()),round29_mse=float(evaluate(rows[29][0]).mean()),other18_round28_mse=float(evaluate(other).mean()),elapsed_s=time.perf_counter()-started)
                metrics.append(row);print(json.dumps(dict(branch=label,**row)),flush=True)
                if loss<best:
                    improvement=best-loss;best=loss
                    saved=dict(actor={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},config=accepted['config'],policy_updates=0,
                               diagnostic=dict(branch=label,fit_steps=step,learning_rate=a.lr,train_mse=loss,parent=str(a.run/'checkpoint_dagger.pt')))
                    torch.save(saved,out/'checkpoint.pt')
                    stale=stale+1 if improvement<max(1e-5,best*.005) else 0
                else:stale+=1
                (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
                if step>=200 and (loss<1e-4 or stale>=5):break
            if step==a.steps:break
            ids=torch.randint(len(train['action']),(min(512,len(train['action'])),),device=dev)
            loss=10*(actor(train['history'][ids],train['condition'][ids])-train['action'][ids]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);opt.step()
        actor.load_state_dict(torch.load(out/'checkpoint.pt',map_location=dev,weights_only=False)['actor'])
        error=evaluate(fresh);joint=error.mean(0)
        result=dict(branch=label,training_samples=len(train['action']),training_rounds=[28] if label=='fixed_round28' else sorted(i for i in rows if i<28),
            selected_checkpoint=str(out/'checkpoint.pt'),best_train_mse=best,metrics=metrics,
            round28_per_frame_mse=error.mean(-1).tolist(),round28_joint_mse={k:float(v) for k,v in zip(G1_JOINT_NAMES,joint)})
        results.append(result)
    (a.output/'report.json').write_text(json.dumps(dict(config=dict(run=str(a.run),lr=a.lr,max_steps=a.steps,loss='10 * action MSE, Adam, grad clip 1',normalization='unchanged'),results=results),indent=2))

if __name__=='__main__':main()
