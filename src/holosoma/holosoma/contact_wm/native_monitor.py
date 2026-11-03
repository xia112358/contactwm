"""Frozen-policy native evaluation episodes and browser trajectory exports."""
import json
from pathlib import Path
import numpy as np
import torch
from holosoma.contact_wm.geometry import LINKS
from holosoma.contact_wm.analyze import material_point_velocity
from holosoma.contact_wm.validation import native_success
from somaforge_core import G1_29DOF_JOINT_ORDER
from somaforge_core.batched_g1_kinematics import BatchedG1ForwardKinematics

def touchdown_summary(x,targets):
    """First observed new-hand contact, distinct from later first goal entry."""
    results=[];points=[];errors=[];goal_frames=[]
    for i,target in enumerate(targets):
        observed=x['active'][5:,i,2]&x['known'][5:,i,2]
        hits=np.flatnonzero(observed)
        index=int(hits[0])+5 if len(hits) else None
        point=x['point'][index,i,2] if index is not None else None
        error=float(np.linalg.norm(point-target)) if point is not None else None
        eligible=observed&(np.linalg.norm(x['point'][5:,i,2]-target,axis=-1)<=.05)
        goals=np.flatnonzero(eligible);goal_frames.append(int(goals[0])+5 if len(goals) else None)
        points.append(point.tolist() if point is not None else None);errors.append(error)
        results.append(dict(index=i,frame=index,error_m=error,success=error is not None and error<=.05))
    response=None
    if all(p is not None for p in points):
        from holosoma.contact_wm.target_response import response_metrics
        response=response_metrics(torch.tensor(points,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32))
    observed_errors=[e for e in errors if e is not None]
    return dict(results=results,successes=sum(r['success'] for r in results),
        strict_2cm_successes=sum(e is not None and e<=.02 for e in errors),
        mean_contact_error_m=float(np.mean(observed_errors)) if observed_errors else None,
        per_target_mean_position_error_m=errors,target_response=response,first_points=points,
        first_goal_frames=goal_frames,goal_reached=sum(f is not None for f in goal_frames),
        scope='first observed new-hand contact after warmup; no post-contact hold/load/slip criterion')

class NativeMonitor:
    def __init__(self,out,joint_names,mesh,atlas):
        self.out=Path(out);self.order=[joint_names.index(n) for n in G1_29DOF_JOINT_ORDER]
        self.links=json.loads(Path(mesh).read_text())['links'];self.fk=BatchedG1ForwardKinematics()
        self.frames=[];self.version=0
        a=np.load(atlas);self.terrain=dict(height=[[float(v) if np.isfinite(v) else None for v in row] for row in a['height'][0,::3,::3]],minimum=a['minimum'][0].tolist(),resolution=float(a['resolution'])*3)
    def begin(self,version,goal):
        self.frames=[];self.version=version
        self.goal={k:v[:9].detach().cpu().numpy().copy() if isinstance(v,torch.Tensor) else v for k,v in vars(goal).items()}
    def add(self,state,action,points,active,known,force,failed):
        self.frames.append({k:v[:9].detach().cpu().numpy().copy() for k,v in dict(state=state,action=action,
            point=points,active=active,known=known,force=force,failed=failed).items()})
    def finish(self):
        if len(self.frames)!=40:raise ValueError('Native evaluation requires full 40-step interval')
        x={k:np.stack([r[k] for r in self.frames]) for k in self.frames[0]}
        x['failed']=np.maximum.accumulate(x['failed'],axis=0)
        states=torch.tensor(x['state']);qpos=torch.cat((states[...,:3],states[...,[6,3,4,5]],states[...,13:42][...,self.order]),-1)
        pos,rot=self.fk.link_poses(qpos,LINKS)
        velocity=material_point_velocity(torch.tensor(x['point']),pos,rot,.02).numpy()
        body,rotation=self.fk.link_poses(qpos,self.links)
        results=[];episodes=[]
        for i in range(9):
            goal={k:v[i] if isinstance(v,np.ndarray) else v for k,v in self.goal.items()}
            # Supports are tested by activation/load/slip; only new contact needs fixed placement.
            result=native_success(x['active'][5:,i],x['known'][5:,i],x['force'][5:,i],x['point'][5:,i],
                velocity[5:,i],x['failed'][5:,i],goal,arrival=goal['arrival_step']-1,hold=goal['hold_steps'],
                position_required=goal['establish'])
            hold_start=5+goal['arrival_step']-1
            observed=x['active'][hold_start:,i,2]&x['known'][hold_start:,i,2]
            distance=np.linalg.norm(x['point'][hold_start:,i,2]-goal['position_w'][2],axis=-1)
            result['observed_hold_steps']=int(observed.sum())
            result['max_position_error']=float(distance[observed].max()) if observed.any() else None
            result['index']=i;results.append(result)
            episodes.append(dict(index=i,result=result,target=goal['position_w'][2].tolist(),
                frames=[dict(root=x['state'][t,i,:3].tolist(),bodyPosition=body[t,i].tolist(),bodyRotation=rotation[t,i].tolist(),
                    point=x['point'][t,i].tolist(),active=x['active'][t,i].tolist(),force=x['force'][t,i].tolist(),
                    failed=bool(x['failed'][:t+1,i].any())) for t in range(40)]))
        summary=dict(initial_state_max_mismatch=float(np.abs(x['state'][4]-x['state'][4,4]).max()),
            matched_initial_state=bool(np.abs(x['state'][4]-x['state'][4,4]).max()<=1e-4),iteration=self.version,successes=sum(r['success'] for r in results),total=9,results=results,
            source='native Newton/MJWarp; frozen policy, no exploration noise; excluded from training replay',
            contact_sampling='last native solve, poses post integration',
            mean_contact_error_m=float(np.linalg.norm(x['point'][-10:,:,2]-self.goal['position_w'][None,:,2],axis=-1)[x['active'][-10:,:,2]&x['known'][-10:,:,2]].mean()) if (x['active'][-10:,:,2]&x['known'][-10:,:,2]).any() else None)
        mask=x['active'][-10:,:,2]&x['known'][-10:,:,2]
        counts=mask.sum(0)
        endpoints=np.where(mask[...,None],x['point'][-10:,:,2],0.).sum(0)/counts[:,None].clip(min=1)
        summary['per_target_observed_hold_steps']=counts.tolist()
        summary['per_target_mean_position_error_m']=[float(np.linalg.norm(x['point'][-10:,i,2]-self.goal['position_w'][i,2],axis=-1)[mask[:,i]].mean()) if counts[i] else None for i in range(9)]
        summary['strict_2cm_successes']=sum(r['success'] and r['max_position_error'] is not None and r['max_position_error']<=.02 for r in results)
        if (counts==10).all():
            from holosoma.contact_wm.target_response import response_metrics
            summary['target_response']=response_metrics(torch.tensor(endpoints,dtype=torch.float32),torch.tensor(self.goal['position_w'][:,2],dtype=torch.float32))
        else:summary['target_response']=None
        summary['touchdown']=touchdown_summary(x,self.goal['position_w'][:,2])
        payload=dict(**summary,monitor_version='solver_points_v2',dt=.02,episodes=episodes,terrain=self.terrain)
        temp=self.out/'native-latest.tmp';temp.write_text(json.dumps(payload,allow_nan=False));temp.replace(self.out/'native-latest.json')
        if self.version==0 and not (self.out/'native-baseline.json').exists():
            (self.out/'native-baseline.json').write_text(json.dumps(payload,allow_nan=False))
        with (self.out/'native-metrics.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        np.savez_compressed(self.out/'native-latest.npz',**x)
        return summary
