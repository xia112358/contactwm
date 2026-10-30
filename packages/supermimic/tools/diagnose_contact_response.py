"""Isolated native contact-response comparison using a frozen tracking teacher."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,json,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];pass
p=argparse.ArgumentParser();p.add_argument('--response',choices=['baseline','firm_v1','firm_v2','multiccd','manifold_v1','manifold_v2','manifold_soft_v1','manifold_balanced_v1'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--motion',type=int,default=246);p.add_argument('--scoped-read',action='store_true');p.add_argument('--max-steps',type=int,default=1600)
a=p.parse_args();out=a.output
import torch,warp as wp,numpy as np
from supermimic.contact_native import NativeContacts
from newton import ModelFlags
original_init=NativeContacts.__init__;original_read=NativeContacts.read
reports=[]
normal_anomalies=[]
read_times=[]
read_comparisons=[]

def init(self,env,*args,**kwargs):
    original_init(self,env,*args,**kwargs)
    self.audit_world=int((env.task_ids==a.motion).nonzero().flatten()[0]);self.audit_step=0
    model=self.model;solver=self.solver
    fields=['geom_solref','geom_solimp','geom_margin','geom_gap']
    before={k:getattr(solver.mjw_model,k).numpy()[self.audit_world].tolist() for k in fields}
    material={k:np.unique(getattr(model,k).numpy(),axis=0).tolist() for k in ['shape_material_ke','shape_material_kd']}
    if a.response in ('firm_v1','firm_v2','manifold_v1','manifold_v2'):
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
        wp.to_torch(model.mujoco.solref_mode).fill_(SOLREF_MODE_RAW)
        wp.to_torch(model.mujoco.solref)[:]=torch.tensor([.004 if a.response in ('firm_v2','manifold_v1','manifold_v2') else .01,1.],device=env.device)
        wp.to_torch(model.mujoco.geom_solimp)[:]=torch.tensor([.99,.999,.001,.5,2.],device=env.device)
        env.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        if a.response in ('firm_v2','manifold_v1','manifold_v2'):
            from supermimic.contact_response import apply
            apply(env,a.response)
    if a.response in ('manifold_soft_v1','manifold_balanced_v1'):
        from supermimic.contact_response import apply
        apply(env,a.response)
    if a.response in ('firm_v2','manifold_v1','manifold_v2'):
        # The teacher still runs the original 296-world scene. Audit only the
        # requested motion; record (do not conceal) background-world anomalies.
        import somaforge_core.newton_support as ns
        original_support=ns.support_motion
        def support(snapshot, **kw):
            count=int(snapshot['count'])
            normals=np.asarray(snapshot['frame_w'])[:count,0]
            err=np.abs(np.linalg.norm(normals,axis=-1)-1)
            bad=kw['eligible'] & (err>1.1e-5)
            if bad.any():
                normal_anomalies.append(dict(step=self.audit_step,worlds=np.asarray(snapshot['worldid'])[:count][bad].tolist(),norms=np.linalg.norm(normals[bad],axis=-1).tolist()))
                (out/'normal-anomalies.json').write_text(json.dumps(normal_anomalies,indent=2))
            if a.response!='manifold_v2':
                kw['eligible']=kw['eligible'] & (np.asarray(snapshot['worldid'])[:count]==self.audit_world)
            return original_support(snapshot,**kw)
        ns.support_motion=support
    after={k:getattr(solver.mjw_model,k).numpy()[self.audit_world].tolist() for k in fields}
    config=dict(response=a.response,motion=a.motion,world=self.audit_world,solver_dt=env.sim.backend_manager.get_solver_dt(),enableflags=int(solver.mj_model.opt.enableflags),before=before,after=after,materials=material,
        body_labels=list(model.body_label),note='Diagnostic physics transfer, frozen original teacher; not a compatible production checkpoint or training dataset.')
    (out/'contact-response.json').write_text(json.dumps(config,indent=2));print('RESPONSE',a.response,'dt',config['solver_dt'],'solref',before['geom_solref'][21],'solimp',before['geom_solimp'][21], 'material',material,flush=True)

def read(self):
    start=time.perf_counter()
    result=original_read(self)
    read_times.append(time.perf_counter()-start)
    if a.scoped_read and self.audit_step in (0,50,100):
        scoped=result
        self.set_observed_worlds(torch.arange(self.env.N))
        start=time.perf_counter();full=original_read(self);full_time=time.perf_counter()-start
        errors={}
        for k in scoped:
            x,y=scoped[k][self.audit_world],full[k][self.audit_world]
            torch.testing.assert_close(x,y,atol=1e-6,rtol=0,equal_nan=True)
            errors[k]=float(torch.nan_to_num(x.float()-y.float()).abs().max())
        assert not bool(scoped['known'][torch.arange(self.env.N,device=self.env.device)!=self.audit_world].any())
        self.set_observed_worlds(torch.tensor([self.audit_world]));result=original_read(self)
        read_comparisons.append(dict(step=self.audit_step,scoped_seconds=read_times[-1],full_seconds=full_time,errors=errors))
        (out/'scope-equivalence.json').write_text(json.dumps(read_comparisons,indent=2))
    r=self.last_snapshot;world=self.audit_world
    if a.response in ('manifold_v2','manifold_soft_v1','manifold_balanced_v1'):
        n=int(r['count']);active=r['active'][:n]
        norms=np.linalg.norm(np.asarray(r['frame_w'])[:n,0],axis=-1)
        if not np.allclose(norms[active],1.,atol=1e-6):
            raise ValueError('Invalid active native constraint normal in full raw snapshot')
    count=int(r['count']);mask=(r['worldid'][:count]==world)&r['active'][:count]
    indices=np.flatnonzero(mask);items=[]
    # All native robot-terrain contacts, including parts outside six-part task labels.
    for i in indices.tolist():
        b0=int(r['body0'][i]);b1=int(r['body1'][i]);labels=self.model.body_label
        n0=labels[b0] if b0>=0 else 'world';n1=labels[b1] if b1>=0 else 'world'
        if ('/Robot/' in n0)==('/Robot/' in n1):continue
        items.append(dict(body=(n0 if '/Robot/' in n0 else n1).rsplit('/',1)[-1],other=n1 if '/Robot/' in n0 else n0,
            dist=float(r['dist'][i]),margin=float(r['includemargin'][i]),allocated=bool(r['constraint_allocated'][i]),force=r['force_on_body1_w'][i].tolist()))
    reports.append(dict(step=self.audit_step,contacts=items));self.audit_step+=1
    if self.audit_step%100==0:(out/'all-body-contacts.json').write_text(json.dumps(reports))
    return result
NativeContacts.__init__=init;NativeContacts.read=read
if a.response in ('firm_v2','multiccd','manifold_v1','manifold_v2','manifold_soft_v1','manifold_balanced_v1'):
    import supermimic.env as env_module
    original_make_sim=env_module.make_sim
    def make_sim(device,args):
        if a.response in ('manifold_v2','manifold_soft_v1','manifold_balanced_v1'):
            from supermimic.contact_manifold_normals import install
            install()
        cfg=original_make_sim(device,args)
        if a.response in ('firm_v2','manifold_v1','manifold_v2'):cfg.physics.num_substeps=10
        if a.response in ('multiccd','manifold_v1','manifold_v2','manifold_soft_v1','manifold_balanced_v1'):cfg.physics.solver_cfg.enable_multiccd=True
        return cfg
    env_module.make_sim=make_sim
sys.argv=['contact_collect','--task_manifest',str(runtime_path('data/php296_surface_v2/task.json')),'--checkpoint',str(runtime_path('logs/php296_surface_v2_startprobe_20261029_v1/tracking/ckpt_010000.pt')),
    '--output',str(out),'--num_envs','296','--motion',str(a.motion),'--steps',str(a.max_steps),'--seed','0','--headless']
try:runpy.run_module('supermimic.contact_collect',run_name='__main__')
finally:
    if out.exists():
        (out/'all-body-contacts.json').write_text(json.dumps(reports))
        (out/'read-times.json').write_text(json.dumps(read_times))
if (out/'rollout.pt').exists():
    saved=torch.load(out/'rollout.pt',map_location='cpu',weights_only=False);saved['metadata']['contact_response_diagnostic']=a.response;saved['metadata']['production_compatible']=False;torch.save(saved,out/'rollout.pt')
    contacts=[c for r in reports for c in r['contacts']];feet=[c for c in contacts if 'ankle_roll_link' in c['body']];other=[c for c in contacts if c['body'] not in ['left_ankle_roll_link','right_ankle_roll_link','left_sphere_hand_link','right_sphere_hand_link','left_knee_link','right_knee_link']]
    def stats(rows):
        v=np.array([-min(0,c['dist']) for c in rows]);return dict(count=len(v),mean_mm=float(v.mean()*1000),p95_mm=float(np.quantile(v,.95)*1000),max_mm=float(v.max()*1000)) if len(v) else dict(count=0)
    result=dict(response=a.response,frames=len(saved['data']['s']),terminated=bool(saved['data']['term'].any()),foot_penetration=stats(feet),all_body_penetration=stats(contacts),unallocated=sum(not c['allocated'] for c in contacts),other_contact_bodies=sorted(set(c['body'] for c in other)))
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2));print('DIAGNOSIS',json.dumps(result),flush=True)
