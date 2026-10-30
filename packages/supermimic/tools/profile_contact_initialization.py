"""Profile solver initialization only, then run the requested frozen collection."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import cProfile,json,runpy,sys,time,pstats
from pathlib import Path
pass
prefix=Path(sys.argv[1]);sys.argv=['supermimic.contact_train']+sys.argv[2:]
from isaaclab_newton.physics.newton_manager import NewtonManager
original=NewtonManager.initialize_solver

def profiled(self):
    import os
    if os.environ.get('SUPERMIMIC_VALIDATE_FAST_INIT')=='1':
        import mujoco,mujoco_warp,copy,dataclasses,numpy as np,warp as wp
        fast=mujoco_warp.put_data
        def checked(model,data,**kwargs):
            mujoco.mj_saveModel(model,str(prefix)+'.mjb',None)
            small=dict(kwargs,nworld=4)
            baseline=fast.original(model,copy.copy(data),**small)
            candidate=fast(model,copy.copy(data),**small)
            skipped={'map_dof2idof','map_idof2dof','contact.geomcollisionid','efc.jtdaj_adr','efc.jtdaj_nrow','efc.jtdaj_nblock'}
            if model.opt.density==0 and model.opt.viscosity==0:skipped.add('efm_fluid')
            verified=[]
            def compare(a,b,prefix=''):
                for f in dataclasses.fields(a):
                    key=prefix+f.name;x=getattr(a,f.name);y=getattr(b,f.name)
                    if dataclasses.is_dataclass(x):compare(x,y,key+'.')
                    elif isinstance(x,wp.array):
                        assert x.shape==y.shape and x.dtype==y.dtype,key
                        if key not in skipped:
                            np.testing.assert_array_equal(x.numpy(),y.numpy(),err_msg=key);verified.append(key)
                    else:assert x==y,key
            compare(baseline,candidate)
            Path(str(prefix)+'.array-check.json').write_text(json.dumps(dict(verified=verified,skipped_inactive_or_uninitialized=sorted(skipped),nq=model.nq,nv=model.nv,nbody=model.nbody,ngeom=model.ngeom),indent=2))
            print('PASS actual robot initial arrays:',len(verified),flush=True)
            return fast(model,data,**kwargs)
        mujoco_warp.put_data=checked
    profile=cProfile.Profile();start=time.perf_counter();profile.enable()
    try:return original(self)
    finally:
        profile.disable();elapsed=time.perf_counter()-start
        profile.dump_stats(str(prefix)+'.pstats')
        stats=pstats.Stats(profile)
        rows=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
        rows.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
        Path(str(prefix)+'.json').write_text(json.dumps(dict(elapsed_s=elapsed,functions=rows[:120]),indent=2))
        print('INITIALIZATION PROFILE',json.dumps(dict(elapsed_s=elapsed,top=rows[:18])),flush=True)
NewtonManager.initialize_solver=profiled
runpy.run_module('supermimic.contact_train',run_name='__main__')
