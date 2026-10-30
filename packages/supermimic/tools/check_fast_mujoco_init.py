"""Compare fresh arrays and contact-bearing trajectories against upstream init."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,dataclasses,json,time
from pathlib import Path
pass
import numpy as np
import mujoco,mujoco_warp,warp as wp
from supermimic.fast_mujoco_init import install
wp.init();install();fast=mujoco_warp.put_data;original=fast.original
xml='''<mujoco><option timestep="0.01"/><worldbody><geom type="plane" size="3 3 .1"/><body pos="0 0 .7"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>'''
results=[]
with wp.ScopedDevice('cuda:0'):
 for jac in [mujoco.mjtJacobian.mjJAC_DENSE,mujoco.mjtJacobian.mjJAC_SPARSE]:
  m=mujoco.MjModel.from_xml_string(xml);m.opt.jacobian=jac;d=mujoco.MjData(m)
  args=dict(nworld=8,nconmax=32,njmax=64)
  baseline=original(m,mujoco.MjData(m),**args);candidate=fast(m,mujoco.MjData(m),**args)
  checked=[]
  # Upstream wp.empty workspaces and island maps with nidof=0 have no defined initial value.
  scratch={'map_dof2idof','map_idof2dof','contact.geomcollisionid','efc.jtdaj_adr','efc.jtdaj_nrow','efc.jtdaj_nblock'}
  def compare(a,b,prefix=''):
   for f in dataclasses.fields(a):
    key=prefix+f.name;x=getattr(a,f.name);y=getattr(b,f.name)
    if dataclasses.is_dataclass(x):compare(x,y,key+'.')
    elif isinstance(x,wp.array):
     assert x.shape==y.shape and x.dtype==y.dtype,(key,x.shape,y.shape)
     if key not in scratch:
      np.testing.assert_array_equal(x.numpy(),y.numpy(),err_msg=key);checked.append(key)
    else:assert x==y,(key,x,y)
  compare(baseline,candidate)
  model=mujoco_warp.put_model(m)
  for _ in range(100):
   mujoco_warp.step(model,baseline);mujoco_warp.step(model,candidate)
  errors={}
  for key in ['qpos','qvel','qacc','qfrc_constraint','nefc','nacon']:
   x=getattr(baseline,key).numpy();y=getattr(candidate,key).numpy()
   errors[key]=float(np.max(np.abs(x-y))) if x.size else 0.
   np.testing.assert_allclose(x,y,rtol=2e-5,atol=1e-6,err_msg=key)
  assert int(baseline.nacon.numpy()[0])>0
  # Nonempty contact/constraint states must also preserve packed contact layout.
  mujoco.mj_forward(m,d);d.qpos[2]=.05;mujoco.mj_forward(m,d)
  assert d.ncon>0
  import copy
  reference=original(m,copy.copy(d),**args);fallback=fast(m,copy.copy(d),**args)
  compare(reference,fallback)
  for _ in range(25):
   mujoco_warp.step(model,reference);mujoco_warp.step(model,fallback)
  for key in ['qpos','qvel','qacc','qfrc_constraint','nefc','nacon']:
   np.testing.assert_allclose(getattr(reference,key).numpy(),getattr(fallback,key).numpy(),rtol=2e-5,atol=1e-5,err_msg='pre-contact '+key)
  results.append(dict(jacobian=int(jac),arrays_checked=len(checked),steps=100,max_abs_errors=errors,contact_count=int(baseline.nacon.numpy()[0])))
print('PASS',json.dumps(results))
