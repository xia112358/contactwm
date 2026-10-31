"""Isolated canonical-foot contact experiment. No policy, training, or dataset output."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import mujoco
import warp as wp
import newton
import mujoco_warp as mjw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from newton.geometry import HydroelasticSDF

p=argparse.ArgumentParser();p.add_argument('--collision_model',type=Path,required=True,help='Canonical single-world solver-collision.mjb with its .mjb.json provenance');p.add_argument('--mode',choices=['mjwarp','hard','hydro'],default='hydro');p.add_argument('--voxel',type=float,default=.004);p.add_argument('--kh',type=float,default=1e9);p.add_argument('--steps',type=int,default=600);p.add_argument('--dt',type=float,default=.002);p.add_argument('--sample_every',type=int,default=5);p.add_argument('--dynamic_only',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
from supermimic.paths import workspace_root
root=workspace_root();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
if a.mode=='mjwarp':
 from supermimic.contact_manifold_normals import install
 install()
wp.init();wp.set_device('cuda:0')
source=a.collision_model.resolve()
meta=json.loads(source.with_suffix('.mjb.json').read_text())
from somaforge_core.robot_assets import canonical_g1_asset_metadata
asset=canonical_g1_asset_metadata(collision_profile='surface_v2')
assert meta['robot_asset']['urdf_sha256']==asset['urdf_sha256'] and meta['robot_asset']['usd_bundle_sha256']==asset['usd_bundle_sha256']
m=mujoco.MjModel.from_binary_path(str(source));gid=next(i for i in range(m.ngeom) if '/left_ankle_roll/' in m.geom(i).name)
mid=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);v=np.array(m.mesh_vert[start:start+count],dtype=np.float64)
R=Rotation.from_quat(m.geom_quat[gid][[1,2,3,0]]).as_matrix();v=v@R.T+m.geom_pos[gid]
v[:,:2]-=(v[:,:2].min(0)+v[:,:2].max(0))/2;v[:,2]-=v[:,2].min()
hull=ConvexHull(v);faces=hull.simplices.copy();center=v.mean(0)
for f in faces:
 if np.dot(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]),v[f].mean(0)-center)<0:f[1],f[2]=f[2],f[1]
load=float(m.body_mass.sum()/2)
mesh=newton.Mesh(v.astype(np.float32),faces.reshape(-1).astype(np.int32))
b=newton.ModelBuilder();b.default_shape_cfg.gap=.0
cfg=b.ShapeConfig(density=0.,mu=.7,margin=0.,gap=0.,ke=1e7,kd=1e4,kh=a.kh,is_hydroelastic=a.mode=='hydro',sdf_target_voxel_size=a.voxel if a.mode=='hydro' else None,sdf_narrow_band_range=(-.02,.02))
b.add_shape_box(body=-1,xform=wp.transform((0.,0.,-.05),wp.quat_identity()),hx=.3,hy=.3,hz=.05,cfg=cfg,label='ground_patch')
ext=v.max(0)-v.min(0);inertia=np.diag([load*(ext[1]**2+ext[2]**2)/12,load*(ext[0]**2+ext[2]**2)/12,load*(ext[0]**2+ext[1]**2)/12]).astype(np.float32)
body=b.add_body(xform=wp.transform((0.,0.,.05),wp.quat_identity()),mass=load,inertia=wp.mat33(inertia),com=wp.vec3(*v.mean(0).tolist()),lock_inertia=True,label='canonical_left_foot_loaded')
if a.mode=='hydro':mesh.build_sdf(target_voxel_size=a.voxel,narrow_band_range=(-.02,.02),margin=.005,cache_dir=str(a.output/'sdf-cache'))
from dataclasses import replace
mesh_cfg=replace(cfg,sdf_target_voxel_size=None,sdf_narrow_band_range=(-.1,.1))
b.add_shape_mesh(body,mesh=mesh,cfg=mesh_cfg,label='canonical_left_ankle_collision_hull')
t0=time.monotonic();model=b.finalize();hydro_cfg=HydroelasticSDF.Config(reduce_contacts=True,output_contact_surface=True)
pipeline=newton.CollisionPipeline(model,broad_phase='explicit',rigid_contact_max=4096,sdf_hydroelastic_config=hydro_cfg)
solver=newton.solvers.SolverMuJoCo(model,use_mujoco_contacts=a.mode=='mjwarp',solver='newton',integrator='implicitfast',cone='pyramidal',njmax=4096,nconmax=4096,iterations=100,ls_iterations=50,disable_sensors=True,enable_multiccd=True)
s0=model.state();s1=model.state();control=model.control();contacts=pipeline.contacts();dt=a.dt
report=dict(schema='canonical_foot_threeway_probe_v2',training=False,mode=a.mode,voxel_m=a.voxel,kh=a.kh,asset=asset,source_collision_model=str(source),source_collision_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),geometry='same canonical foot convex hull; local 0.6x0.6x0.1m floor patch, top z=0; no whole-robot controller',load_kg=load,foot_extents_m=ext.tolist(),status='running',build_seconds=time.monotonic()-t0,metrics=[],scans={})
report['hydroelastic_active']=pipeline.hydroelastic_sdf is not None
if a.mode=='hydro' and not report['hydroelastic_active']:raise RuntimeError('Hydroelastic pair was not activated')
report['collision_interval_s']=dt
report['solver_settings']={'geom_solref':solver.mj_model.geom_solref.tolist(),'geom_solimp':solver.mj_model.geom_solimp.tolist(),'disableflags':int(solver.mj_model.opt.disableflags),'solver':int(solver.mj_model.opt.solver),'iterations':int(solver.mj_model.opt.iterations)}
qbase=model.joint_q.numpy().copy();qd=np.zeros_like(model.joint_qd.numpy())
def reset(z,x=0.,pitch=0.):
 q=qbase.copy();q[:3]=[x,0,z];q[3:7]=Rotation.from_euler('y',pitch).as_quat()
 s0.joint_q.assign(q);s0.joint_qd.assign(qd)
 newton.eval_fk(model,s0.joint_q,s0.joint_qd,s0)
 solver.reset(s0,flags=0)
 np.testing.assert_allclose(s0.joint_q.numpy(),q,atol=1e-7)
def capture(label,t):
 d=solver.mjw_data;n=int(d.nacon.numpy()[0]);kind=d.contact.type.numpy()[:n];dist=d.contact.dist.numpy()[:n];margin=d.contact.includemargin.numpy()[:n];address=d.contact.efc_address.numpy()[:n];active=((kind&1)!=0)&(dist<margin)
 pos=d.contact.pos.numpy()[:n];norm=d.contact.frame.numpy()[:n,0];geoms=d.contact.geom.numpy()[:n];ground_first=solver.mj_model.geom_bodyid[geoms[:,0]]==0;sign=np.where(ground_first,1.,-1.);norm=norm*sign[:,None];indices=np.flatnonzero(active)
 if len(indices) and (address[indices,:4]<0).any():raise RuntimeError('Active constraint lacks allocation')
 force=np.zeros((n,6),dtype=np.float32)
 if n:
  wrench=wp.zeros(n,dtype=wp.spatial_vector);mjw.contact_force(solver.mjw_model,d,wp.array(np.arange(n,dtype=np.int32)),True,wrench);force=wrench.numpy()
 # contact_force returns world-frame (force, torque) in MJWarp.
 fz=force[:,2]*sign;selected=np.maximum(fz,0)*active;total=float((fz*active).sum());positive_total=float(selected.sum());cop=(pos*selected[:,None]).sum(0)/positive_total if positive_total>1e-6 else None
 np.testing.assert_allclose(s0.body_q.numpy()[body],s0.joint_q.numpy()[:7],atol=1e-6)
 pose=s0.body_q.numpy()[body];worldv=Rotation.from_quat(pose[3:7]).apply(v)+pose[:3];penetration=max(0.,float(-worldv[:,2].min()))
 item=dict(label=label,time=t,geometric_penetration_m=penetration,geometric_gap_m=float(worldv[:,2].min()),active=int(active.sum()),raw_contacts=n,solver_min_dist_m=float(dist[active].min()) if active.any() else None,force_abs_z_n=total,cop_world=cop.tolist() if cop is not None else None,point_centroid=pos[active].mean(0).tolist() if active.any() else None,downward_normals=int((norm[active,2]<-.5).sum()),pose=pose.tolist(),qacc_max=float(np.abs(d.qacc.numpy()).max()),finite=bool(np.isfinite(d.qacc.numpy()).all() and np.isfinite(s0.body_q.numpy()).all()),points=pos[active].tolist(),fz=selected[active].tolist())
 if pipeline.hydroelastic_sdf is not None:
  surface=pipeline.hydroelastic_sdf.get_contact_surface();nf=int(surface.face_contact_count.numpy()[0])
  if nf>surface.max_num_face_contacts:raise RuntimeError('Hydroelastic face buffer overflow')
  tri=surface.contact_surface_point.numpy()[:nf*3].reshape(-1,3,3)
  item['surface_faces']=nf
  item['surface_area_m2']=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2) if nf else 0.
  if (label=='translate' and abs(t)<1e-9) or (label=='depth' and abs(t-.002)<1e-9):
   np.savez(a.output/(label+'_patch.npz'),triangles=tri,points=pos[active],force_z=selected[active])
 if not item['finite']:raise RuntimeError('Nonfinite native state')
 return item

def save():
 temp=a.output/'report.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(a.output/'report.json')
try:
 reset(.05);save()
 for i in range(a.steps):
  s0.clear_forces();pipeline.collide(s0,contacts) if a.mode!='mjwarp' else None;solver.step(s0,s1,control,contacts if a.mode!='mjwarp' else None,dt)
  # Capture against pre-integration pose: these are the actual solved contacts.
  if i%a.sample_every==0:report['metrics'].append(capture('drop_and_settle',i*dt))
  s0,s1=s1,s0
  if i%100==0:save();print('DYNAMIC',i,{k:report['metrics'][-1][k] for k in ('geometric_penetration_m','active','force_abs_z_n','downward_normals','finite')},flush=True)
 # A small horizontal velocity perturbation after settling; no pose change.
 q=s0.joint_q.numpy().copy();velocity=s0.joint_qd.numpy().copy();velocity[0]=.02
 s0.joint_q.assign(q);s0.joint_qd.assign(velocity)
 newton.eval_fk(model,s0.joint_q,s0.joint_qd,s0)
 solver.reset(s0,flags=0)
 np.testing.assert_allclose(s0.joint_qd.numpy(),velocity,atol=1e-7)
 for i in range(round(.2/dt)):
  s0.clear_forces();pipeline.collide(s0,contacts) if a.mode!='mjwarp' else None;solver.step(s0,s1,control,contacts if a.mode!='mjwarp' else None,dt)
  if i%a.sample_every==0:report['metrics'].append(capture('sliding',a.steps*dt+i*dt))
  s0,s1=s1,s0
 save();print('SLIDING completed',flush=True)
 for label,values in ([] if a.dynamic_only else [('translate',np.linspace(-.002,.002,41)),('pitch',np.linspace(-.01,.01,41)),('depth',np.array([0.,.00025,.0005,.001,.002]))]):
  rows=[]
  for value in values:
   z=-float(value) if label=='depth' else -.002
   reset(z,x=float(value) if label=='translate' else 0.,pitch=float(value) if label=='pitch' else 0.)
   pipeline.collide(s0,contacts) if a.mode!='mjwarp' else None;solver.step(s0,s1,control,contacts if a.mode!='mjwarp' else None,dt);item=capture(label,float(value));rows.append(item)
  report['scans'][label]=rows;save();print('SCAN',label,'completed',len(rows),flush=True)
 report['status']='passed';report['elapsed_seconds']=time.monotonic()-t0;save();print('DONE',a.output,flush=True)
except Exception:
 import traceback
 report['status']='failed';report['exception']=traceback.format_exc();save();raise
