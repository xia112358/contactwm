"""Deterministic regression for the cylinder/large-box distant-witness failure.

Run with PYTHONPATH pointing to the project-local patched Newton directory.
Tests physical overlap, separation, and near-edge cases through the fused writer.
"""
import argparse
from pathlib import Path
import numpy as np,warp as wp,json,os
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
from scipy.spatial.transform import Rotation
from newton._src.geometry.collision_convex import ConvexQueryResult,create_write_convex_query_result,create_solve_convex_multi_contact
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.support_function import GenericShapeData,SupportMapDataProvider,support_map
from newton._src.geometry.types import GeoType
@wp.struct
class Writer:
 pos:wp.array(dtype=wp.vec3)
 dist:wp.array(dtype=float)
 normal:wp.array(dtype=wp.vec3)
 counts:wp.array(dtype=int)
@wp.func
def writer(c:ContactData,w:Writer,index:int):
 if c.contact_distance < 0.0:
  slot=c.shape_a;k=wp.atomic_add(w.counts,slot,1)
  if k<8:
   w.pos[slot*8+k]=c.contact_point_center;w.dist[slot*8+k]=c.contact_distance;w.normal[slot*8+k]=c.contact_normal_a_to_b
@wp.func
def post(c:ContactData,a:GenericShapeData,pa:wp.vec3,qa:wp.quat,b:GenericShapeData,pb:wp.vec3,qb:wp.quat)->ContactData:return c
write=create_write_convex_query_result(support_map,writer,post)
solve=create_solve_convex_multi_contact(support_map,writer,post)
@wp.kernel
def run(pos:wp.array(dtype=wp.vec3),quat:wp.array(dtype=wp.quat),size:wp.array(dtype=wp.vec3),out:Writer):
 i=wp.tid();a=GenericShapeData();a.shape_type=int(GeoType.CYLINDER);a.scale=wp.vec3(.03,.025,0.)
 b=GenericShapeData();b.shape_type=int(GeoType.BOX);b.scale=size[i]
 qa=quat[i];qb=wp.quat_identity();pa=pos[i];pb=wp.vec3(0.,0.,-size[i][2]);relq=wp.quat_inverse(qa)*qb;relp=wp.quat_rotate_inv(qa,pb-pa);provider=SupportMapDataProvider()
 template=ContactData();template.shape_a=i
 solve(a,b,qa,qb,pa,pb,provider,0.,False,out,template)

wp.init();rng=np.random.default_rng(20261101);N=2400;box_half=np.repeat([.2,1.,100.],N//3);q=rng.normal(size=(N,4));q/=np.linalg.norm(q,axis=1,keepdims=True);axis=Rotation.from_quat(q).apply(np.tile([0.,0.,1.],(N,1)));vertical=.025*np.abs(axis[:,2])+.03*np.sqrt(np.maximum(0.,1.-axis[:,2]**2))
kind=np.arange(N)%4;depth=np.choose(kind,[.001,.000001,-.000006,.001]);x=np.zeros(N);x[kind==3]=box_half[kind==3]-.005
pos=np.column_stack((x,np.zeros(N),vertical-depth)).astype(np.float32);sizes=np.column_stack((box_half,box_half,np.full(N,.05))).astype(np.float32)
w=Writer();w.pos=wp.zeros(N*8,dtype=wp.vec3,device='cpu');w.dist=wp.zeros(N*8,dtype=float,device='cpu');w.normal=wp.zeros(N*8,dtype=wp.vec3,device='cpu');w.counts=wp.zeros(N,dtype=int,device='cpu');wp.launch(run,N,inputs=[wp.array(pos,dtype=wp.vec3,device='cpu'),wp.array(q.astype(np.float32),dtype=wp.quat,device='cpu'),wp.array(sizes,dtype=wp.vec3,device='cpu'),w],device='cpu')
points=w.pos.numpy().reshape(N,8,3);counts=w.counts.numpy();mask=np.arange(8)[None,:]<counts[:,None];radius=np.linalg.norm(points-pos[:,None,:],axis=-1);far=(radius>2.)&mask;summary={}
for k,name in enumerate(['overlap_1mm','overlap_1um','separated_6um','near_edge_1mm']):
 m=kind==k;summary[name]={'cases':int(m.sum()),'contact_cases':int((m&(counts>0)).sum()),'contacts':int(mask[m].sum()),'far_contacts':int(far[m].sum()),'max_contact_body_distance_m':float(radius[m][mask[m]].max()) if mask[m].any() else None}
bad=np.argwhere(far)
np.savez_compressed(args.output/'geometry.npz',positions=pos,quaternions=q,half_sizes=sizes,points=points,counts=counts,distances=w.dist.numpy().reshape(N,8),normals=w.normal.numpy().reshape(N,8,3),bad=bad)
print('first_far',[(int(i),int(j),float(radius[i,j])) for i,j in bad[:5]])
print(json.dumps(summary,indent=2))
(args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert not far.any(), 'contact witness is outside its cylinder'
assert summary['separated_6um']['contact_cases']==0, 'separated shapes have negative-distance contacts'
assert summary['overlap_1mm']['contact_cases']==600, 'face-interior overlap lost contacts'
