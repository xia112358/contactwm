"""Check collision-surface consistency, native FK and finite-difference gradients."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];pass
import torch,numpy as np
spec=json.loads((runtime_path('data/lowclimb_boxes/task.json')).read_text());pass
from supermimic.contact_surface import nearest_triangle,CollisionSurface
from supermimic.contact_control import ContactGeometry,ContactClock,LINKS
from supermimic.mdp import unflatten_state
from supermimic.robot import G1_JOINT_NAMES
from supermimic.rotations import mat_to_quat
tri=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]]])
p=torch.tensor([[.2,.3,1.],[-1.,0.,.2],[2.,2.,1.]])
r=nearest_triangle(p,tri);torch.testing.assert_close(r,torch.tensor([[.2,.3,0],[0.,0,0],[.5,.5,0]]))
base=runtime_path('tmp/contact-tracker-20261027-v1')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recording',type=Path,default=base/'demo0-roles-v1/rollout.pt')
parser.add_argument('--collision_model',type=Path,default=base/'initialization-fast-v3-solver.mjb')
parser.add_argument('--collision_profile',choices=('legacy_v1','surface_v2'),default='legacy_v1')
args=parser.parse_args()
demo=torch.load(args.recording,map_location='cpu',weights_only=False)['data']
geom=ContactGeometry(demo,'cpu');modelpath=args.collision_model
surface=CollisionSurface(modelpath,geom,collision_profile=args.collision_profile)
import mujoco
m=mujoco.MjModel.from_binary_path(str(modelpath));d=mujoco.MjData(m);state=unflatten_state(demo['s'][200:201]);d.qpos[:3]=state[0][0].numpy();d.qpos[3:7]=mat_to_quat(state[1])[0].numpy()
for i in range(m.njnt):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or ''
 for k,canonical in enumerate(G1_JOINT_NAMES):
  if name.endswith('_'+canonical) or name==canonical:d.qpos[m.jnt_qposadr[i]]=float(state[4][0,k])
mujoco.mj_forward(m,d);p,R=geom.poses(state)
for k,link in enumerate(LINKS):
 ids=[i for i in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('_'+link)];assert len(ids)==1
 np.testing.assert_allclose(p[0,k].numpy(),d.xpos[ids[0]],atol=3e-6)
 np.testing.assert_allclose(R[0,k].numpy(),d.xmat[ids[0]].reshape(3,3),atol=3e-6)
clock=ContactClock(demo,hold=1,role_aware=True);t=torch.tensor([clock.event_ends[0]-1]);S=unflatten_state(demo['s1'][t]);target=demo['point'][t].clone();target=target.double();target+=torch.tensor([.08,.07,.09],dtype=torch.float64);surface.double();S=tuple(v.double() for v in S)
root=S[0].clone().requires_grad_();parts=[1]
def L(pos):return (surface.contact_points((pos,)+S[1:],target,parts)[:,1]-target[:,1]).square().sum()
loss=L(root);grad=torch.autograd.grad(loss,root)[0]
for axis in range(3):
 delta=torch.zeros_like(root);delta[:,axis]=1e-5
 fd=(L(root.detach()+delta)-L(root.detach()-delta))/2e-5
 torch.testing.assert_close(grad[0,axis],fd,atol=1e-5,rtol=.02)
print('PASS triangle nearest points, actual collider-link FK versus MuJoCo, finite-difference root gradient')
print(surface.catalog)
# Contact sensor activation must no longer change the position-loss witness.
import runpy
from unittest.mock import patch
x=runpy.run_path(str(ROOT/'tools/check_contact_duration.py'))
class FixtureSurface:
    def contact_points(self,S,target,parts):return x['Geometry']()(S,None)
values=[]
for active in (False,True):
    dat={k:v.clone() for k,v in x['d'].items()};dat['active'].fill_(active)
    with patch('supermimic.contact_dmo.observation',lambda S,*args:S[0][:,:1]):
        actor=x['Actor']();loss,details=x['replay_loss'](actor,x['WM'](),x['clock'],x['Geometry'](),None,lambda v:v,None,dat,completion_mode='accumulated_target',stop_steps=torch.tensor([2]),contact_surface=FixtureSurface())
        loss.backward();values.append((details['terminal_contact'],actor.weight.grad.clone()))
assert values[0][0]==values[1][0]==16
assert torch.equal(values[0][1],values[1][1])
print('PASS sensor active/inactive at same state: identical surface loss and parameter gradients')
