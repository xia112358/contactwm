"""Check the FK/material surrogate derivative against finite differences."""
import sys,json
from pathlib import Path
import torch
from supermimic.lowstep import install
install('data/lowclimb_boxes/task.json',contact_model='force6')
from supermimic.contact_control import ContactGeometry,ContactClock
from supermimic.mdp import unflatten_state
torch.set_num_threads(4);torch.manual_seed(19)
p=Path('tmp/contact-tracker-20261027-v1/dmo-segment7-onlinewm-adaptive-v1');cfg=json.loads((p/'config.json').read_text());d=torch.load(cfg['recording'],map_location='cpu',weights_only=False)['data'];geo=ContactGeometry(d,'cpu').double();clock=ContactClock(d,hold=1,role_aware=True)
b=torch.load(p/'replay.pt',map_location='cpu',weights_only=False);rows=(b['ptr']-54+torch.arange(54))%b['T'];v={k:x[rows,:64] for k,x in b['data'].items()};S=unflatten_state(v['s1'][45].double());point=v['point'][45,:,0].double();target=d['point'][clock.event_ends[6]-1,0].double()
with torch.no_grad():pos,rot=geo.poses(S);local=(rot[:,0].transpose(-1,-2)@(point-pos[:,0]).unsqueeze(-1)).squeeze(-1)
p0=S[0].detach().requires_grad_();q0=S[4].detach().requires_grad_()
def objective(p,q):
 state=(p,*S[1:4],q,S[5]);pos,rot=geo.poses(state);material=pos[:,0]+(rot[:,0]@local.unsqueeze(-1)).squeeze(-1)
 return ((material-target)/.05).square().sum(-1).mean()
loss=objective(p0,q0);gp,gq=torch.autograd.grad(loss,(p0,q0));checks=[]
for kind in ['root_translation','joint_angles']:
 dp=torch.randn_like(p0) if kind=='root_translation' else torch.zeros_like(p0)
 dq=torch.randn_like(q0) if kind=='joint_angles' else torch.zeros_like(q0)
 eps=1e-6;analytic=float((gp*dp).sum()+(gq*dq).sum());finite=float((objective(p0+eps*dp,q0+eps*dq)-objective(p0-eps*dp,q0-eps*dq))/(2*eps));assert abs(analytic-finite)<1e-5*max(1,abs(analytic)),(analytic,finite)
 checks.append(dict(kind=kind,analytic=analytic,finite_difference=finite))
print(json.dumps(dict(result='PASS',scope='Fixed body-local material point FK derivative only; not the native-contact derivative or WM derivative.',checks=checks),indent=2))
