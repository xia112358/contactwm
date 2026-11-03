"""Shared visual encoding preserves the parallel motion/force predictions."""
import sys
from pathlib import Path
import torch
from supermimic.lowstep import install
install('data/mixed_terrain_walk/task.json',contact_model='force6',policy_force_observation='none',perception='panorama_rolling',perception_buffer_steps=10)
from supermimic import networks
from supermimic.mdp import unflatten_state,normalize_xi
wm=networks.WorldModel().cuda()
ck=torch.load('logs/mixed_panorama_smoke/ckpt_000004.pt',map_location='cuda',weights_only=False)
wm.load_state_dict(ck['wm'])
s=torch.zeros(2,71,device='cuda');s[:,2]=1;s[:,3]=1;s[:,0]=torch.tensor([1.,2.],device='cuda')
state=unflatten_state(s);a=torch.zeros(2,29,device='cuda');xi=torch.zeros(2,58,device='cuda');xi[:,0]=1
count=[0]
hook=wm.depth_encoder.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
with torch.no_grad():
 separate_xi=wm.next_context(state,a,xi,normalize_xi(xi));separate_s=wm.step(state,a,normalize_xi(xi));assert count[0]==2
 count[0]=0;shared_s,shared_xi=wm.joint_step(state,a,xi,normalize_xi(xi));assert count[0]==1
 for first,second in zip(separate_s,shared_s):torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)
 torch.testing.assert_close(separate_xi,shared_xi,atol=1e-6,rtol=1e-6)
hook.remove();print('PASS shared input gives identical parallel motion/force outputs with one CNN call instead of two')
