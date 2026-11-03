"""Exercise the new surface-region branch in actual DMO replay, on CPU."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
# Reuse the maintained episode/cutoff fixture and its legacy regression checks.
x=runpy.run_path(str(ROOT/'tools/check_contact_duration.py'))
from somaforge_core.contact_system.objectives import smooth_region_loss


class Surface:
    def region_loss(self,S,target,parts,**kw):
        point=x['Geometry']()(S,None)
        candidates=point[:,:,None,:].expand(-1,-1,2,-1)
        normal=point.new_tensor([0.,0.,1.]).expand_as(point)
        result=smooth_region_loss(candidates,target,normal,point.new_ones(6,2),
                                  normal_weight=kw.get('normal_weight',1.),tangent_weight=kw.get('tangent_weight',1.))
        return {key:result[key] for key in ('loss','normal','tangent')}


outputs=[]
for active in (False,True):
    data={k:v.clone() for k,v in x['d'].items()}
    data['active'].fill_(active)
    with patch('supermimic.contact_dmo.observation',lambda S,*args:S[0][:,:1]):
        actor=x['Actor']()
        loss,details=x['replay_loss'](actor,x['WM'](),x['clock'],x['Geometry'](),None,
            lambda v:v,None,data,completion_mode='accumulated_target',stop_steps=torch.tensor([2]),
            contact_surface=Surface())
        loss.backward()
        assert abs(details['terminal_contact']-16)<1e-4
        assert actor.weight.grad[:2].abs().min()>0 and actor.weight.grad[2]==0
        outputs.append(actor.weight.grad.clone())
torch.testing.assert_close(outputs[0],outputs[1])
print('PASS surface-region DMO: scalar loss, early-action gradients, cutoff/reset boundary, activation independence')

from types import SimpleNamespace
from supermimic.contact_control import event_policy_loss
clock=x['clock']
data={**clock.data,'s':x['d']['s'][:,0],'xi':x['d']['xi'][:,0],'a_prev':x['d']['a_prev'][:,0]}
clock=SimpleNamespace(**{**vars(clock),'data':data})
actor=x['Actor']();actor.include_time=False
actor.weight.data.fill_(.01)
with patch('supermimic.contact_control.observation',lambda S,*args:S[0][:,:1]):
    loss,details=event_policy_loss(actor,x['WM'](),clock,x['Geometry'](),None,lambda v:v,None,
        terminal_steps=torch.tensor([2]),contact_surface=Surface(),return_details=True)
    loss.backward()
    assert details['terminal_contact']>0 and actor.weight.grad[:2].abs().min()>0 and actor.weight.grad[2]==0
print('PASS surface-region imagination: actual entrypoint uses new objective and preserves early-action chain')
