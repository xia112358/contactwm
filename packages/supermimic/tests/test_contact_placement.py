import itertools
import numpy as np
import torch
from supermimic.contact_placement import footprint,placement_link,associate_support_placements
from supermimic.contact_substep_support import SubstepSupport


def mesh():return np.array(list(itertools.product([-.1,.1],[-.04,.04],[0.,.04])))


def test_common_placement_cannot_follow_a_chain_of_small_moves():
    v=mesh();region=footprint(v);R=np.eye(3)[None]
    for x in [.06,.12,.18]:
        metric,region=placement_link(region,0.,(np.array([[x,0,0]]),R),v)
        assert metric['matched']
    metric,_=placement_link(region,0.,(np.array([[.24,0,0]]),R),v)
    assert not metric['matched']


def test_return_to_same_place_after_real_excursion_is_not_continuation():
    v=mesh();R=np.tile(np.eye(3),(3,1,1))
    for p in [np.array([[0,0,0],[.3,0,0],[0,0,0]]),np.array([[0,0,0],[0,0,.04],[0,0,0]])]:
        metric,_=placement_link(footprint(v),0.,(p,R),v)
        assert not metric['matched']


def evidence(onset=80,unknown=False,other_face=False):
    T=max(200,onset+80);P=torch.zeros(T,1,6,3);R=torch.eye(3).expand(T,1,6,3,3).clone()
    known=torch.ones(T,1,6,dtype=torch.bool);active=torch.zeros_like(known);active[:40,0,0]=True;active[onset:,0,0]=True
    if unknown:known[50,0,0]=False
    surfaces=torch.full((T,1,6),-1,dtype=torch.long);surfaces[active]=0
    if other_face:surfaces[onset:,0,0]=1
    model=SubstepSupport(.002);samples=[]
    for t in range(T):
        obs=dict(active=active[t],known=known[t],surface_local=surfaces[t])
        features=dict(contact_load=active[t].float()*100,contact_material_speed=torch.zeros(1,6))
        samples.append(model.update(obs,features,(P[t],R[t]),torch.tensor([t])))
    data={'substep_support_'+k:torch.stack([s[k][0] for s in samples]).reshape(-1,10,6) for k in samples[0]}
    data.update(ep_step=torch.arange(T//10),surface_local=surfaces[::10,0])
    raw=dict(position=P.numpy(),rotation=R.numpy(),surfaces=surfaces.numpy(),known=known.numpy(),dt=.002,meshes=[mesh() for _ in range(6)])
    return data,raw


def test_same_surface_short_unload_associated_without_changing_observations():
    d,e=evidence();before=e['known'].copy();f,a=associate_support_placements(d,e,0)
    assert f['support_placement_resume'][8,0]
    assert not d['substep_support_state'][6,0,0]
    assert np.array_equal(e['known'],before)
    assert a[0]['linked'] and a[0]['gap_s']==.08


def test_unknown_face_change_and_long_idle_are_not_associated():
    for kw in [dict(unknown=True),dict(other_face=True),dict(onset=240)]:
        d,e=evidence(**kw);f,a=associate_support_placements(d,e,0)
        assert not f['support_placement_resume'].any()
