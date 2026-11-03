import torch
from supermimic.contact_control import ContactClock


def data(n=80):
    active=torch.zeros(n,6,dtype=torch.bool)
    return dict(s=torch.zeros(n,71),active=active,known=torch.ones_like(active),surface=torch.zeros(n,6,dtype=torch.long),load=torch.ones(n,6))


def test_close_real_arrivals_are_retained_without_priority_deletion():
    d=data();d['active'][10:,0]=True;d['active'][25:,2]=True;d['active'][40:,1]=True
    c=ContactClock(d,hold=1,role_aware=True,boundary_policy='event_preserving_v2')
    assert c.bounds==[0,11,26,41,80]
    assert all(g['retained'] for g in c.boundary_audit['arrival_groups'])
    for end,part in [(11,0),(26,2),(41,1)]:assert c.target_masks[end-1,part]
    assert not c.target_masks[-1].any()
    assert ContactClock(d,hold=1,role_aware=True).bounds!=c.bounds


def test_cooperative_arrivals_and_short_recontact_keep_their_semantics():
    d=data();d['active'][:,0]=True;d['active'][15:20,0]=False
    d['active'][30:,2]=True;d['active'][35:,3]=True
    original=d['active'].clone()
    c=ContactClock(d,hold=1,role_aware=True,boundary_policy='event_preserving_v2')
    assert c.bounds==[0,36,80]
    assert c.target_masks[35].nonzero().flatten().tolist()==[2,3]
    assert len(c.boundary_audit['suppressed_recontacts'])==1
    assert c.boundary_audit['departures']==[dict(frame=15,part=0)]
    assert torch.equal(d['active'],original)


def test_surface_transfer_and_final_arrival_are_not_lost():
    d=data(30);d['active'][:,0]=True;d['surface'][12:,0]=1;d['active'][29:,2]=True
    c=ContactClock(d,hold=1,role_aware=True,boundary_policy='event_preserving_v2')
    assert c.bounds==[0,13,30]
    assert c.target_masks[12,0] and c.target_masks[29,2]
