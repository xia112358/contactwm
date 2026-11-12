import torch
from types import SimpleNamespace
from supermimic.contact_initial import reference_entry_phase


def clock(bits,known=None,floors=None):
    a=torch.tensor(bits,dtype=torch.bool).reshape(-1,2,1).expand(-1,-1,6).clone()
    k=torch.ones_like(a) if known is None else torch.tensor(known,dtype=torch.bool).reshape(-1,2,1).expand_as(a)
    return SimpleNamespace(data={'substep_raw_contact_active':a,'substep_contact_known':k,'substep_part_position':torch.zeros(*a.shape,3)},frame_source_floor=torch.zeros(len(a),dtype=torch.long) if floors is None else torch.tensor(floors),segment_starts=torch.arange(len(a)))


def test_history_carries_short_dropout_and_already_observed_release_count():
    b,a=reference_entry_phase(clock([1,1,1,0,0,0,0,0]),4)
    assert b[:,0].tolist()==[False,True,True,True]
    assert a[:,0].tolist()==[0,0,1,3]


def test_airborne_entry_and_confirmed_release_are_immediately_armed():
    b,a=reference_entry_phase(clock([0,0,0,0,1,0,0,0,0,0,0,0]),4)
    assert b[:,0].tolist()==[False,False,False,True,True,False]


def test_source_boundary_cannot_inherit_another_trajectory_contact():
    b,_=reference_entry_phase(clock([1,1,1,1,0,0,0,0],floors=[0,0,2,2]),4)
    assert b[:,0].tolist()==[False,True,False,False]


def test_unknown_does_not_confirm_release():
    b,a=reference_entry_phase(clock([1,0,0,0,0,0,0,0],known=[1,1,1,0,1,1,1,1]),4)
    assert b[3,0] and a[3,0]==2
