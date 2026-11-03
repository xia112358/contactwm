import torch
from supermimic.contact_task import has_fallen


def test_fall_accepts_deep_lean_and_rejects_inversion_before_low_root():
    p=torch.zeros(4,3);p[:,2]=torch.tensor([.4,.4,.172,.149])
    R=torch.eye(3).repeat(4,1,1);R[:,2,2]=torch.tensor([.1149,-.0267,-.243,1.])
    assert has_fallen((p,R)).tolist()==[False,True,True,True]
