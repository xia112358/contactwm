import torch
import pytest
from generator.research.archived_experiments.paired_closed_loop_replay import compatible_current,successor_roles,DemonstrationSuccessorTeacher
from generator.research.archived_experiments.paired_closed_loop_replay import embedded_hold_depth
from generator.research.archived_experiments.paired_replay_gate import passes_gate
import numpy as np


def test_extra_real_support_is_not_dropped_for_match():
    active=torch.tensor([True,True]);surface=torch.tensor([0,1])
    candidates=torch.tensor([[True,False],[True,True],[True,True]])
    surfaces=torch.tensor([[0,-1],[0,0],[0,1]])
    assert compatible_current(active,surface,candidates,surfaces).tolist()==[False,False,True]


def test_missing_support_is_recovery_not_persistence():
    c=dict(active_contact=torch.tensor([True,True]),contact_surface=torch.tensor([1,0]),
        persistent_support=torch.tensor([True,True]),touchdown=torch.tensor([False,False]))
    assert successor_roles(torch.tensor([False,True]),torch.tensor([-1,0]),c).tolist()==[1,2]


def test_validation_demonstrations_cannot_be_teachers():
    with pytest.raises(ValueError,match='TRAINING'):
        DemonstrationSuccessorTeacher([dict(sequence=0,supervision_kind='demonstration',split='validation_height_090')],0,None)


def test_embedded_hold_is_detected_without_erasing_actual_contact():
    active=np.array([True,True]);surface=np.array([0,1]);anchor=np.array([[0,0,-.06],[0,0,1.]])
    scene=dict(box_center=torch.tensor([[0.,0.,.5]]),box_half_extents=torch.tensor([[1.,1.,.5]]),ground_height=torch.tensor([0.]))
    original=active.copy();depth=embedded_hold_depth(active,surface,anchor,scene,[2,2])
    np.testing.assert_allclose(depth,[.06,0]);np.testing.assert_array_equal(active,original)


def test_nonpersistent_contact_is_not_reclassified_as_absent():
    active=np.array([True]);scene=dict(box_center=torch.tensor([[0.,0.,.5]]),box_half_extents=torch.tensor([[1.,1.,.5]]),ground_height=torch.tensor([0.]))
    depth=embedded_hold_depth(active,np.array([0]),np.array([[0,0,-.06]]),scene,[1])
    assert depth[0]==0 and active[0]


@pytest.mark.parametrize('exact,pen,hold,joint',[(False,0.,0.,0.),(True,.01,0.,0.),(True,0.,.1,0.),(True,0.,0.,.1)])
def test_gate_requires_all_conditions_not_just_low_loss(exact,pen,hold,joint):
    assert not passes_gate(exact,pen,hold,joint)


def test_gate_accepts_verified_consistent_witness():
    assert passes_gate(True,.0001,.0001,0.)
