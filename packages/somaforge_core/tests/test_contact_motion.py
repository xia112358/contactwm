import numpy as np
import pytest
import torch
from somaforge_core.contact_motion import (calibrate_pivot_budgets,
    contact_region_statistics,loaded_motion_statistics,same_material_tangent_motion)


def test_pivot_and_loaded_slip_are_independent():
    distances=torch.tensor([0.,.01,.01,.01])
    region=contact_region_statistics(distances,torch.zeros(4,dtype=torch.long),2)
    assert region['pivot'][0]==0 and region['lower_quartile'][0]>0
    loaded=loaded_motion_statistics(distances.numpy(),[1,3,3,3],[0]*4,2)
    assert loaded['rms_tangent_speed_m_s'][0]>.009
    assert np.isnan(loaded['rms_tangent_speed_m_s'][1])
    assert torch.isnan(region['pivot'][1])


def test_empirical_budget_uses_event_duration_without_witness_resets():
    steps=torch.tensor([[.001],[.002],[.003],[.004]])
    row=calibrate_pivot_budgets(steps,[(0,4,0)])[0]
    assert row['budget_m']>=row['source_path_m']
    assert row['budget_m']==pytest.approx(.0055*4)
    steps[2]=float('nan')
    with pytest.raises(ValueError,match='missing'):calibrate_pivot_budgets(steps,[(0,4,0)])


def test_source_excursion_does_not_expand_noise_budget():
    steps=torch.full((100,1),.001);steps[20:25]=.02
    row=calibrate_pivot_budgets(steps,[(0,100,0)])[0]
    assert row['budget_m']==pytest.approx(.1)
    assert row['source_over_budget']


def test_projection_uses_actual_plane_and_keeps_material_identity():
    p=torch.zeros(2,1,3,requires_grad=True);r=torch.eye(3).repeat(2,1,1,1)
    p.data[1,0]=torch.tensor([.02,.01,0])
    delta=same_material_tangent_motion(p,r,torch.tensor([0]),torch.tensor([0]),
                                    torch.tensor([[10.,0,0]]),torch.tensor([[1.,0,0]]))
    torch.testing.assert_close(delta,torch.tensor([[0.,.01,0]]))
    delta.square().sum().backward();assert torch.isfinite(p.grad).all()
