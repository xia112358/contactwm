"""Loss gradients at a fixed Newton witness; no synthetic contact truth."""
from types import SimpleNamespace

import pytest
import torch

from contact_solver.device_contact_objective import realization
from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE
from contact_solver.newton_witness_loss import selected_contact_cost


def fixture(distance, *, margin=.02, activated=None):
    q=torch.tensor([[distance]],dtype=torch.float64,requires_grad=True)
    active=torch.tensor([[True,False,False,False,False,False]])
    surface=torch.zeros(1,6,dtype=torch.long)
    on=distance<margin if activated is None else activated
    pair=dict(sample=torch.tensor([0]),part=torch.tensor([0]),
        task_pair=torch.tensor([True]),primary_surface=torch.tensor([0]),
        active=torch.tensor([on]),constraint_allocated=torch.tensor([on]),
        eligible=torch.tensor([on]),upward=torch.tensor([True]),
        includemargin=q.new_tensor([margin]),dist=q.detach().flatten())
    observed=dict(pairs=pair,configured_margin=q.new_tensor([margin]),
        contact_part_mask=active if on else torch.zeros_like(active),contact_surface=surface)
    class Rows:
        def __len__(self): return 1
        def full_body(self): return (-self.distances).relu(),self.q.new_zeros(1)
    rows=Rows()
    rows.q=q;rows.pair=pair;rows.observed=observed;rows.sample=pair['sample']
    rows.part=pair['part'];rows.group=pair['part'];rows.distances=q.flatten()
    rows.collision_aggregation='max'
    metadata=dict(part=0,surface=0,includemargin=margin,
                  constraint_active=on,allocated=on)
    return q,active,surface,rows,[[(metadata,q[0,0])]]


@pytest.mark.parametrize('distance,sign',[(.02404916286468506,1),(.019,0),
    (.002,0),(0.,0),(-.0005,0),(-.001,0),(-.004,-1)])
def test_complete_interval_gradient_and_no_duplicate_lower_penalty(distance,sign):
    q,active,surface,rows,cpu=fixture(distance)
    loss,metrics=realization(rows,active,surface)
    upper,missing,_=selected_contact_cost(q,cpu,active,surface,continuous_gap=True)
    torch.testing.assert_close(metrics['newton_plan_realization_loss'],2*upper[:,0]/.02**2)
    expected_lower=max(0.,-distance-DEFAULT_ACCEPTANCE.shallow_penetration_m)**2/.005**2
    expected_upper=2*max(0.,distance-.02)**2/.02**2
    torch.testing.assert_close(loss,q.new_tensor([expected_upper+expected_lower]))
    loss.sum().backward()
    assert q.grad.sign().item()==sign
    assert not missing.any()


def test_activation_flags_do_not_switch_off_frozen_witness_gradient():
    # A frozen active query cannot switch off a residual as FK moves outside.
    q,active,surface,rows,cpu=fixture(.024,activated=True)
    loss,_=realization(rows,active,surface)
    upper,_,realized=selected_contact_cost(q,cpu,active,surface,continuous_gap=True)
    assert realized[0,0] and upper[0,0]>0
    loss.sum().backward()
    assert q.grad.item()>0


def test_margin_is_actual_pair_value_and_boundary_does_not_claim_contact():
    for margin in (.01,.03):
        q,active,surface,rows,_=fixture(margin,margin=margin)
        loss,metrics=realization(rows,active,surface)
        assert loss.item()==0
        assert metrics['newton_contact_accepted'].item()==0  # Strict dist < margin.


def test_missing_pair_is_reported_not_realized():
    q=torch.zeros(1,1,requires_grad=True)
    active=torch.tensor([[True,False,False,False,False,False]])
    loss,missing,realized=selected_contact_cost(q,[[]],active,
        torch.zeros(1,6,dtype=torch.long),continuous_gap=True)
    assert missing[0,0] and not realized.any()
    loss.sum().backward()
    assert q.grad is not None


@pytest.mark.parametrize('height,sign',[(.024,1),(.019,0),(-.005,0)])
def test_no_witness_approach_keeps_upper_gradient_without_pinning_surface(height,sign):
    from generator.conditioned_pose_predictor import _missing_intent_surface_distance
    class Geometry:
        shapes=[SimpleNamespace(link_name='foot',part=0)]
        def approach_residual(self,fk,q,part,normal,offset,*args,**kwargs):
            return (q[:,2]-offset)[:,None]*normal
    class FK:
        def link_poses(self,q,names):
            return q[:,:3,None].transpose(1,2),torch.eye(3).to(q)[None,None]
    q=torch.zeros(1,36,dtype=torch.float64);q[0,2]=height;q.requires_grad_()
    model=SimpleNamespace(_establishment_geometry=Geometry(),fk=FK())
    scene=dict(box_center=q.new_zeros(1,3),box_rotation=torch.eye(3).to(q)[None],
               box_half_extents=q.new_ones(1,3),ground_height=q.new_zeros(1))
    surface=torch.tensor([[0,-1,-1,-1,-1,-1]])
    loss=_missing_intent_surface_distance(model,q,surface,scene,q.new_tensor([.02]))
    loss.sum().backward()
    assert q.grad[0,2].sign().item()==sign
