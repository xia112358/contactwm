import pytest
import torch

from generator.endpoint_objective import compose_endpoint_loss, LOSS_GROUPS
from generator.full1000_training_variants import endpoint_feedback_mask, endpoint_layout_pass
from generator.rollout_termination import TerminationLimits, native_continuation_masks
from contact_solver.contact_layout import endpoint_position_loss
from contact_solver.interaction_acceptance import configured_acceptance
from contact_solver.endpoint_task import ENDPOINT_POSITION_TOLERANCE_M


def test_loss_contributions_are_exact_and_each_gradient_is_counted_once():
    terms={name:torch.tensor([float(i),float(i+1)],requires_grad=True)
           for i,name in enumerate(LOSS_GROUPS,1)}
    # A compatibility mask disables only demonstration imitation on row 2.
    weighted=dict(terms,imitation=terms['imitation']*torch.tensor([1.,0.]))
    total,contributions=compose_endpoint_loss(**weighted)
    torch.testing.assert_close(total,sum(contributions.values()),rtol=0,atol=0)
    assert len(contributions)==5
    total.sum().backward()
    for name,value in terms.items():
        torch.testing.assert_close(value.grad,torch.tensor([1.,0.]) if name=='imitation' else torch.ones(2))


@pytest.mark.parametrize('tolerance',[.02,ENDPOINT_POSITION_TOLERANCE_M,.06])
def test_spatial_loss_and_acceptance_share_configured_boundary(tolerance):
    limits=configured_acceptance({'acceptance_penetration_m':.03,
                                  'endpoint_position_tolerance_m':tolerance,'joint_excess_rad':.07})
    error=torch.tensor([tolerance**2,(tolerance+.001)**2],dtype=torch.float64)
    complete=torch.ones(2,dtype=torch.bool)
    accepted=endpoint_layout_pass(error,complete,torch.ones(2),limits.endpoint_position_m)
    assert accepted.tolist()==[True,False]
    loss=endpoint_position_loss(error,limits.endpoint_position_m)
    assert loss[0]==0 and loss[1]>0


def test_joint_threshold_is_shared_by_endpoint_feedback_and_continuation():
    limits=configured_acceptance({'acceptance_penetration_m':.03,'joint_excess_rad':.07})
    q=torch.zeros(2,36);q[:,3]=1
    actual=torch.ones(2,6,dtype=torch.bool)
    metrics=dict(solid_penetration_cm=torch.zeros(2),newton_contact_accepted=torch.ones(2),
        newton_invalid_fullbody_witnesses=torch.zeros(2),joint_violation_rad=torch.tensor([.07,.071]))
    passed=endpoint_feedback_mask(metrics,torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),limits=limits)
    continuation,_=native_continuation_masks(q,metrics,actual,
        limits=TerminationLimits(penetration_m=limits.shallow_penetration_m,joint_excess_rad=limits.joint_excess_rad))
    assert passed.tolist()==continuation.tolist()==[True,False]


def test_old_checkpoints_keep_historical_defaults():
    old=configured_acceptance({'acceptance_penetration_m':.005})
    assert old.shallow_penetration_m==.005
    assert old.endpoint_position_m==.04 and old.joint_excess_rad==.15
    assert configured_acceptance({}).shallow_penetration_m==.001
