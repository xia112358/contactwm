import pytest
import torch
from generator.full1000_position_predictor import Full1000PositionPredictor
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS


def test_execution_gradient_mask_preserves_predictions_and_supervised_gradients():
    torch.manual_seed(12); torch.set_num_threads(2)
    m=Full1000PositionPredictor(24, 1, 4, event_roles=True, execution_observation_gradients=True).eval()
    q=torch.zeros(2,36);q[:,2]=.8;q[:,3]=1
    obs=dict(current_q=q,current_contact=torch.zeros(2,6,dtype=torch.bool),current_anchor=torch.zeros(2,6,3),
        heightmap=torch.zeros(2,HEIGHTMAP_ROWS,HEIGHTMAP_COLS))
    mask=torch.tensor([True,False])
    a=m(**obs);b=m(**obs,execution_observation_supervised=mask)
    torch.testing.assert_close(a.qpos,b.qpos,rtol=0,atol=0)
    torch.testing.assert_close(a.raw_location_logits if a.raw_location_logits is not None else a.location_logits,
        b.raw_location_logits if b.raw_location_logits is not None else b.location_logits)
    shared=m.part_encoder.weight
    executor=m.pose_head.weight
    # Unsupervised execution still trains the executor, but not shared inputs.
    grad=torch.autograd.grad(b.qpos[1].square().sum(), (shared,executor), retain_graph=True)
    assert grad[0].count_nonzero()==0 and grad[1].abs().sum()>0
    masked=torch.autograd.grad(b.qpos[0].square().sum(),shared)[0]
    unmasked=torch.autograd.grad(a.qpos[0].square().sum(),shared)[0]
    torch.testing.assert_close(masked,unmasked,rtol=0,atol=0)
    assert masked.abs().sum()>0
    # The planner path remains trainable even on a masked execution sample.
    c=m(**obs,execution_observation_supervised=mask)
    pg=torch.autograd.grad(c.role_logits[1].square().sum()+c.location_logits[1].square().mean(),shared)[0]
    assert pg.abs().sum()>0
    with pytest.raises(ValueError,match='boolean'):
        m(**obs,execution_observation_supervised=torch.ones(2))
