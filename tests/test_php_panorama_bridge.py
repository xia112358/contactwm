import torch
import pytest
from holosoma.acp.php_bridge import PLANNER_PARTS,PHP_PARTS,reorder,tracker_contacts
from holosoma.acp.panorama_model import PanoramaContactPlanner,panorama_planner_loss


def test_permutation_and_actor_contract():
    xyz=torch.arange(18.).reshape(1,6,3);active=torch.tensor([[True,False,True,False,True,False]])
    command=tracker_contacts(dict(positions=xyz,occupancy=active))
    assert command.shape==(1,24)
    expected=[0,1,4,5,2,3]
    assert torch.equal(command[0,18:],active[0,expected].float())
    assert torch.equal(command[:,:18].reshape(1,6,3),torch.where(active[:,expected,None],xyz[:,expected],0))
    assert torch.equal(reorder(reorder(xyz,PLANNER_PARTS,PHP_PARTS),PHP_PARTS,PLANNER_PARTS),xyz)
    with pytest.raises(ValueError):reorder(xyz,PLANNER_PARTS,PHP_PARTS[:-1])


def test_world_targets_backpropagate_after_php_reordering():
    m=PanoramaContactPlanner(height=4,width=8)
    p=m(torch.zeros(1,465),torch.zeros(1,3),torch.ones(1,4,8),torch.ones(1,4,8,dtype=torch.bool),
        torch.zeros(1,4,8),torch.zeros(1,3),torch.tensor([[0.,0.,.73]]),torch.zeros(1))
    php_target=reorder(p['ungated_positions_w'].detach(),PLANNER_PARTS,PHP_PARTS)
    target=reorder(php_target,PHP_PARTS,PLANNER_PARTS)
    loss=panorama_planner_loss(p,target,torch.ones(1,6,dtype=torch.bool),torch.ones(1,6,dtype=torch.bool))
    loss['loss'].backward()
    assert loss['positive_count']==6
    assert m.attention.in_proj_weight.grad.abs().sum()>0
