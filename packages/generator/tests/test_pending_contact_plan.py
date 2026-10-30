from types import SimpleNamespace
import torch
from generator.pending_contact_plan import PendingContactPlans, planner_weights
from generator.full1000_position_predictor import Full1000PositionPredictor
from generator.next_interaction_heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
from contact_solver.device_contact_objective import DeviceWitnessRows


def test_retry_freezes_issued_intent_and_reset_clears_only_its_slot():
    pending = PendingContactPlans(3, 'cpu')
    pred = SimpleNamespace(conditioned_role=torch.tensor([[1,2,0,0,0,0],[2,1,0,0,0,0]]),
                           planned_regions=torch.ones(2,6,4,dtype=torch.bool))
    points = torch.randn(2,6,3,requires_grad=True)
    pending.update(torch.tensor([2,0]),pred,points,torch.tensor([True,False]))
    assert pending.mask.tolist()==[False,False,True]
    assert not pending.points.requires_grad
    torch.testing.assert_close(pending.batch(torch.tensor([2]))['repair_points_world'],points[0:1])


def test_repair_rows_have_zero_planner_gradient_without_diluting_fresh_rows():
    logits = torch.randn(3,4,requires_grad=True)
    weights = planner_weights(torch.tensor([True,False,True]))
    loss = (torch.nn.functional.cross_entropy(logits,torch.tensor([1,2,3]),reduction='none')*weights).mean()
    loss.backward()
    assert logits.grad[[0,2]].abs().sum()==0
    assert logits.grad[1].abs().sum()>0
    assert planner_weights(torch.ones(3,dtype=torch.bool)).sum()==0


def test_repair_preserves_world_plan_as_robot_moves_and_keeps_gradient_isolation():
    torch.manual_seed(10)
    model=Full1000PositionPredictor(24,1,8,event_roles=True).eval()
    q=torch.zeros(1,36);q[:,2]=.8;q[:,3]=1
    role=torch.tensor([[1,2,0,0,0,0]])
    points=torch.tensor([[[.3,.1,0.],[0.,-.1,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]])
    obs=dict(current_q=q,current_contact=role!=0,current_anchor=points,
             heightmap=torch.full((1,HEIGHTMAP_ROWS,HEIGHTMAP_COLS),-.8))
    kwargs=dict(repair_mask=torch.tensor([True]),repair_role=role,repair_points_world=points)
    pred=model(**obs,**kwargs)
    q2=q.clone();q2[:,0]+=.2
    other=model(**{**obs,'current_q':q2},**kwargs)
    torch.testing.assert_close(pred.conditioned_points_local+q[:,None,:3],points)
    torch.testing.assert_close(other.conditioned_points_local+q2[:,None,:3],points)
    assert torch.equal(other.conditioned_role,role)
    assert (other.conditioned_cell==-1).all()
    other.qpos.square().sum().backward()
    assert all(p.grad is None for p in model.training_parameter_groups()['planner'])
    assert model.execution_role_encoder.weight.grad.abs().sum()>0


def test_anchored_single_contact_can_pass_and_global_shift_is_free():
    rows=object.__new__(DeviceWitnessRows);rows.q=torch.zeros(1,36)
    active=torch.tensor([[True,False,False,False,False,False]])
    points=torch.zeros(1,6,3);points[0,0]=torch.tensor([1.,2.,0.])
    rows.pair={'geometry_point1_w':points[0].clone()}
    rows.spatial_representatives=lambda points,active,surface,actual=False,regions=None: (torch.arange(6)[None],active)
    surface=torch.zeros(1,6,dtype=torch.long)
    error,complete,count=rows.witness_position_statistics(points,active,surface,actual=True)
    assert complete.item() and count.item()==1 and error.item()==0
    rows.pair['geometry_point1_w']+=3
    error,complete,_=rows.witness_position_statistics(points+3,active,surface,actual=True)
    assert complete.item() and error.item()==0
    displaced=points+3;displaced[:,0,0]+=.2
    error,_,_=rows.witness_position_statistics(displaced,active,surface,actual=True)
    assert error.item()>.03


def test_body_relative_cell_loss_penalizes_future_only_shift():
    from generator.planned_contact_predictor import contact_plan_objective
    from contact_solver.contact_layout import relative_plan_objective
    H,W=HEIGHTMAP_ROWS,HEIGHTMAP_COLS
    role=torch.tensor([[1,2,0,0,0,0]])
    cell=torch.tensor([[H//2*W+W//2,H//2*W+W//2+5,0,0,0,0]])
    target=dict(role=role,contact_cell=cell,contact_cell_valid=torch.ones(1,6,dtype=torch.bool))
    logits=torch.full((1,6,4),-40.);logits.scatter_(2,role[...,None],40.)
    losses=[]
    for shift in (0,10*W):
        cells=cell.clone();cells[:,:2]+=shift
        locations=torch.full((1,6,H*W),-40.);locations.scatter_(2,cells[...,None],40.)
        pred=SimpleNamespace(role_logits=logits,location_logits=locations,role=role,contact=role!=0,cell=cells)
        losses.append(contact_plan_objective(pred,target)[0].item())
        assert relative_plan_objective(pred,target,torch.zeros(1,H,W))[0].item()==losses[-1]
    assert losses[0]==0 and losses[1]>0
