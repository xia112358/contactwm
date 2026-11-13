import torch
from supermimic.contact_collision_objective import classify_contacts,collision_loss


def test_old_support_and_unknown_are_not_collisions():
    active=torch.ones(4,dtype=torch.bool);known=torch.tensor([True,True,False,True]);transfer=torch.tensor([False,True,True,True])
    target,other=classify_contacts(active,known,transfer,torch.tensor([[0.,0.,0.],[2.,0.,0.],[0.,0.,0.],[0.,0.,0.]]),torch.full((4,1,3),-.1),torch.full((4,1,3),.1),torch.ones(4,1,dtype=torch.bool))
    assert target.tolist()==[False,False,False,True]
    assert other.tolist()==[False,True,False,False]


def test_touch_and_penetration_have_outward_spatial_gradient():
    p=torch.zeros(3,3,requires_grad=True);n=torch.tensor([[0.,0.,1.]]).expand(3,-1)
    cost=collision_loss(p,n,torch.tensor([0.,-.01,.02]),p.detach(),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    torch.testing.assert_close(cost,torch.tensor([.04,.16,0.]))
    cost.sum().backward()
    torch.testing.assert_close(p.grad[:,2],torch.tensor([-8.,-16.,0.]))
    assert not p.grad[:,:2].any()


def test_in_region_and_inactive_have_zero_cost_and_gradient():
    p=torch.zeros(3,3,requires_grad=True);n=torch.tensor([[0.,0.,1.]]).expand(3,-1)
    cost=collision_loss(p,n,torch.full((3,),-.01),p.detach(),torch.tensor([True,False,True]),torch.tensor([True,False,False]))
    cost.sum().backward()
    assert not cost[:2].any() and not p.grad[:2].any()
    assert cost[2]>0 and p.grad[2,2]<0


def test_only_native_harmful_collision_interrupts():
    from supermimic.contact_collision_objective import collision_interruption
    # Native monitoring owns region, support, duration and impulse classification.
    harmful=torch.tensor([[False,False],[False,False],[True,False],[False,True]])
    assert collision_interruption(harmful).tolist()==[False,False,True,True]


def test_native_impact_is_not_erased_by_substep_to_frame_rebound():
    # The native solve touched z=0. The frame ended 3 cm above that surface.
    p=torch.tensor([[0.,0.,.03]],requires_grad=True)
    n=torch.tensor([[0.,0.,1.]])
    cost=collision_loss(p,n,torch.zeros(1),torch.zeros(1,3),torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
    cost.sum().backward()
    torch.testing.assert_close(cost,torch.tensor([.04]))
    assert p.grad[0,2]<0


def test_deep_penetration_has_bounded_slope_without_changing_small_errors():
    p=torch.zeros(2,3,requires_grad=True);normal=torch.tensor([[0.,0.,1.]]).expand(2,-1)
    cost=collision_loss(p,normal,torch.tensor([-.10,-1.]),p.detach(),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
    cost.sum().backward()
    torch.testing.assert_close(p.grad[:,2],torch.tensor([-40.,-40.]))
    assert cost[1]>cost[0]>0


def test_prefix_spatial_loss_reaches_earlier_action_but_not_distant_plane():
    from supermimic.contact_collision_objective import collision_prefix_loss
    earlier=torch.tensor(0.,requires_grad=True)
    # Both samples are 5 mm above a future collision surface; one is 1 m away.
    point=torch.stack((torch.stack((earlier*0,earlier*0,.005+earlier)),
                       torch.stack((1.+earlier*0,earlier*0,.005+earlier))))
    cost,mask=collision_prefix_loss(point,torch.tensor([[0.,0.,1.]]).expand(2,-1),
        torch.zeros(2),torch.zeros(2,3),torch.ones(2,dtype=torch.bool))
    cost.sum().backward()
    assert mask.tolist()==[True,False] and cost[1]==0
    assert earlier.grad<0  # gradient descent raises the pre-collision material point


def test_future_witness_cannot_cross_contact_stage_or_attempt_boundary():
    from supermimic.contact_collision_objective import collision_prefix_witnesses
    T,B=6,4
    data=dict(done=torch.zeros(T,B,dtype=torch.bool),goal_index=torch.zeros(T,B,dtype=torch.long),
        active=torch.zeros(T,B,6,dtype=torch.bool),known=torch.ones(T,B,6,dtype=torch.bool),
        collision_mask=torch.zeros(T,B,6,dtype=torch.bool),collision_in_region=torch.zeros(T,B,6,dtype=torch.bool))
    data['collision_mask'][4,:,0]=True
    data['active'][4,:,0]=True
    data['active'][2,1,0]=True  # already present support blocks carry
    data['goal_index'][3:,2]=1  # another task's collision cannot supervise this task
    stop=torch.tensor([6,6,6,3])  # stored reset rollout after frame 3 is excluded
    indices=collision_prefix_witnesses(data,stop,torch.ones(T,B,6,dtype=torch.bool))
    assert indices[:,0,0].tolist()==[4,4,4,4,4,-1]
    assert indices[:,1,0].tolist()==[-1,-1,-1,4,4,-1]
    assert indices[:,2,0].tolist()==[-1,-1,-1,4,4,-1]
    assert indices[:,3,0].eq(-1).all()
    assert indices[:,:,1:].eq(-1).all()


def test_no_prefix_for_in_region_or_old_support_or_absent_witness():
    from supermimic.contact_collision_objective import collision_prefix_witnesses
    data=dict(done=torch.zeros(3,1,dtype=torch.bool),goal_index=torch.zeros(3,1,dtype=torch.long),
        active=torch.zeros(3,1,6,dtype=torch.bool),known=torch.ones(3,1,6,dtype=torch.bool),
        collision_mask=torch.zeros(3,1,6,dtype=torch.bool),collision_in_region=torch.zeros(3,1,6,dtype=torch.bool))
    data['collision_mask'][2,0,:2]=True
    data['collision_in_region'][2,0,0]=True
    required=torch.ones(3,1,6,dtype=torch.bool);required[:,:,1]=False
    indices=collision_prefix_witnesses(data,torch.tensor([3]),required)
    assert indices.eq(-1).all()


def test_prefix_distance_uses_native_anchor_not_frame_end_as_zero_gap():
    from supermimic.contact_collision_objective import collision_prefix_loss
    point=torch.tensor([[0.,0.,.03]],requires_grad=True)
    loss,_=collision_prefix_loss(point,torch.tensor([[0.,0.,1.]]),torch.zeros(1),torch.zeros(1,3),torch.ones(1,dtype=torch.bool))
    loss.sum().backward()
    assert loss.item()==0 and not point.grad.any()
