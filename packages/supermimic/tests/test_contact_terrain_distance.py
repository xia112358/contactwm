import torch
from supermimic.contact_terrain_distance import TerrainNavigationDistance

BOX=[[0,0,.5,1,0,0,0,1,1,1]]

def test_all_contact_surfaces_keep_original_zero_target():
    metric=TerrainNavigationDistance(BOX)
    targets=torch.tensor([[0,0,1.],[-.5,0,.4],[-1.,0,0.],[0,0,.9999]])
    prepared=metric.prepare(targets)
    assert torch.equal(prepared[0],targets)
    torch.testing.assert_close(metric.distance(targets,prepared),torch.zeros(4),atol=1e-6,rtol=0)

def test_ground_target_routes_around_box_and_retains_gradient():
    metric=TerrainNavigationDistance(BOX);goal=torch.tensor([[1.,0,0.]])
    point=torch.tensor([[-1.,.1,0.]],requires_grad=True)
    value=metric.distance(point,metric.prepare(goal))
    assert value.item()>(point-goal).norm().item()
    grad=torch.autograd.grad(value.sum(),point)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    delta=torch.tensor([[0,.001,0.]])
    fd=(metric.distance(point.detach()+delta,metric.prepare(goal))-metric.distance(point.detach()-delta,metric.prepare(goal)))/.002
    torch.testing.assert_close(grad[:,1],fd,atol=2e-3,rtol=2e-3)

def test_free_space_and_no_boxes_are_exact_euclidean():
    for boxes in [[],BOX]:
        metric=TerrainNavigationDistance(boxes);goal=torch.tensor([[-1.,0,.3]])
        point=torch.tensor([[-1.,0,.7]],requires_grad=True)
        torch.testing.assert_close(metric.distance(point,metric.prepare(goal)),torch.tensor([.4]),atol=1e-6,rtol=0)

def test_floating_box_has_underpass_and_multiple_boxes_work():
    metric=TerrainNavigationDistance([[0,0,1.5,1,0,0,0,1,1,1],[3,0,.5,1,0,0,0,1,1,1]])
    goal=torch.tensor([[1.,0,.1]]);point=torch.tensor([[-1.,0,.1]])
    torch.testing.assert_close(metric.distance(point,metric.prepare(goal)),torch.tensor([2.]),atol=1e-5,rtol=0)


def test_yaw_rotation_preserves_navigation_distance():
    angle=torch.tensor(.6);c,s=angle.cos(),angle.sin()
    rotation=torch.tensor([[c,-s,0],[s,c,0],[0,0,1.]])
    original=TerrainNavigationDistance(BOX)
    rotated=TerrainNavigationDistance([[0,0,.5,float((angle/2).cos()),0,0,float((angle/2).sin()),1,1,1]])
    p=torch.tensor([[-1.,.1,.1]]);g=torch.tensor([[1.,0,.1]])
    torch.testing.assert_close(original.distance(p,original.prepare(g)),rotated.distance(p@rotation.T,rotated.prepare(g@rotation.T)),atol=2e-4,rtol=1e-4)


def test_every_part_uses_own_material_goal_without_cross_pair_matching():
    from types import SimpleNamespace
    from supermimic.handoff.navigation_objective import MaterialNavigationObjective
    local=torch.zeros(1,6,4,3);local[0,:2]=torch.tensor([[-.1,-.05,0],[.1,-.05,0],[-.1,.05,0],[.1,.05,0]]);target=local.clone()
    paired=torch.zeros(1,6,2,3);goals=paired.clone();valid=torch.ones(1,6,2,dtype=torch.bool)
    paired[0,4,1,0]=10;goals[0,4,0,0]=10
    clock=SimpleNamespace(event_ends=torch.tensor([1]),segment_motion=torch.tensor([0]),case_ids=torch.tensor([1]),target_masks=torch.ones(1,6,dtype=torch.bool),keep_masks=torch.zeros(1,6,dtype=torch.bool),data={'point':torch.zeros(1,6,3)},material_local=local,material_targets=target,paired_material_local=paired,paired_material_targets=goals,paired_material_valid=valid)
    objective=MaterialNavigationObjective(clock,SimpleNamespace(tasks=[dict(boxes=[])]),SimpleNamespace(triangles_0=local[0,0],triangles_1=local[0,1]))
    assert {row['part'] for row in objective.rows}==set(range(6))
    position=torch.zeros(1,6,3);rotation=torch.eye(3).expand(1,6,3,3)
    required=torch.ones(1,6,dtype=torch.bool);candidate=torch.zeros(1,6,dtype=torch.long)
    cost=objective.cost(position,rotation,torch.tensor([0]),required,required,candidate,position)
    assert cost[0,4]>39999 and torch.equal(cost[0,:4],torch.zeros(4))


def test_hand_navigation_ignores_material_anchor_and_matches_native_point_error():
    from types import SimpleNamespace
    from supermimic.handoff.navigation_objective import MaterialNavigationObjective
    from supermimic.contact_material import mixed_error_sq
    local=torch.randn(1,6,4,3)*10
    material_targets=torch.randn_like(local)*10
    target=torch.zeros(1,6,3)
    required=torch.tensor([[False,False,True,True,False,False]])
    clock=SimpleNamespace(event_ends=torch.tensor([1]),segment_motion=torch.tensor([0]),case_ids=torch.tensor([85]),target_masks=required,keep_masks=torch.zeros_like(required),material_local=local,material_targets=material_targets,data={'point':target})
    objective=MaterialNavigationObjective(clock,SimpleNamespace(tasks=[dict(boxes=[])]),SimpleNamespace(triangles_0=local[0,0],triangles_1=local[0,1]))
    p=torch.zeros(1,6,3);R=torch.eye(3).expand(1,6,3,3)
    point=torch.zeros(1,6,3);point[:,2,0]=.03;point[:,3,1]=.04
    point.requires_grad_()
    args=(torch.tensor([0]),required,required,torch.zeros(1,6,dtype=torch.long),point)
    cost=objective.cost(p,R,*args)
    expected=mixed_error_sq(local,material_targets,point,target)/.05**2
    torch.testing.assert_close(cost[:,2:4],expected[:,2:4])
    torch.testing.assert_close(objective.cost(p+10,-R,*args),cost)
    gradient=torch.autograd.grad(cost.sum(),point)[0]
    torch.testing.assert_close(gradient[:,2:4],2*point[:,2:4]/.05**2)
    zero=objective.cost(p,R,*args[:-1],torch.zeros_like(point))
    assert torch.equal(zero,torch.zeros_like(zero))


def test_exit_switch_is_continuous_and_penetration_descent_escapes():
    metric=TerrainNavigationDistance(BOX)
    x=torch.tensor([[-.0001,0,.5],[.0001,0,.5]])
    value=metric.distance(x,metric.prepare(torch.tensor([[1.,0,.5]])))
    assert abs(float(value[0]-value[1]))<.003
    x=torch.tensor([[.4,.1,.5]],requires_grad=True)
    prepared=metric.prepare(torch.tensor([[-1.,.1,.5]]))
    before=metric.distance(x,prepared)
    grad=torch.autograd.grad(before.sum(),x)[0]
    after=x.detach()-.001*grad
    depth=lambda p:(.5-(p-p.new_tensor([0,0,.5])).abs()).amin(-1)
    assert depth(after)<depth(x)
    assert metric.distance(after,prepared)<before


def test_parallel_near_surface_paths_have_continuous_escape_cost():
    metric=TerrainNavigationDistance(BOX)
    values=[]
    for eps in [1e-3,1e-4,1e-5]:
        p=torch.tensor([[-.3,0,1-eps]])
        g=torch.tensor([[.3,0,1-eps]])
        values.append(float(metric.distance(p,metric.prepare(g))))
    assert values[0]>values[1]>values[2]
    assert abs(values[-1]-.6)<.001


def test_union_does_not_double_penalize_overlapping_solids():
    one=TerrainNavigationDistance(BOX);two=TerrainNavigationDistance(BOX*2)
    x=torch.tensor([[-.2,.1,.5]],requires_grad=True);g=torch.tensor([[.2,.1,.5]])
    torch.testing.assert_close(one.distance(x,one.prepare(g)),two.distance(x,two.prepare(g)))


def test_whole_foot_overlap_detected_with_all_corners_outside_and_goal_zero():
    from supermimic.contact_foot_occupancy import FootOccupancy
    metric=TerrainNavigationDistance([[0,0,.5,1,0,0,0,.1,.1,1]])
    local=torch.tensor([[-.4,-.1,-.05],[.4,-.1,-.05],[-.4,.1,-.05],[.4,.1,-.05]])
    vertices=torch.cat((local,local+torch.tensor([0,0,.1])))
    goal=local+torch.tensor([2.,0,.5])
    foot=FootOccupancy(vertices,local,goal,metric)
    p=torch.tensor([[0.,0,.5]],requires_grad=True);R=torch.eye(3)[None]
    corners=p[:,None]+vertices
    assert (((corners-metric.centers[0]).abs()>=metric.half[0]).any(-1)).all()
    assert foot.overlap(p,R)[0,1]>.1
    assert foot.cost(p,R)>0
    zero=foot.cost(foot.goal_position[None],foot.goal_rotation[None])
    torch.testing.assert_close(zero,torch.zeros_like(zero),atol=1e-8,rtol=0)
    assert torch.isfinite(torch.autograd.grad(foot.cost(p,R).sum(),p)[0]).all()


def test_batched_routes_match_separate_nested_min_including_ties():
    metric=TerrainNavigationDistance(BOX)
    goals=torch.tensor([[1.,0,.5],[0.,0,1.],[-1.,0,0.]])
    prepared=metric.prepare(goals)
    torch.manual_seed(25)
    points=(goals[None]+torch.randn(12,3,3)*.3).requires_grad_()
    points.data[0]=goals
    points.data[1]=torch.tensor([[-1.,0,.5],[0.,0,.5],[1.,0,0.]])
    _,target_exits,ending,tail=prepared
    exits=metric.exits(points)
    prefix=metric.segment_cost(points[...,None,:],exits)
    direct=(metric.segment_cost(exits[..., :,None,:],target_exits[...,None,:,:])+ending[...,None,:]).amin(-1)
    via=(metric.segment_cost(exits[...,None,:],metric.nodes)+tail[...,None,:]).amin(-1)
    dense=(prefix+torch.minimum(direct,via)).amin(-1)
    sparse=metric.distance(points,prepared)
    gd=torch.autograd.grad(dense.square().sum(),points,retain_graph=True)[0]
    gs=torch.autograd.grad(sparse.square().sum(),points)[0]
    torch.testing.assert_close(sparse,dense,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(gs,gd,atol=3e-5,rtol=1e-5)


def test_compiled_query_normalizes_broadcast_views(monkeypatch):
    from types import SimpleNamespace
    from supermimic.handoff.navigation_objective import MaterialNavigationObjective
    metric=TerrainNavigationDistance(BOX)
    eager=metric.segment_cost
    def compile_for_contract(function,**options):
        def checked(start,end):
            assert start.ndim==end.ndim==2
            assert start.is_contiguous() and end.is_contiguous()
            assert start._base is None and end._base is None
            return function(start,end)
        return checked
    monkeypatch.setattr(torch,'compile',compile_for_contract)
    MaterialNavigationObjective.compile_queries(SimpleNamespace(metrics={0:metric}))
    for n,k in [(1,1),(3,4),(17,5),(2,12)]:
        a=torch.randn(n,k,1,3,requires_grad=True)
        b=torch.randn(k,8,3,requires_grad=True)
        expected=eager(a,b);actual=metric.segment_cost(a,b)
        torch.testing.assert_close(actual,expected)
        for got,ref in zip(torch.autograd.grad(actual.sum(),(a,b)),
                           torch.autograd.grad(expected.sum(),(a,b))):
            torch.testing.assert_close(got,ref)


def test_near_surface_tangential_goal_does_not_create_large_normal_gradient():
    metric=TerrainNavigationDistance(BOX)
    goal=torch.tensor([[.25,0.,1.-1e-5]])
    point=torch.tensor([[-.25,0.,1.-1e-5]],requires_grad=True)
    value=metric.distance(point,metric.prepare(goal)).square().sum()
    g=torch.autograd.grad(value,point)[0]
    assert abs(float(g[0,2])) < .02
    assert float(g[0,0]) < -.9


def test_deep_crossing_remains_more_expensive_than_surface_travel():
    metric=TerrainNavigationDistance(BOX)
    a=torch.tensor([[-.5,0.,.5]]);b=torch.tensor([[.5,0.,.5]])
    assert metric.segment_cost(a,b).item()>3.
    surface=metric.segment_cost(a+torch.tensor([0.,0.,.5]),b+torch.tensor([0.,0.,.5]))
    torch.testing.assert_close(surface,torch.ones(1))
