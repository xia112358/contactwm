from types import SimpleNamespace

import torch

from supermimic.contact_soft_arrival import SoftArrival, SoftLanding, soft_arrival_loss, soft_landing_loss
from supermimic.contact_surface import ContinuousArrivalSurface
from somaforge_core.contact_system.atlas import SurfaceAtlas


def sequence(x, gap=None, initial=.1, dt=.002):
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1, 1, 1)
    points = torch.stack((x, x*0, x*0), -1)
    gap = torch.zeros_like(x) if gap is None else torch.as_tensor(gap, dtype=x.dtype).reshape_as(x)
    return soft_arrival_loss(points, gap, points.new_zeros(1, 1, 3),
        points.new_ones(1, 1), points.new_full((1, 1, 1), initial), dt=dt)


def test_first_arrival_not_later_best_point():
    bad_first = sequence([.2]*100 + [0.]*100)
    good_first = sequence([0.]*100 + [.2]*100)
    assert bad_first['loss'][-1].item() > 15.9
    assert good_first['loss'][-1].item() < .001


def test_initial_support_cannot_rearm_after_microscopic_gap():
    baseline = sequence([0.]*500, initial=0.)
    flicker = sequence([0.]*500, gap=[0.]*100+[1e-6]+[0.]*399, initial=0.)
    assert baseline['arrival_mass'][-1].item() == 0
    assert flicker['arrival_mass'][-1].item() < 1e-6
    released = sequence([0.]*500, gap=[0.]*100+[.03]*30+[0.]*370, initial=0.)
    assert released['arrival_mass'][-1].item() > .99


def test_error_is_not_error_of_mean_or_best_point():
    points = torch.tensor([[-.2,0,0],[.2,0,0]], dtype=torch.float64).reshape(1,1,1,2,3)
    gap = torch.zeros(1,1,1,2, dtype=torch.float64)
    def loss(p, g, area):
        return soft_arrival_loss(p,g,p.new_zeros(1,1,3),area,
                                g[0]+.1,dt=.002)['loss']
    area = torch.ones(1,2,dtype=torch.float64)
    torch.testing.assert_close(loss(points,gap,area),torch.full((1,1,1),16.,dtype=torch.float64))
    # Splitting quadrature nodes without changing area cannot change the score.
    torch.testing.assert_close(loss(points,gap,area),
        loss(points.repeat_interleave(2,-2),gap.repeat_interleave(2,-1),area.repeat_interleave(2,-1)/2))
    mixed = points.clone(); mixed[...,0,:] = 0
    assert loss(mixed,gap,area).item() == 8


def test_continuity_across_zero_gap_and_substep_independence():
    losses = [sequence([.1]*50, gap=[d]*50)['loss'][-1] for d in (-1e-7,0,1e-7)]
    torch.testing.assert_close(losses[0],losses[1],atol=1e-10,rtol=0)
    torch.testing.assert_close(losses[2],losses[1],atol=1e-10,rtol=0)
    a = sequence([.2]*30+[0.]*30,dt=.002)
    b = sequence([.2]*60+[0.]*60,dt=.001)
    torch.testing.assert_close(a['loss'][-1],b['loss'][-1])
    torch.testing.assert_close(a['arrival_mass'][-1],b['arrival_mass'][-1])


def test_position_and_gap_gradients_match_finite_difference():
    torch.manual_seed(11)
    points = (torch.rand(4,1,1,3,3,dtype=torch.float64)*.1).requires_grad_()
    gap = (torch.rand(4,1,1,3,dtype=torch.float64)*.01+.003).requires_grad_()
    area = torch.ones(1,3,dtype=torch.float64)
    target = torch.zeros(1,1,3,dtype=torch.float64)
    initial = torch.full((1,1,3),.1,dtype=torch.float64)
    def fn(p,g):
        return soft_arrival_loss(p,g,target,area,initial,dt=.002)['loss'][-1]
    assert torch.autograd.gradcheck(fn,(points,gap),eps=1e-6,atol=1e-5)


def test_dead_slots_stop_accumulating():
    state = SoftArrival(torch.ones(1,1,1),torch.ones(1,1))
    target = torch.zeros(1,1,3)
    state.update(torch.ones(1,1,1,3),torch.zeros(1,1,1),target,dt=.002)
    before = state.loss.clone(),state.survival.clone()
    state.update(torch.zeros(1,1,1,3),torch.ones(1,1,1),target,dt=.002,
                 live=torch.zeros(1,1,dtype=torch.bool))
    assert torch.equal(before[0],state.loss) and torch.equal(before[1],state.survival)


def test_finite_terrain_edges_not_infinite_target_plane():
    nodes = torch.zeros(1,1,3,dtype=torch.float64)
    normals = nodes.clone(); normals[...,2]=1
    atlas = SurfaceAtlas(nodes,normals,torch.ones(1,1),parts=('foot',))
    surface = SimpleNamespace(atlas=atlas,geometry=SimpleNamespace(poses=lambda s:s))
    triangles = torch.tensor([[[0.,0,0],[1.,0,0],[0,1.,0]]],dtype=torch.float64)
    objective = ContinuousArrivalSurface(surface,triangles,dt=.002)
    position = torch.tensor([[[2.,0,0]]],dtype=torch.float64,requires_grad=True)
    rotation = torch.eye(3,dtype=torch.float64).reshape(1,1,3,3)
    _, gap = objective.samples((position,rotation))
    assert gap.item() == 1
    gap.sum().backward()
    assert torch.isfinite(position.grad).all()


def test_dmo_loss_and_gradients_ignore_native_row_switches(monkeypatch):
    from supermimic import contact_dmo
    monkeypatch.setattr(contact_dmo,'observation',lambda s,*args:s[0])
    class Actor(torch.nn.Module):
        hist=1
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,history,condition):
            return self.weight.expand(len(history),29)
    class WM(torch.nn.Module):
        def step(self,s,a,xi):
            return (s[0]+a[:,:3]*.1,*s[1:])
        def next_context(self,s,a,xi,normalized):
            return xi
    T=3
    state=torch.zeros(T,1,71); state[...,3]=1; state[...,0]=.2;state[...,2]=.01
    done=torch.zeros(T,1,dtype=torch.bool);done[-1]=True
    flags=torch.zeros(T,1,6,dtype=torch.bool)
    data=dict(s=state,s1=state.clone(),a=torch.zeros(T,1,29),a_prev=torch.zeros(T,1,29),
        xi=torch.zeros(T,1,33),next_xi=torch.zeros(T,1,33),done=done,
        ep_step=torch.arange(T)[:,None],goal_index=torch.zeros(T,1,dtype=torch.long),
        contact_first=flags.clone(),contact_goal=flags.clone(),known=flags.clone(),
        active=flags.clone(),point=torch.zeros(T,1,6,3),normal_load=torch.full((T,1,6),100.),
        node_load=torch.full((T,1,6,1),100.))
    data['node_position']=state[...,:3][:,:,None,None].expand(T,1,6,1,3).clone()
    target_mask=torch.zeros(T,6,dtype=torch.bool); target_mask[:,0]=True
    reference=state[:,0].clone(); reference[:,3]=torch.cos(torch.tensor(.1));reference[:,6]=torch.sin(torch.tensor(.1))
    reference[:,0]=.3
    clock=SimpleNamespace(event_ends=torch.tensor([T]),segment_starts=torch.tensor([0]),
        segment_stops=torch.tensor([T]),target_masks=target_mask,keep_masks=torch.zeros_like(target_mask),
        data=dict(point=torch.zeros(T,6,3),s1=reference),condition=lambda *args,**kwargs:None)
    nodes=torch.zeros(6,1,3);normals=nodes.clone();normals[...,2]=1
    atlas=SurfaceAtlas(nodes,normals,torch.ones(6,1),parts=tuple(str(i) for i in range(6)))
    geometry=SimpleNamespace(poses=lambda s:(s[0][:,None].expand(-1,6,-1),s[1][:,None].expand(-1,6,-1,-1)))
    objective=ContinuousArrivalSurface(SimpleNamespace(atlas=atlas,geometry=geometry),
        [[[-1.,-1,0],[1.,-1,0],[0,1.,0]]],dt=.02)
    results=[]
    for switched in (False,True):
        actor=Actor()
        if switched:
            data['contact_first'][0,:,0]=True
            data['active'][:]=True; data['known'][:]=True
            data['point'][:]=70 # adversarial native winner; never a loss input
        loss,info=contact_dmo.replay_loss(actor,WM(),clock,geometry,None,lambda x:x,
            torch.zeros(29),data,pose_weight=0.,first_contact_targets=True,
            continuous_arrival=objective,initial_contact_active=torch.zeros(1,6,dtype=torch.bool))
        loss.backward()
        assert info['continuous_landing_process_load_v2']
        assert actor.weight.grad.isfinite() and actor.weight.grad.abs()>0
        results.append((loss.detach(),actor.weight.grad.clone()))
    for a,b in zip(*results):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_native_initial_support_excluded_despite_coarse_nodes():
    state=SoftArrival(torch.full((1,1,1),.008),torch.ones(1,1),
                      initial_supported=torch.ones(1,1,dtype=torch.bool))
    for _ in range(20):
        out=state.update(torch.zeros(1,1,1,3),torch.full((1,1,1),.008),
                         torch.zeros(1,1,3),dt=.002)
    assert out['arrival_mass'].item()==0


def landing(x, gap=None, *, dt=.002, normal_load=None):
    x=torch.as_tensor(x,dtype=torch.float64).reshape(-1,1,1,1)
    points=torch.stack((x,x*0,x*0),-1)
    gaps=torch.zeros_like(x) if gap is None else torch.as_tensor(gap,dtype=x.dtype).reshape_as(x)
    load=(gaps==0).to(x.dtype)*100 if normal_load is None else torch.full_like(gaps,normal_load)
    return soft_landing_loss(points,gaps,points.new_zeros(1,1,3),points.new_ones(1,1),
                             points.new_full((1,1,1),.1),dt=dt,normal_load=load.squeeze(-1),node_load=load)


def test_landing_scores_late_sliding_and_keeps_early_error():
    good=landing([0.]*200)['loss'][-1].item()
    late_bad=landing([0.]*100+[.2]*100)['loss'][-1].item()
    early_bad=landing([.2]*100+[0.]*100)['loss'][-1].item()
    assert late_bad > good+5
    assert early_bad > good+5
    assert late_bad < 17 and early_bad < 17


def test_landing_gap_and_recontact_do_not_erase_history():
    good=landing([0.]*200)
    gap=landing([0.]*200,[0.]*80+[.1]*40+[0.]*80)
    assert gap['loss'][-1] > good['loss'][-1]
    # An unloaded gap cannot accumulate landing history or erase prior history.
    assert (gap['process_time'].diff(dim=0)[90:110]==0).all()
    assert (gap['process_time'].diff(dim=0)[130:150]>0).all()
    assert gap['proximity_time'][-1] < good['proximity_time'][-1]
    assert landing([.2]*80+[0.]*120,[0.]*80+[.1]*40+[0.]*80)['loss'][-1] > gap['loss'][-1]


def test_landing_late_points_still_have_gradients():
    x=torch.full((200,),.1,dtype=torch.float64,requires_grad=True)
    loss=landing(x)['loss'][-1]
    loss.backward()
    assert x.grad.isfinite().all()
    assert x.grad[-20:].min() > .01


def test_landing_no_support_flicker_and_dead_slots():
    state=SoftLanding(torch.zeros(1,1,1),torch.ones(1,1),initial_supported=torch.ones(1,1,dtype=torch.bool))
    p=torch.zeros(1,1,1,3);target=torch.zeros(1,1,3)
    for gap in [0.]*100+[1e-6]+[0.]*100:
        row=state.update(p,torch.full((1,1,1),gap),target,dt=.002,
                         normal_load=torch.full((1,1),100.),node_load=torch.full((1,1,1),100.))
    assert row['arrival_mass'].item() < 1e-6
    assert row['persistence_loss'].item() > .999
    before=state.loss.clone()
    state.update(p+1,torch.ones(1,1,1),target,dt=.002,live=torch.zeros(1,1,dtype=torch.bool),
                 normal_load=torch.zeros(1,1),node_load=torch.zeros(1,1,1))
    assert torch.equal(state.loss,before)
    torch.testing.assert_close(state.position_loss+state.persistence_loss,state.loss)


def test_landing_timestep_and_geometry_continuity():
    a=landing([.2]*30+[0.]*30,dt=.002)['loss'][-1]
    b=landing([.2]*60+[0.]*60,dt=.001)['loss'][-1]
    torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-3)
    a=landing([.1]*100,[0.]*100)['loss'][-1]
    b=landing([.1]*100,[1e-7]*100,normal_load=100)['loss'][-1]
    torch.testing.assert_close(a,b,rtol=0,atol=1e-8)


def test_unloaded_departure_and_hover_never_accumulate_landing():
    state=SoftLanding(torch.full((1,1,2),.0036),torch.ones(1,2),
                      initial_supported=torch.zeros(1,1,dtype=torch.bool))
    target=torch.zeros(1,1,3)
    for i in range(120):
        # Unloaded near-ground oscillation, including zero gap and far swing.
        gap=torch.full((1,1,2),(.0036,0.,.02,.15)[i%4])
        row=state.update(torch.full((1,1,2,3),.5),gap,target,dt=.002,
                         normal_load=torch.zeros(1,1),node_load=torch.zeros(1,1,2))
    assert row['arrival_mass'].item()==0
    assert state.duration.item()==0 and state.process_error.item()==0
    for _ in range(100):
        row=state.update(torch.zeros(1,1,2,3),torch.zeros(1,1,2),target,dt=.002,
                         normal_load=torch.full((1,1),100.),node_load=torch.full((1,1,2),50.))
    assert row['position_loss'].item()==0 # the departure error was not retained
    assert row['arrival_mass'].item()>.99


def test_weak_extra_contact_changes_loss_in_proportion_to_load():
    points=torch.tensor([[[[0.,0,0],[.2,0,0]]]],dtype=torch.float64)
    def evaluate(extra):
        state=SoftLanding(torch.ones(1,1,2,dtype=torch.float64),torch.ones(1,2,dtype=torch.float64))
        for _ in range(200):
            row=state.update(points,torch.zeros(1,1,2,dtype=torch.float64),torch.zeros(1,1,3,dtype=torch.float64),
                dt=.002,normal_load=torch.tensor([[100.+extra]],dtype=torch.float64),
                node_load=torch.tensor([[[100.,extra]]],dtype=torch.float64))
        return row['loss']
    assert (evaluate(.01)-evaluate(0)).abs().item()<.002
    assert (evaluate(100)-evaluate(0)).item()>7


def test_force_redistribution_is_integrated_not_an_instant_point_switch():
    state=SoftLanding(torch.ones(1,1,2),torch.ones(1,2))
    points=torch.tensor([[[[0.,0,0],[.2,0,0]]]])
    gap=torch.zeros(1,1,2);target=torch.zeros(1,1,3);load=torch.full((1,1),100.)
    for _ in range(200):
        state.update(points,gap,target,dt=.002,normal_load=load,node_load=torch.tensor([[[100.,0.]]]))
    before=state.loss.clone()
    state.update(points,gap,target,dt=.002,normal_load=load,node_load=torch.tensor([[[0.,100.]]]))
    assert 0 < (state.loss-before).item() < .1


def test_load_field_and_geometry_gradients_are_finite():
    target=torch.zeros(1,1,3,dtype=torch.float64)
    p=torch.tensor([[[[.02,0,.001],[.08,0,.003]]]],dtype=torch.float64,requires_grad=True)
    field=torch.tensor([[[10.,.1]]],dtype=torch.float64,requires_grad=True)
    def fn(points,loads):
        state=SoftLanding(torch.ones(1,1,2,dtype=torch.float64),torch.ones(1,2,dtype=torch.float64))
        return state.update(points,points[...,2],target,dt=.002,normal_load=loads.sum(-1),node_load=loads)['loss']
    assert torch.autograd.gradcheck(fn,(p,field),eps=1e-6,atol=1e-5)
