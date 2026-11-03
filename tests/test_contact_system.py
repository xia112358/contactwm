"""Physics evidence and smooth surface objectives, independent of a simulator."""
import numpy as np
import pytest
import torch

from somaforge_core.contact_system.atlas import SurfaceAtlas
from somaforge_core.contact_system.evidence import BodyBinding, ContactEvidence, Sampling
from somaforge_core.contact_system.objectives import region_cost, smooth_region_loss, temporal_region_loss


def snapshot():
    # Terrain->foot, foot->terrain, self contact, inactive terrain->foot.
    return dict(count=4,dist=np.array([-.01,-.01,-.01,.1]),includemargin=np.zeros(4),
        type=np.ones(4,int),efc_address=np.zeros((4,4),int),constraint_rows=np.full(4,4),
        worldid=np.zeros(4,int),shape0=np.array([0,1,1,0]),shape1=np.array([1,0,2,1]),
        body0=np.array([-1,0,0,-1]),body1=np.array([0,-1,1,0]),position_w=np.zeros((4,3)),
        frame_w=np.tile(np.eye(3),(4,1,1)),force_on_body1_w=np.array([[0,0,2],[0,0,-3],[0,0,100],[0,0,0.]]),
        torque_on_body1_w=np.zeros((4,3)))


def evidence(s=None,known=None):
    return ContactEvidence(snapshot() if s is None else s,
        sampling=Sampling(.02,.005,substep=3),known=torch.ones(1,1,dtype=torch.bool) if known is None else known)


def binding():
    return BodyBinding(['foot','other_robot_link'],{0:0,1:0},('foot',),{'foot':('foot',)})


def test_evidence_owns_raw_rows_and_excludes_self_without_redefining_activation():
    s=snapshot();e=evidence(s);v=e.terrain_view(binding())
    s['position_w'][:]=99;s['force_on_body1_w'][:]=99
    assert e.count==4 and e.active.tolist()==[True,True,True,False]
    assert v.rows.tolist()==[0,1]
    torch.testing.assert_close(v.reduce()['force_w'],torch.tensor([[[0.,0.,5.]]],dtype=torch.float64))
    assert e.export()['position_w'].max()==0


def test_zero_force_active_constraint_is_preserved_and_unknown_is_separate():
    s=snapshot();s['force_on_body1_w'][:]=0
    assert evidence(s).terrain_view(binding()).reduce()['active'].all()
    e=evidence(s,torch.zeros(1,1,dtype=torch.bool))
    assert e.active.any() and not e.terrain_view(binding()).reduce()['active'].any()
    assert not e.known.any()


def test_all_required_friction_rows_must_be_allocated():
    s=snapshot();s['efc_address'][0,3]=-1
    with pytest.raises(ValueError,match='Unallocated'):evidence(s)
    s=snapshot();s.pop('force_on_body1_w')
    with pytest.raises(ValueError,match='Missing'):evidence(s)


def test_explicit_task_selection_does_not_discard_raw_side_evidence():
    e=evidence();v=e.terrain_view(binding(),eligible=torch.tensor([False,True,False,False]))
    assert v.rows.tolist()==[1] and e.count==4 and e.active.sum()==3


def test_single_snapshot_cannot_claim_substep_complete_sampling():
    with pytest.raises(ValueError,match='complete'):Sampling(.02,.005,complete_interval=True)
    assert not evidence().sampling.pose_contact_same_state


def test_area_normalized_surface_loss_is_invariant_to_node_refinement():
    p=torch.tensor([[[.01,0,0],[.04,0,0]]],dtype=torch.float64)
    target=torch.zeros(1,3,dtype=torch.float64);normal=target.new_tensor([[0.,0.,1.]])
    a=p.new_tensor([[1.,3.]])
    old=smooth_region_loss(p,target,normal,a)['loss']
    new=smooth_region_loss(p.repeat_interleave(3,1),target,normal,a.repeat_interleave(3,1)/3)['loss']
    torch.testing.assert_close(old,new)


def test_opposite_wrong_points_cannot_cancel_at_the_target():
    p=torch.tensor([[[-.1,0,0],[.1,0,0]]])
    target=torch.zeros(1,3);normal=torch.tensor([[0.,0.,1.]])
    result=smooth_region_loss(p,target,normal,torch.ones(1,2))
    torch.testing.assert_close(result['loss'],torch.tensor([4.]))


def test_anisotropic_selection_uses_the_scoring_metric():
    p=torch.tensor([[[.05,0,0],[0,0,.049]]],dtype=torch.float64)
    result=smooth_region_loss(p,torch.zeros(1,3,dtype=p.dtype),p.new_tensor([[0.,0.,1.]]),
                              p.new_ones(1,2),normal_weight=4.,temperature=.001)
    assert result['weights'][0,0]>.999
    assert 1. <= result['loss'].item()<1.001


def test_surface_cost_and_gradient_are_continuous_through_a_winner_switch():
    x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    base=torch.tensor([[[-.01,0,0],[.01,0,0]]],dtype=x.dtype)
    def loss(value):
        p=base+torch.stack((value,value*0,value*0))
        return smooth_region_loss(p,torch.zeros(1,3,dtype=x.dtype),x.new_tensor([[0.,0.,1.]]),
                                  x.new_ones(1,2),temperature=.1)['loss'].sum()
    torch.autograd.gradcheck(loss,(x,),eps=1e-6,atol=1e-5)
    grads=[]
    for value in (-1e-7,1e-7):
        y=x.detach().new_tensor(value,requires_grad=True)
        grads.append(torch.autograd.grad(loss(y),y)[0])
    assert abs(grads[1]-grads[0])<1e-3


def test_bounded_region_penalizes_normal_gap_and_outside_tangent_distance():
    p=torch.tensor([[[.01,0,.01],[.04,0,0.]]])
    cost,normal,tangent=region_cost(p,torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),halfwidth=.02)
    torch.testing.assert_close(tangent,torch.tensor([[0.,.16]]))
    assert normal[0,0]>0 and cost[0,1]>0


def test_atlas_projection_preserves_force_and_carries_moment_residual():
    atlas=SurfaceAtlas(torch.tensor([[[-.1,0,0],[.1,0,0]]],dtype=torch.float64),
                       torch.tensor([[[0.,0.,1.],[0.,0.,1.]]],dtype=torch.float64),
                       torch.ones(1,2,dtype=torch.float64),parts=('foot',))
    s=snapshot();s['position_w'][0]=[.05,0,0];s['position_w'][1]=[-.02,0,0]
    v=evidence(s).terrain_view(binding())
    p=torch.zeros(1,1,3,dtype=torch.float64);R=torch.eye(3,dtype=p.dtype).reshape(1,1,3,3)
    projected=atlas.project(v,p,R,bandwidth=.03)
    torch.testing.assert_close(projected['force_w'].sum(-2),v.reduce()['force_w'])
    moment=torch.linalg.cross(atlas.world(p,R)-p[...,None,:],projected['force_w']).sum(-2)
    actual=torch.linalg.cross(v.position_w,v.force_w).sum(0).reshape(1,1,3)
    torch.testing.assert_close(moment+projected['moment_residual_w'],actual)


def test_normal_load_projection_uses_native_frame_not_net_vertical_force():
    atlas=SurfaceAtlas(torch.tensor([[[-.1,0,0],[.1,0,0]]],dtype=torch.float64),
        torch.tensor([[[0.,0.,1.],[0.,0.,1.]]],dtype=torch.float64),
        torch.ones(1,2,dtype=torch.float64),parts=('foot',))
    s=snapshot()
    s['frame_w'][0]=[[0,1,0],[1,0,0],[0,0,-1]]
    s['frame_w'][1]=[[0,-1,0],[1,0,0],[0,0,1]]
    s['force_on_body1_w'][0]=[0,2,10]
    s['force_on_body1_w'][1]=[0,-4,-20]
    v=evidence(s).terrain_view(binding());p=torch.zeros(1,1,3,dtype=torch.float64)
    R=torch.eye(3,dtype=p.dtype).reshape(1,1,3,3)
    projected=atlas.project(v,p,R,bandwidth=.01)
    assert projected['normal_load'].item()==6
    torch.testing.assert_close(projected['node_load'].sum(-1),projected['normal_load'])
    assert v.reduce()['force_w'][0,0,2]==30
    torch.testing.assert_close(projected['node_position'],atlas.world(p,R))


def test_fixed_atlas_retains_small_spheres_and_known_area():
    triangle=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
    sphere=np.array([[0,0,0,.001]])
    atlas=SurfaceAtlas.from_surfaces([triangle],[sphere],parts=('foot',),samples_per_part=32)
    other=SurfaceAtlas.from_surfaces([triangle],[sphere],parts=('foot',),samples_per_part=32)
    assert atlas.fingerprint==other.fingerprint
    assert atlas.points_local.shape==(1,32,3)
    np.testing.assert_allclose(atlas.areas.sum().item(),.5+4*np.pi*.001**2,rtol=1e-6)
    assert (atlas.areas[0]<1e-4).sum()>=4


def test_temporal_cost_never_averages_positions_or_unknown_frames():
    cost=torch.tensor([[4.,float('nan')],[4.,1.]],requires_grad=True)
    valid=torch.tensor([[True,False],[True,True]])
    result=temporal_region_loss(cost,valid,dt=.005)
    torch.testing.assert_close(result,torch.tensor([4.,1.]))
    result.sum().backward()
    assert torch.isfinite(cost.grad).all() and cost.grad[0,1]==0
    with pytest.raises(ValueError,match='observed'):temporal_region_loss(cost,torch.zeros_like(valid),dt=.005)


def test_native_adapter_maps_solver_rows_and_reads_local_points_without_physics(monkeypatch):
    import contextlib
    import sys
    from types import SimpleNamespace as NS
    from somaforge_core.contact_system.evidence import solver_snapshot
    c=NS(dist=torch.tensor([-.01]),includemargin=torch.zeros(1),type=torch.ones(1,dtype=torch.int32),
        efc_address=torch.zeros(1,4,dtype=torch.int32),worldid=torch.zeros(1,dtype=torch.int32),
        pos=torch.tensor([[1.,2.,3.]]),frame=torch.eye(3)[None],
        geom=torch.tensor([[0,1]],dtype=torch.int32),dim=torch.tensor([3],dtype=torch.int32))
    d=NS(contact=c,nacon=torch.tensor([1]),xpos=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]]),
         xmat=torch.eye(3).expand(1,2,3,3).clone())
    m=NS(opt=NS(cone=0),geom_bodyid=torch.tensor([0,1]))
    solver=NS(mjw_data=d,mjw_model=m,mjc_geom_to_newton_shape=torch.tensor([[1,0]]))
    model=NS(device='cpu',shape_body=torch.tensor([7,-1]))
    warp=NS(to_torch=lambda x:x,from_torch=lambda x:x,spatial_vector=object(),
            zeros=lambda n,**kw:torch.zeros(n,6),ScopedDevice=lambda d:contextlib.nullcontext())
    def decode(m,d,ids,world,wrench):
        assert world and ids.tolist()==[0]
        wrench[:]=torch.tensor([[0.,0.,10.,0.,0.,2.]])
    monkeypatch.setitem(sys.modules,'warp',warp)
    monkeypatch.setitem(sys.modules,'mujoco_warp',NS(contact_force=decode))
    snap=solver_snapshot(solver,model)
    assert snap['shape0'].tolist()==[1] and snap['body1'].tolist()==[7]
    torch.testing.assert_close(snap['application_point1_local'],torch.tensor([[0.,2.,3.]]))
    torch.testing.assert_close(snap['torque_on_body1_w'],torch.tensor([[0.,0.,2.]]))


def test_diagnostic_variation_does_not_cross_unknown_or_reset_rows():
    from somaforge_core.contact_system.diagnostics import variation,witness_motion
    r=variation(torch.tensor([0.,1.,100.,102.]),torch.tensor([True,True,False,True]))
    assert r['pairs']==1 and r['maximum']==1
    p=torch.zeros(2,1,3);p[1,0,0]=.1
    point=p.clone();rotation=torch.eye(3).expand(2,1,3,3)
    report=witness_motion(point,p,rotation,torch.ones(2,1,dtype=torch.bool))
    assert report['reselection_residual']['mean_m']==0


def test_masked_surface_nodes_cannot_contribute_or_poison_gradients():
    p=torch.tensor([[[.01,0,0],[float('nan'),0,0]]],dtype=torch.float64,requires_grad=True)
    result=smooth_region_loss(p,torch.zeros(1,3,dtype=p.dtype),p.new_tensor([[0.,0.,1.]]),
        p.new_tensor([[1.,0.]]),mask=torch.tensor([[True,False]]))
    result['loss'].sum().backward()
    assert p.grad[0,1].abs().sum()==0
    assert torch.isfinite(result['loss']).all() and torch.isfinite(result['normal']).all()


def test_opposing_contact_forces_retain_their_moment_and_zero_load_position_evidence():
    atlas=SurfaceAtlas(torch.tensor([[[-.1,0,0],[.1,0,0]]],dtype=torch.float64),
        torch.tensor([[[0.,0.,1.],[0.,0.,1.]]],dtype=torch.float64),
        torch.ones(1,2,dtype=torch.float64),parts=('foot',))
    s=snapshot();s['position_w'][0]=[-.1,0,0];s['position_w'][1]=[.1,0,0]
    s['force_on_body1_w'][0]=[0,1,0];s['force_on_body1_w'][1]=[0,1,0]
    v=evidence(s).terrain_view(binding())
    p=torch.zeros(1,1,3,dtype=torch.float64);R=torch.eye(3,dtype=p.dtype).reshape(1,1,3,3)
    projected=atlas.project(v,p,R,bandwidth=.02)
    torch.testing.assert_close(projected['force_w'].sum(-2),torch.zeros_like(p))
    moment=torch.linalg.cross(atlas.world(p,R),projected['force_w']).sum(-2)+projected['moment_residual_w']
    torch.testing.assert_close(moment,torch.tensor([[[0.,0.,-.2]]],dtype=p.dtype))
    s['force_on_body1_w'][:]=0
    projected=atlas.project(evidence(s).terrain_view(binding()),p,R,bandwidth=.02)
    assert projected['active'].all() and projected['evidence_mass'].sum()==2 and not projected['force_w'].any()


def test_wbt_adapter_exports_owned_evidence_with_actual_control_and_physics_timing():
    from types import SimpleNamespace as NS
    from holosoma.utils.wbt_contacts import MeasuredContacts
    reader=MeasuredContacts.__new__(MeasuredContacts)
    from somaforge_core.collision_profiles import contact_body_names
    reader.body_names=contact_body_names("legacy_v1")
    reader.robot_asset={"collision_profile":"legacy_v1"}
    reader.model=NS(body_label=['left_ankle_roll_link','right_ankle_roll_link'])
    reader.binding=dict(body_env={0:0,1:0},model_fingerprint='fixture-model')
    reader.env=NS(dt=.02,device='cpu',episode_length_buf=torch.tensor([7]),
        simulator=NS(simulator_config=NS(sim=NS(fps=200,substeps=1,control_decimation=4))))
    s=snapshot()
    reader.read=lambda:dict(snapshot=s,known=torch.ones(1,6,dtype=torch.bool))
    e=reader.read_evidence()
    s['position_w'][:]=99
    assert e.sampling.physics_dt==.005 and e.sampling.substep==3
    assert e.control_step.tolist()==[7] and not e.rows['position_w'].any()
    assert e.provenance['model_fingerprint']=='fixture-model'
