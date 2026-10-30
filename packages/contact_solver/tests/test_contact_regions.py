import pytest
import torch

from contact_solver.contact_regions import ContactRegions, unified_region_objective
from generator.full1000_position_predictor import Full1000PositionPredictor
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics
from generator.next_interaction_heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
from contact_solver.device_contact_objective import DeviceWitnessRows as NativeWitnessRows
from contact_solver.solid_witness_loss import SolidWitnessRows


def DeviceWitnessRows(fk, q, observed):
    """Synthetic interval tests supply a separate, upward plane distance.

    Native activation/normal errors remain independently testable. Production
    receives complete solids from the realized scene, with no such fallback.
    """
    rows = NativeWitnessRows(fk, q, observed)
    p = observed['pairs']
    pair = {key: p[key].clone() for key in ('sample', 'body_link0', 'body_link1', 'full_kind')}
    pair.update(dist=p['dist'].double(), point0_w=p['geometry_point0_w'].double(),
                point1_w=p['geometry_point1_w'].double(),
                normal_w=q.new_tensor([0., 0., 1.]).double().expand(len(p['dist']), 3))
    rows.solid = SolidWitnessRows(q, pair, fk.link_poses(q.double(), observed['link_names']))
    return rows


def test_regions_cover_canonical_geometry_and_mask_padded_regions():
    geometry = ContactRegions(CanonicalG1ForwardKinematics())
    assert geometry.valid.sum(-1).tolist() == [4, 4, 2, 2, 3, 3]
    for part in range(6):
        for region in range(len(geometry.names[part])):
            cloud = getattr(geometry, f'cloud_{part}_{region}')
            assert (geometry.classify_local(torch.full((len(cloud),), part), cloud) == region).all()
            assert ((cloud-geometry.landmark[part, region]).abs().sum(-1) == 0).any()
    p = torch.zeros(2, 6, 3); rot = torch.eye(3).expand(2, 6, 3, 3)
    h = torch.zeros(2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS)
    result = geometry(p, rot, h).reshape(2, 6, 4, 8)
    assert torch.isfinite(result).all()
    assert (result[:, ~geometry.valid] == 0).all()
    p[..., 0] += 100
    assert geometry(p, rot, h).reshape(2, 6, 4, 8)[..., -2:].count_nonzero() == 0


def test_regional_plan_warm_start_preserves_pose_and_has_nonempty_valid_subsets():
    torch.manual_seed(4)
    base = Full1000PositionPredictor(24, 1, 8).eval()
    model = Full1000PositionPredictor(24, 1, 8, region_plan=True, unified_contact=True).eval()
    missing = model.load_state_dict(base.state_dict(), strict=False)
    assert not missing.unexpected_keys and all(n.startswith(('region_', 'execution_geometry_encoder.')) for n in missing.missing_keys)
    q = torch.zeros(2, 36); q[:, 2] = .8; q[:, 3] = 1
    obs = dict(current_q=q, current_contact=torch.zeros(2, 6, dtype=torch.bool),
        current_anchor=torch.zeros(2, 6, 3), heightmap=torch.full((2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8))
    a, b = base(**obs), model(**obs)
    torch.testing.assert_close(a.qpos, b.qpos, rtol=0, atol=0)
    torch.testing.assert_close(a.role_logits, b.role_logits, rtol=0, atol=0)
    assert torch.equal(b.planned_regions.any(-1), b.conditioned_contact)
    assert not (b.planned_regions & ~model.region_geometry.valid).any()
    b.qpos.square().sum().backward()
    assert model.region_plan_encoder.weight.grad.abs().sum() > 0
    assert model.region_geometry_encoder[-1].weight.grad is None
    assert model.execution_geometry_encoder.weight.grad.abs().sum() > 0


class TwoFeet:
    def link_poses(self, q, names):
        p = torch.stack([q[:, :3] if n.startswith('left') else q[:, 7:10] for n in names], 1)
        return p, torch.eye(3).to(q).expand(len(q), len(names), 3, 3)


class TestRegions:
    valid = torch.ones(6, 4, dtype=torch.bool)
    def witness_regions(self, fk, q, sample, part, points):
        return (points[:, 0] > 0).long()
    def missing_region_distance(self, fk, q, surface, scene, margin):
        return q.sum(-1)[:, None, None].expand(-1, 6, 4)*0
    def actual_mask(self, fk, rows, surface):
        from contact_solver.device_contact_objective import group_any
        p = rows.pair
        group = (p['sample']*6+p['part'])*4+self.witness_regions(fk, rows.q, p['sample'], p['part'], rows.points[:, 1])
        same_surface=p['primary_surface']==surface[p['sample'],p['part']]
        return group_any(group, p['eligible'] & same_surface, len(rows)*24).reshape(len(rows), 6, 4)


def witness_fixture(depths):
    q = torch.zeros(1, 36, requires_grad=True)
    point = torch.tensor([[-.1, 0., 0.], [.1, 0., 0.]])
    p = dict(sample=torch.tensor([0, 0]), part=torch.tensor([0, 1]), task_pair=torch.ones(2, dtype=torch.bool),
        dist=torch.tensor(depths), geometry_point0_w=point, geometry_point1_w=point,
        normal_w=torch.tensor([[0., 0., 1.]]*2), body_link0=torch.tensor([-1, -1]), body_link1=torch.tensor([0, 1]),
        full_kind=torch.zeros(2, dtype=torch.long), primary_surface=torch.zeros(2, dtype=torch.long),
        upward=torch.ones(2, dtype=torch.bool), active=torch.tensor(depths)<.02,
        constraint_allocated=torch.tensor(depths)<.02, includemargin=torch.full((2,), .02), eligible=torch.tensor(depths)<.02)
    observed=dict(schema='newton_device_witness_batch_v1', pairs=p,
        link_names=('left_ankle_roll_link', 'right_ankle_roll_link'), configured_margin=torch.tensor([.02]))
    model=type('Model', (), {'fk':TwoFeet(), 'region_geometry':TestRegions()})()
    active=torch.tensor([[True, True, False, False, False, False]])
    regions=torch.zeros(1, 6, 4, dtype=torch.bool);regions[0,0,0]=True;regions[0,1,1]=True
    return q, observed, model, active, regions


def interval_query_fixture(rows, regions):
    from contact_solver.contact_surface_interval import SurfaceIntervalQuery
    q=rows.q;link=torch.full((len(q),6,4),-1,dtype=torch.long)
    point=q.new_zeros(len(q),6,4,3);normal=torch.zeros_like(point);gap=q.new_zeros(len(q),6,4)
    unique=torch.zeros(len(q),6,4,dtype=torch.bool)
    for i,part in enumerate((0,1)):
        region=int(regions[0,part].nonzero()[0]);link[0,part,region]=i
        point[0,part,region]=rows.solid.material_local[i,1]
        normal[0,part,region]=rows.solid.normal_local[i]
        gap[0,part,region]=rows.solid.distances[i]
        unique[0,part,region]=True
    return SurfaceIntervalQuery(tuple(rows.observed['link_names']),link,point,normal,gap,unique,unique.clone())


def test_unified_contact_remains_repulsive_after_activation_and_both_feet_receive_gradient():
    q, obs, model, active, regions = witness_fixture([-.02, -.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)
    assert metrics['region_plan_realized'].item() == 1
    grad=torch.autograd.grad(loss.sum(),q)[0]
    assert grad[0,2]<0 and grad[0,9]<0
    duplicate=dict(obs,pairs={k:v.repeat_interleave(3,0) for k,v in obs['pairs'].items()})
    other,_=unified_region_objective(model,DeviceWitnessRows(model.fk,q,duplicate),active,torch.zeros(1,6,dtype=torch.long),{},regions)
    torch.testing.assert_close(loss,other)


@pytest.mark.parametrize('penalty',['log1p','huber'])
def test_target_recovery_replaces_generic_approach_without_certifying_contact(penalty):
    q,obs,model,active,regions=witness_fixture([.10,.10])
    rows=DeviceWitnessRows(model.fk,q,obs)
    generic=torch.tensor(1.,requires_grad=True)
    surface_interval=generic.square().expand(1,6,4)
    recovery=(q[:,:3]-q.new_tensor([.1,.2,.3])).square().sum(-1)[:,None,None].expand(1,6,4)
    loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=surface_interval,penetration_penalty=penalty,
        target_recovery=recovery,target_recovery_mask=active)
    grad,old_grad=torch.autograd.grad(loss.sum(),(q,generic))
    assert bool((grad[0,:3]<0).all())
    assert old_grad.item()==0  # No second approach gradient remains.
    assert metrics['region_plan_realized'].item()==0
    assert metrics['region_missing_count'].item()==2
    torch.testing.assert_close(loss,metrics['unified_attraction_component']+metrics['unified_separation_component'])


def test_target_recovery_cannot_override_actual_contact_or_unknown_geometry():
    q,obs,model,active,regions=witness_fixture([0.,0.])
    rows=DeviceWitnessRows(model.fk,q,obs)
    kwargs=dict(target_recovery=q.new_zeros(1,6,4),target_recovery_mask=active)
    with pytest.raises(ValueError,match='actual same-part/surface'):
        unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,**kwargs)
    q,obs,model,active,regions=witness_fixture([.10,.10])
    rows=DeviceWitnessRows(model.fk,q,obs)
    kwargs['target_recovery'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Undefined target recovery'):
        unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,**kwargs)


@pytest.mark.parametrize('penalty',['log1p','huber'])
def test_target_recovery_retains_independent_complete_solid_repulsion(penalty):
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    # Existing contacts are on another surface, so the intended relations
    # are actually missing. Their physical penetration remains independent.
    obs['pairs']['primary_surface'][:]=1
    rows=DeviceWitnessRows(model.fk,q,obs)
    recovery=(q[:,:1]-.1).square()[:,:,None].expand(1,6,4)
    loss,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        penetration_penalty=penalty,target_recovery=recovery,target_recovery_mask=active)
    grad=torch.autograd.grad(loss.sum(),q)[0]
    assert grad[0,0]<0  # Target position recovery.
    assert grad[0,2]<0 and grad[0,9]<0  # Both solid penetration gradients survive.


@pytest.mark.parametrize('explicit_regions',[False,True])
def test_one_target_point_does_not_pull_heel_and_toe_to_the_same_location(explicit_regions):
    q,obs,model,active,regions=witness_fixture([.02,.02])
    obs['pairs']['part'][:]=0;obs['pairs']['body_link1'][:]=0
    active[:,1]=False;regions[:,1]=False;regions[0,0,1]=True
    rows=DeviceWitnessRows(model.fk,q,obs)
    recovery=q.new_zeros(1,6,4)
    recovery[0,0]=(q[0,0]+.3).square().expand(4)
    recovery[0,0,0]=(q[0,0]-.01).square()
    remaining_surface=torch.tensor(.0001,requires_grad=True)
    intervals=q.new_zeros(1,6,4);intervals[0,0,1]=remaining_surface
    loss,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions if explicit_regions else None,
        surface_interval=intervals,target_recovery=recovery,target_recovery_mask=active)
    pose_grad,surface_grad=torch.autograd.grad(loss.sum(),(q,remaining_surface))
    assert pose_grad[0,0]<0  # Only the nearest region follows the representative point.
    if explicit_regions:
        assert surface_grad.item()>0  # Other required regions still learn surface contact.
    else:
        assert surface_grad.item()==0  # Coarse plans require only the closest region.


def test_unified_contact_attracts_above_band_and_allows_unselected_regions_to_lift():
    q,obs,model,active,regions=witness_fixture([.025,.0005])
    loss,_=unified_region_objective(model,DeviceWitnessRows(model.fk,q,obs),active,torch.zeros(1,6,dtype=torch.long),{},regions)
    grad=torch.autograd.grad(loss.sum(),q)[0]
    assert grad[0,2]>0 and grad[0,9]==0
    # The rear-only plan does not demand every regional point to touch.
    assert regions.sum()==2 and loss>0


def test_region_truth_requires_selected_allocated_witnesses():
    geo=ContactRegions(CanonicalG1ForwardKinematics())
    q=torch.zeros(1,36);q[:,3]=1
    fk=CanonicalG1ForwardKinematics()
    p,r=fk.link_poses(q,('left_ankle_roll_link',))
    point=(p[0,0]+r[0,0]@geo.landmark[0,2]).tolist()
    required=torch.zeros(6,4,dtype=torch.bool);required[0,2]=True
    assert not geo.audit_pairs(fk,q,[],[0]*6,required)['realized']
    pair=dict(part=0,surface=0,allocated=True,constraint_active=True,position_w=point)
    assert geo.audit_pairs(fk,q,[pair],[0]*6,required)['realized']


def test_front_and_rear_of_same_foot_are_independent_and_extra_collision_cannot_reduce_loss():
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    obs['pairs']['part'][:]=0;obs['pairs']['body_link1'][:]=0
    active[:,1]=False;regions[:,1]=False
    # Rear is planned; the forefoot still has to avoid penetration.
    loss,_=unified_region_objective(model,DeviceWitnessRows(model.fk,q,obs),active,torch.zeros(1,6,dtype=torch.long),{},regions)
    rear, front = torch.tensor(17.).log(), torch.tensor(5.).log()
    torch.testing.assert_close(loss,(rear+(rear+front)/24).reshape(1))
    lifted=dict(obs,pairs={k:v.clone() for k,v in obs['pairs'].items()})
    lifted['pairs']['dist'][1]=.03;lifted['pairs']['eligible'][1]=False
    less,_=unified_region_objective(model,DeviceWitnessRows(model.fk,q,lifted),active,torch.zeros(1,6,dtype=torch.long),{},regions)
    assert loss>less


def test_unknown_native_normal_requires_audit_but_retains_verified_solid_gradients(tmp_path):
    import pytest
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    obs['pairs']['task_pair'][1]=False
    obs['pairs']['normal_w'][1]=float('nan')
    rows=DeviceWitnessRows(model.fk,q,obs)
    with pytest.raises(ValueError,match='explicit audit'):
        unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)
    loss,m=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,audit_path=tmp_path/'audit.jsonl')
    assert m['region_gradient_valid'].item()==1 and m['region_invalid_penetrating_normals'].item()==1
    loss.sum().backward()
    assert loss.item()>0 and q.grad[0,2]<0 and q.grad[0,9]<0
    assert (tmp_path/'audit.jsonl').exists()


def test_undefined_complete_geometry_still_raises_instead_of_inventing_gradient():
    import pytest
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    rows.solid.distances = rows.solid.distances*torch.nan
    with pytest.raises(ValueError,match='Undefined complete-solid'):
        unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)


def test_zero_shape_interval_cannot_erase_native_gap_or_create_contact():
    q,obs,model,active,regions=witness_fixture([.05,.05])
    rows=DeviceWitnessRows(model.fk,q,obs)
    loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=q.sum()*0+torch.zeros(1,6,4))
    assert loss.item()>0
    grad=torch.autograd.grad(loss.sum(),q)[0]
    assert grad[0,2]>0 and grad[0,9]>0
    assert metrics['region_plan_realized'].item()==0
    assert metrics['region_missing_count'].item()==2


def test_shape_interval_does_not_erase_complete_terrain_or_self_clearance():
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    rows.solid.pair['full_kind'][1]=1
    loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=q.sum()*0+torch.zeros(1,6,4))
    grad=torch.autograd.grad(loss.sum(),q)[0]
    assert grad[0,2]<0 and grad[0,9]<0
    assert metrics['solid_terrain_penetration_cm'].item()>0
    assert metrics['solid_self_penetration_cm'].item()>0


def test_undefined_shape_interval_requires_explicit_error_without_proxy_fallback():
    import pytest
    q,obs,model,active,regions=witness_fixture([.01,.01])
    cost=torch.zeros(1,6,4);cost[0,0,0]=torch.inf
    with pytest.raises(ValueError,match='no geometric fallback'):
        unified_region_objective(model,DeviceWitnessRows(model.fk,q,obs),active,
            torch.zeros(1,6,dtype=torch.long),{},regions,surface_interval=cost)


@pytest.mark.parametrize('penalty', ['log1p', 'huber'])
def test_shared_interval_merges_duplicate_recovery_without_erasing_solids(penalty):
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    base,original=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        penetration_penalty=penalty)
    lower=q.new_zeros(1,6,4)
    for i,part in enumerate((0,1)):
        region=int(regions[0,part].nonzero()[0])
        lower[0,part,region]=(-rows.solid.distances[i]).relu().square()
    merged,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        penetration_penalty=penalty,
        surface_interval=SurfaceIntervalBounds(lower,torch.zeros_like(lower),
            solid_target_mask=torch.tensor([[False,True],[False,True]]),query=interval_query_fixture(rows,regions)))
    torch.testing.assert_close(merged,base)
    torch.testing.assert_close(torch.autograd.grad(merged.sum(),q,retain_graph=True)[0],
        torch.autograd.grad(base.sum(),q)[0])
    torch.testing.assert_close(metrics['unified_physical_clearance_component'],torch.zeros(1))
    torch.testing.assert_close(metrics['solid_terrain_penetration_cm'],original['solid_terrain_penetration_cm'])
    torch.testing.assert_close(metrics['solid_self_penetration_cm'],original['solid_self_penetration_cm'])


def test_huber_unified_loss_keeps_deep_repulsion_and_contact_approach():
    gradients = []
    for depth in (.005, .03, .1):
        q, obs, model, active, regions = witness_fixture([-depth, .03])
        loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
            active, torch.zeros(1, 6, dtype=torch.long), {}, regions,
            penetration_penalty='huber')
        gradient, = torch.autograd.grad(loss.sum(), q)
        assert torch.isfinite(gradient).all()
        # Root z lifts the penetrated foot; right relative z approaches its face.
        assert gradient[0, 9] > 0
        gradients.append(gradient[0, 2]-gradient[0, 9])
        torch.testing.assert_close(loss, metrics['unified_attraction_component']
                                   + metrics['unified_separation_component'])
    assert all(g < 0 for g in gradients)
    # Once the penetration group dominates, its correction must not decay.
    torch.testing.assert_close(gradients[1], gradients[2])


def test_shared_recovery_bound_cannot_suppress_a_larger_physical_lower_bound():
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    base,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)
    field=q.sum()*0+q.new_zeros(1,6,4)
    shared,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=SurfaceIntervalBounds(field,field))
    torch.testing.assert_close(shared,base)
    grad=torch.autograd.grad(shared.sum(),q)[0]
    assert grad[0,2]<0 and grad[0,9]<0


def test_target_recovery_retains_independent_self_collision_gradient():
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,observed,model,active,regions=witness_fixture([-.1,-.1])
    regions[0,1]=False;regions[0,1,0]=True
    observed['pairs']['geometry_point0_w'][:,2]=.1
    rows=DeviceWitnessRows(model.fk,q,observed)
    pair=dict(sample=torch.tensor([0]),body_link0=torch.tensor([0]),body_link1=torch.tensor([1]),
        full_kind=torch.tensor([1]),dist=torch.tensor([-.02],dtype=torch.float64),
        point0_w=torch.tensor([[-.1,0.,0.]],dtype=torch.float64),
        point1_w=torch.tensor([[-.12,0.,0.]],dtype=torch.float64),
        normal_w=torch.tensor([[1.,0.,0.]],dtype=torch.float64))
    rows.solid=SolidWitnessRows(q,pair,model.fk.link_poses(q.double(),observed['link_names']))
    lower=q.new_zeros(1,6,4)
    lower[0,0,0]=(.1-q[0,2]).square();lower[0,1,0]=(.1-q[0,9]).square()
    loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=SurfaceIntervalBounds(lower,torch.zeros_like(lower)))
    grad=torch.autograd.grad(loss.sum(),q)[0]
    # The larger z recovery must not erase the smaller x self-collision repair.
    assert metrics['solid_self_penetration_cm'].item()>0
    assert grad[0,0]>0 and grad[0,7]<0


def test_target_interval_binding_cannot_claim_a_self_collision():
    import pytest
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs)
    rows.solid.pair['full_kind'][1]=1
    lower=q.new_zeros(1,6,4)
    with pytest.raises(ValueError,match='Self collision cannot be deduplicated'):
        unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
            surface_interval=SurfaceIntervalBounds(lower,lower,
                solid_target_mask=torch.tensor([[False,True],[False,True]])))


def test_shared_bounds_preserve_original_material_scalar_and_gradient_without_collision():
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([0.,0.])
    rows=DeviceWitnessRows(model.fk,q,obs)
    lower=q.new_zeros(1,6,4);upper=torch.zeros_like(lower)
    lower[0,0,0]=(.03-q[0,2]).square()
    upper[0,0,0]=(.04+q[0,0]).square()
    scale=.25*obs['configured_margin'][:,None,None]
    model.region_geometry.missing_region_distance=lambda *args:(lower+upper)/scale.square()
    control,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)
    shared,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=SurfaceIntervalBounds(lower,upper))
    torch.testing.assert_close(shared,control)
    torch.testing.assert_close(torch.autograd.grad(shared.sum(),q,retain_graph=True)[0],
        torch.autograd.grad(control.sum(),q)[0])


def test_different_witness_bounds_keep_independent_penalties_and_gradients():
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([.03,-.02])
    class IndependentFootLinks:
        def link_poses(self,q,names):
            return torch.stack([q[:,:3],q[:,7:10]],1),torch.eye(3).to(q).expand(len(q),2,3,3)
    model.fk=IndependentFootLinks()
    obs['link_names']=('left_ankle_roll_link','left_ankle_roll_sphere_1_link')
    obs['pairs']['geometry_point0_w'][:,0]=-.1
    obs['pairs']['geometry_point1_w'][:,0]=-.1
    obs['pairs']['part'][:]=0;obs['pairs']['upward'][1]=False
    active[:,1]=False;regions[:,1]=False
    zero=q.sum()*0+q.new_zeros(1,6,4)
    loss,metrics=unified_region_objective(model,DeviceWitnessRows(model.fk,q,obs),active,
        torch.zeros(1,6,dtype=torch.long),{},regions,surface_interval=SurfaceIntervalBounds(zero,zero,
            solid_target_mask=torch.tensor([[False,True],[False,True]])))
    torch.testing.assert_close(loss,q.new_tensor([(1+1/24)*(torch.log(torch.tensor(17.))+torch.log(torch.tensor(5.)))]))
    grad=torch.autograd.grad(loss.sum(),q,retain_graph=True)[0]
    assert grad[0,2]>0 and grad[0,9]<0
    torch.testing.assert_close(grad[0,2],q.new_tensor((1+1/24)/5*2*.01/.005**2))
    torch.testing.assert_close(grad[0,9],q.new_tensor(-(1+1/24)/17*2*.02/.005**2))
    pieces=metrics['unified_attraction_component']+metrics['unified_separation_component']
    torch.testing.assert_close(pieces,loss)
    torch.testing.assert_close(torch.autograd.grad(pieces.sum(),q)[0],grad)


def test_whole_native_and_material_queries_cannot_supply_mixed_bounds():
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds
    q,obs,model,active,regions=witness_fixture([.03,0.])
    rows=DeviceWitnessRows(model.fk,q,obs)
    lower=q.new_zeros(1,6,4);lower[0,0,0]=(.04-q[0,2]).square()
    scale=.25*obs['configured_margin'][:,None,None]
    model.region_geometry.missing_region_distance=lambda *args:lower/scale.square()
    control,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions)
    shared,_=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
        surface_interval=SurfaceIntervalBounds(lower,torch.zeros_like(lower)))
    torch.testing.assert_close(shared,control)
    torch.testing.assert_close(torch.autograd.grad(shared.sum(),q,retain_graph=True)[0],torch.autograd.grad(control.sum(),q)[0])


def test_entity_region_match_without_same_material_query_cannot_erase_clearance():
    import dataclasses
    from contact_solver.contact_surface_interval import SurfaceIntervalBounds,same_interval_query
    q,obs,model,active,regions=witness_fixture([-.02,-.01])
    rows=DeviceWitnessRows(model.fk,q,obs);query=interval_query_fixture(rows,regions)
    part=torch.tensor([0,1]);region=torch.tensor([0,1])
    assert same_interval_query(rows.solid,query,obs['link_names'],part,region,1).all()
    point=query.point_local.clone();point[0,0,0,0]+=.01
    normal=query.normal.clone();normal[0,0,0]=torch.tensor([1.,0.,0.])
    gap=query.signed_gap.clone();gap[0,0,0]-=.005
    unique=query.unique.clone();unique[0,0,0]=False
    for wrong in (dataclasses.replace(query,point_local=point),dataclasses.replace(query,normal=normal),
                  dataclasses.replace(query,signed_gap=gap),dataclasses.replace(query,unique=unique)):
        matched=same_interval_query(rows.solid,wrong,obs['link_names'],part,region,1)
        assert matched.tolist()==[False,True]
        lower=q.new_zeros(1,6,4)
        lower[0,0,0]=(-rows.solid.distances[0]).relu().square();lower[0,1,1]=(-rows.solid.distances[1]).relu().square()
        loss,metrics=unified_region_objective(model,rows,active,torch.zeros(1,6,dtype=torch.long),{},regions,
            surface_interval=SurfaceIntervalBounds(lower,torch.zeros_like(lower),
                solid_target_mask=torch.tensor([[False,True],[False,True]]),query=wrong))
        assert metrics['unified_physical_clearance_component'].item()>0
        assert torch.autograd.grad(loss.sum(),q,retain_graph=True)[0][0,2]<0


def test_opposite_violations_in_one_region_both_receive_gradients():
    q, obs, model, active, regions = witness_fixture([.03, -.02])
    class IndependentFootLinks:
        def link_poses(self, q, names):
            p = torch.stack([q[:, :3], q[:, 7:10]], 1)
            return p, torch.eye(3).to(q).expand(len(q), 2, 3, 3)
    model.fk = IndependentFootLinks()
    obs['link_names'] = ('left_ankle_roll_link', 'left_ankle_roll_sphere_1_link')
    obs['pairs']['geometry_point0_w'][:, 0] = -.1
    obs['pairs']['geometry_point1_w'][:, 0] = -.1
    obs['pairs']['part'][:] = 0
    # A top-face gap and a different side collision belong to the same
    # anatomical region, but only the top face is a valid contact target.
    obs['pairs']['upward'][1] = False
    obs['pairs']['eligible'][:] = False
    active[:, 1] = False; regions[:, 1] = False
    loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    both = torch.tensor(5.).log()+torch.tensor(17.).log()
    torch.testing.assert_close(loss, (both+both/24).reshape(1))
    grad = torch.autograd.grad(loss.sum(), q, retain_graph=True)[0]
    assert grad[0, 2] > 0  # Pull the top-facing witness down.
    assert grad[0, 9] < 0  # Push the other witness out of penetration.
    pieces = metrics['unified_attraction_component']+metrics['unified_separation_component']
    torch.testing.assert_close(pieces, loss)
    torch.testing.assert_close(torch.autograd.grad(pieces.sum(), q)[0], grad)


def test_activated_contacts_inside_actual_margin_have_no_attraction_gradient():
    q, obs, model, active, regions = witness_fixture([.019, .002])
    loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    assert metrics['region_plan_realized'].item() == 1
    assert loss.item() == 0
    assert torch.autograd.grad(loss.sum(), q)[0].abs().sum() == 0


def test_frozen_activated_witness_still_attracts_after_fk_crosses_margin():
    q, obs, model, active, regions = witness_fixture([.019, .002])
    rows = DeviceWitnessRows(model.fk, q, obs)
    # Move a frozen differentiable witness without changing query-time flags.
    rows.distances = rows.distances + q.new_tensor([.003, 0.])
    loss, _ = unified_region_objective(model, rows,
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    assert obs['pairs']['eligible'][0]
    assert torch.autograd.grad(loss.sum(), q)[0][0, 2] > 0


def test_regional_attraction_uses_each_actual_pair_margin():
    q, obs, model, active, regions = witness_fixture([.019, .019])
    obs['pairs']['includemargin'] = torch.tensor([.01, .03])
    for key in ('active', 'constraint_allocated', 'eligible'):
        obs['pairs'][key] = torch.tensor([False, True])
    loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    gradient = torch.autograd.grad(loss.sum(), q)[0]
    assert gradient[0, 2] > 0 and gradient[0, 9] == 0
    assert metrics['region_plan_realized'].item() == 0


def test_zero_regional_residual_at_margin_does_not_certify_contact():
    q, obs, model, active, regions = witness_fixture([.02, .002])
    loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    assert loss.item() == 0
    assert not obs['pairs']['eligible'][0]
    assert metrics['region_plan_realized'].item() == 0


def test_native_candidate_does_not_turn_off_material_recovery_field():
    q, obs, model, active, regions = witness_fixture([.019, .002])
    class EmbeddedMaterials(TestRegions):
        def missing_region_distance(self, fk, q, surface, scene, margin):
            result = q.new_zeros(len(q), 6, 4)
            result[:, 0, 0] = ((.09-q[:, 2]).relu()/(.25*margin)).square()
            return result
    model.region_geometry = EmbeddedMaterials()
    loss, metrics = unified_region_objective(model, DeviceWitnessRows(model.fk, q, obs),
        active, torch.zeros(1, 6, dtype=torch.long), {}, regions)
    assert metrics['region_plan_realized'].item() == 1
    assert obs['pairs']['task_pair'][0] and obs['pairs']['eligible'][0]
    assert loss.item() > 0 and torch.autograd.grad(loss.sum(), q)[0][0, 2] < 0
