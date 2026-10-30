import pytest
import torch
from contact_solver.newton_witness_loss import query_local_distances, selected_contact_cost, unwanted_contact_cost, full_body_violation


class TranslationFK:
    def link_poses(self, q, names):
        assert names == ('left_knee_link',)
        return q[:, None, :3], torch.eye(3, dtype=q.dtype)[None, None].expand(len(q), 1, 3, 3)


def pair(**updates):
    p = dict(witness_schema='newton_geometry_pair_v1', geometry_witness_available=True,
             constraint_type=1, constraint_active=True, allocated=True, efc_address=[0],
             body_name='left_knee_link', position_w=[.5, 0, .01], terrain_position_w=[.5, 0, 0],
             normal_w=[0, 0, 1], dist=.01, includemargin=.02, part=4, surface=1)
    return dict(p, **updates)


def test_solver_value_and_local_derivative():
    q = torch.zeros((1, 36), dtype=torch.float64, requires_grad=True)
    rows, _ = query_local_distances(TranslationFK(), q, query=lambda q,s:dict(candidate_pairs=[[pair()]]))
    d = rows[0][0][1]
    assert d.item() == .01
    d.backward()
    torch.testing.assert_close(q.grad[0,:3], torch.tensor([0.,0.,1.],dtype=q.dtype))


def test_each_call_refreshes_and_missing_is_not_contact():
    calls = []
    def query(q,s):
        calls.append(q.copy())
        return dict(candidate_pairs=[[]])
    q = torch.zeros((1,36), requires_grad=True)
    query_local_distances(TranslationFK(),q,query=query)
    rows,_ = query_local_distances(TranslationFK(),q+1,query=query)
    assert len(calls)==2 and calls[1][0,0]==1
    active=torch.zeros((1,6),dtype=torch.bool); active[0,4]=True
    cost,missing,realized=selected_contact_cost(q,rows,active,torch.ones((1,6),dtype=torch.long))
    assert missing[0,4] and not realized.any() and cost.sum()==0


def test_side_pair_does_not_realize_top_and_any_pair_can_realize_contact():
    q=torch.zeros((1,36),requires_grad=True)
    active=torch.zeros((1,6),dtype=torch.bool);active[0,4]=True
    surfaces=torch.ones((1,6),dtype=torch.long)
    rows=[[(pair(surface=9),q.sum()*0+.01)]]
    _,missing,realized=selected_contact_cost(q,rows,active,surfaces)
    assert missing[0,4] and not realized[0,4]
    rows[0].extend([(pair(),q.sum()*0+.01),
                   (pair(dist=.03,constraint_active=False,allocated=False),q.sum()*0+.03)])
    cost,missing,realized=selected_contact_cost(q,rows,active,surfaces)
    assert cost.sum()==0 and realized[0,4] and not missing[0,4]


@pytest.mark.parametrize('update', [dict(geometry_witness_available=False),
    dict(allocated=False),dict(constraint_active=False),dict(normal_w=[0,0,0])])
def test_invalid_solver_data_fails_closed(update):
    with pytest.raises(ValueError):
        query_local_distances(TranslationFK(),torch.zeros((1,36)),
            query=lambda q,s:dict(candidate_pairs=[[pair(**update)]]))


def test_native_frame_transforms_robot_terrain_witness_and_normal_together():
    q=torch.zeros((1,36),dtype=torch.float64,requires_grad=True)
    with torch.no_grad():q[0,3]=1
    basis=torch.tensor([[[0.,-1,0],[1,0,0],[0,0,1]]],dtype=q.dtype)
    origin=torch.tensor([[2.,3,4]],dtype=q.dtype)
    def query(value,scene):
        assert scene is None
        assert value[0,:3].tolist()==[2.,3,4]
        return dict(candidate_pairs=[[pair(position_w=[2,3.5,4.01],
            terrain_position_w=[2,3.5,4],normal_w=[1,0,0])]])
    rows,_=query_local_distances(TranslationFK(),q,query=query,world_frame=(origin,basis))
    p,d=rows[0][0]
    torch.testing.assert_close(q.new_tensor(p['terrain_position_w']),q.new_tensor([.5,0,0]))
    d.backward()
    torch.testing.assert_close(q.grad[0,:3],q.new_tensor([0,-1,0]))


def test_buffer_has_gradient_at_strict_activation_boundary_without_relabeling():
    q=torch.zeros((1,36),requires_grad=True)
    active=torch.zeros((1,6),dtype=torch.bool);active[0,4]=True
    distance=q[0,2]+.02
    rows=[[(pair(dist=.02,constraint_active=False,allocated=False),distance)]]
    cost,missing,realized=selected_contact_cost(q,rows,active,torch.ones((1,6),dtype=torch.long),activation_buffer_fraction=.25)
    assert cost.sum()>0 and not realized.any() and not missing.any()
    cost.sum().backward();assert q.grad[0,2]>0


def test_unwanted_contact_uses_max_not_manifold_count_and_ignores_wanted_parts():
    q=torch.zeros((1,36),requires_grad=True)
    active=torch.zeros((1,6),dtype=torch.bool)
    item=(pair(),q[0,2]+.01)
    a,flags=unwanted_contact_cost(q,[[item]],active,[{0,1}])
    b,_=unwanted_contact_cost(q,[[item,item,item]],active,[{0,1}])
    torch.testing.assert_close(a,b);assert flags[0,4]
    a.sum().backward();assert q.grad[0,2]<0
    active[0,4]=True
    c,_=unwanted_contact_cost(q,[[item]],active,[{0,1}]);assert c.sum()==0
    active[0,4]=False
    c,_=unwanted_contact_cost(q,[[(pair(surface=9),item[1])]],active,[{0,1}]);assert c.sum()==0


def test_self_collision_derivative_moves_both_bodies_apart():
    class TwoBodies:
        def link_poses(self,q,names):
            assert names==('a','b')
            return torch.stack((q[:,:3],q[:,3:6]),1),torch.eye(3)[None,None].expand(len(q),2,3,3)
    q=torch.zeros((1,36),requires_grad=True)
    w=dict(body_name0='a',body_name1='b',point0_w=[0,0,0],point1_w=[0,0,0],normal_w=[0,0,1],dist=-.02)
    d=full_body_violation(TwoBodies(),q,[dict(worst_terrain=None,worst_self=w)])
    assert d.item()==pytest.approx(.02)
    d.sum().backward();assert q.grad[0,2]==1 and q.grad[0,5]==-1


def test_batched_reduction_splits_tied_manifold_gradients():
    q=torch.zeros((2,36),requires_grad=True)
    rows=[[(pair(dist=.03,constraint_active=False,allocated=False,efc_address=[-1]),q[0,0]+.03),
           (pair(dist=.03,constraint_active=False,allocated=False,efc_address=[-1]),q[0,1]+.03)],[]]
    active=torch.ones((2,6),dtype=torch.bool);surface=torch.ones((2,6),dtype=torch.long)
    cost,missing,_=selected_contact_cost(q,rows,active,surface)
    cost.sum().backward()
    torch.testing.assert_close(q.grad[0,:2],q.new_tensor([.01,.01]))
    assert missing[1].all() and cost[1].sum()==0
    q.grad=None
    rows=[[(pair(),q[0,0]+.01),(pair(),q[0,1]+.01)],[]]
    cost,_=unwanted_contact_cost(q,rows,~active,[{1},{1}])
    cost.sum().backward()
    torch.testing.assert_close(q.grad[0,:2],q.new_tensor([-.01,-.01]))


def test_invalid_fullbody_normal_preserves_forensic_context():
    import json
    q=torch.zeros((1,36))
    w=dict(body_name0=None,body_name1=None,point0_w=[0,0,0],point1_w=[0,0,0],
           normal_w=[0,0,0],dist=-.01,shape0=2,shape1=9)
    with pytest.raises(ValueError,match='Invalid full-body normal: ') as error:
        full_body_violation(None,q,[dict(worst_terrain=w,worst_self=None)])
    details=json.loads(str(error.value).split(': ',1)[1])
    assert details['witness']==w and details['transformed_norm']==0
    assert details['q']==q[0].tolist() and details['basis'] is None


def test_captured_zero_normal_detaches_only_bad_derivative_and_audits(tmp_path):
    import json
    class TwoBodies:
        def link_poses(self,q,names):
            return torch.stack((q[:,:3],q[:,3:6]),1),torch.eye(3)[None,None].expand(len(q),2,3,3)
    q=torch.zeros((2,36),requires_grad=True)
    bad=dict(body_name0='a',body_name1='b',shape0=60,shape1=74,
        point0_w=[0.6504883170127869,-0.0665808767080307,0.7732830047607422],
        point1_w=[0.6504882574081421,-0.0665808841586113,0.7732829451560974],
        normal_w=[0.,0.,0.],dist=-0.004999999888241291)
    good=dict(bad,normal_w=[0,0,1],dist=-.02)
    audit=tmp_path/'invalid.jsonl'
    depth,counts=full_body_violation(TwoBodies(),q,
        [dict(worst_terrain=None,worst_self=w) for w in (bad,good)],
        invalid_policy='detach',return_validity=True,audit_path=audit)
    torch.testing.assert_close(depth,q.new_tensor([.005,.02]))
    assert counts.tolist()==[1,0]
    # Other supervision on the invalid sample and valid collision gradients survive.
    (depth.sum()+q[0,8]).backward()
    assert q.grad[0,8]==1 and q.grad[0,:6].abs().sum()==0
    assert q.grad[1,2]==1 and q.grad[1,5]==-1
    record=json.loads(audit.read_text())
    assert record['witness']==bad and record['geometry_valid'] is False


def test_invalid_witness_detachment_requires_explicit_validity():
    with pytest.raises(ValueError,match='explicit validity'):
        full_body_violation(None,torch.zeros(1,36),[],invalid_policy='detach')


def test_nonfinite_distance_is_not_silently_detached():
    w=dict(body_name0=None,body_name1=None,dist=float('nan'))
    with pytest.raises(ValueError,match='Nonfinite full-body distance'):
        full_body_violation(None,torch.zeros(1,36),[dict(worst_terrain=w,worst_self=None)],
                            invalid_policy='detach',return_validity=True)


def test_fullbody_ties_split_both_witness_gradients_and_empty_signed_is_unknown():
    class Bodies:
        def link_poses(self, q, names):
            return torch.stack([q[:, :3] if name == 'a' else q[:, 3:6] for name in names], 1), torch.eye(3).expand(len(q), len(names), 3, 3)
    a = dict(body_name0=None,body_name1='a',point0_w=[0,0,0],point1_w=[0,0,0],normal_w=[0,0,1],dist=-.02)
    b = dict(a,body_name1='b')
    q = torch.zeros(2,36,requires_grad=True)
    rows = [dict(worst_terrain=a,worst_self=b),dict(worst_terrain=None,worst_self=None)]
    value = full_body_violation(Bodies(),q,rows)
    torch.testing.assert_close(value,torch.tensor([.02,0.]))
    value.sum().backward()
    assert q.grad[0,2] == -.5 and q.grad[0,5] == -.5 and q.grad[1].abs().sum() == 0
    signed = full_body_violation(None,q,[dict(closest_terrain=None,closest_self=None)]*2,signed=True)
    assert torch.isneginf(signed).all()


def test_invalid_nan_normal_does_not_poison_other_collision_gradients(tmp_path):
    class Body:
        def link_poses(self,q,names):
            return q[:,None,:3],torch.eye(3).expand(len(q),1,3,3)
    good = dict(body_name0=None,body_name1='a',point0_w=[0,0,0],point1_w=[0,0,0],normal_w=[0,0,1],dist=-.02)
    bad = dict(good,normal_w=[float('nan'),0,0],dist=-.03)
    q = torch.zeros(2,36,requires_grad=True)
    value, invalid = full_body_violation(Body(),q,[dict(worst_terrain=w,worst_self=None) for w in (bad,good)],
        invalid_policy='detach',return_validity=True,audit_path=tmp_path/'invalid.jsonl')
    value.sum().backward()
    assert invalid.tolist() == [1,0] and torch.isfinite(q.grad).all()
    assert q.grad[0].abs().sum() == 0 and q.grad[1,2] == -1


def test_fullbody_native_frame_rotates_both_body_derivatives():
    class Bodies:
        def link_poses(self,q,names):
            return torch.stack((q[:,:3],q[:,3:6]),1),torch.eye(3).expand(len(q),2,3,3)
    q = torch.zeros(1,36,requires_grad=True)
    frame = (torch.tensor([[3.,4.,0.]]),torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]]))
    witness = dict(body_name0='a',body_name1='b',point0_w=[3,4,0],point1_w=[3,4,0],normal_w=[1,0,0],dist=-.02)
    value = full_body_violation(Bodies(),q,[dict(worst_terrain=None,worst_self=witness)],world_frame=frame)
    value.sum().backward()
    assert q.grad[0,1] == -1 and q.grad[0,4] == 1
    assert q.grad[0,0] == q.grad[0,3] == 0
