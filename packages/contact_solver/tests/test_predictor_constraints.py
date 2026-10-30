from types import SimpleNamespace
import pytest
import torch
from contact_solver.predictor_constraints import PredictorConstraintAdapter
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def fixture(distance=.01):
    fk=CanonicalG1ForwardKinematics().double()
    q=torch.zeros(1,36,dtype=torch.double);q[:,2]=1.;q[:,3]=1.;q.requires_grad_()
    dist=q[:,2]*0+distance
    p=dict(type=torch.tensor([1]),dist=dist.detach(),includemargin=torch.tensor([.02]),
        efc_address=torch.tensor([[0]]),constraint_rows=torch.tensor([1]),
        active=torch.tensor([True]),constraint_allocated=torch.tensor([True]),
        eligible=torch.tensor([True]),upward=torch.tensor([True]),task_pair=torch.tensor([True]),
        full_kind=torch.tensor([0]),sample=torch.tensor([0]),part=torch.tensor([0]),primary_surface=torch.tensor([0]))
    rows=SimpleNamespace(q=q,pair=p,distances=dist,normal_valid=torch.tensor([True]),observed={'configured_margin':torch.tensor([.02])})
    scene=dict(box_center=torch.tensor([[10.,10.,1.]]),box_rotation=torch.eye(3)[None],
               box_half_extents=torch.tensor([[.5,.5,.5]]),ground_height=torch.tensor([0.]))
    active=torch.tensor([[True,False,False,False,False,False]])
    return fk,rows,scene,dict(active=active,surface=torch.tensor([[0,-1,-1,-1,-1,-1]]),
        sample_ids=['motion/frame'],context_ids=['scene/task'],prior=q.sum()*0,nominal_q=q.detach())


def test_canonical_geometry_finite_gradients_and_active_band():
    fk,rows,scene,kwargs=fixture()
    result=PredictorConstraintAdapter()(SimpleNamespace(fk=fk),rows,scene,**kwargs)
    result.validate()
    contact=next(r for r in result.residuals if r.constraint_id.startswith('contact/'))
    assert float(contact.value.detach())==0
    grad=torch.autograd.grad(sum(r.value.square().sum() for r in result.residuals)+result.prior,rows.q)[0]
    assert torch.isfinite(grad).all()
    assert all(result.evidence.values())


def test_active_deep_pair_does_not_disable_penetration():
    fk,rows,scene,kwargs=fixture(-.09)
    result=PredictorConstraintAdapter()(SimpleNamespace(fk=fk),rows,scene,**kwargs)
    assert next(r for r in result.residuals if r.constraint_id=='collision/verified_fullbody').value.item()>8


def test_missing_and_unallocated_contact_evidence_raise():
    fk,rows,scene,kwargs=fixture()
    adapter=PredictorConstraintAdapter()
    rows.pair['efc_address'].fill_(-1);rows.pair['constraint_allocated'].fill_(False)
    with pytest.raises(ValueError,match='no allocated'):adapter(SimpleNamespace(fk=fk),rows,scene,**kwargs)
    del rows.pair['type']
    with pytest.raises(ValueError,match='Missing actual'):adapter(SimpleNamespace(fk=fk),rows,scene,**kwargs)


def test_self_constraint_identity_ignores_query_world_shape_offsets():
    fk,rows,scene,kwargs=fixture(-.001)
    rows.pair['full_kind'].fill_(1)
    rows.pair['body_link0']=torch.tensor([2]);rows.pair['body_link1']=torch.tensor([5])
    rows.pair['shape0']=torch.tensor([10]);rows.pair['shape1']=torch.tensor([20])
    adapter=PredictorConstraintAdapter()
    first=adapter(SimpleNamespace(fk=fk),rows,scene,**kwargs)
    rows.pair['shape0']+=100;rows.pair['shape1']+=100
    second=adapter(SimpleNamespace(fk=fk),rows,scene,**kwargs)
    ids=lambda b:[r.key for r in b.residuals if r.constraint_id.startswith('collision/self/')]
    assert ids(first)==ids(second)==[('motion/frame','scene/task','collision/self/2/5')]
