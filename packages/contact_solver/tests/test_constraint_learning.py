import copy
import pytest
import torch
from contact_solver.constraint_learning import Residual,AugmentedLagrangian,penalty_loss
from contact_solver.constraint_residuals import PlaneSurface,ContactTask,contact_residuals,collision_residual,joint_limit_residuals


def r(sample,value,kind='le',ids=None,context='task1'):
    x=torch.as_tensor(value,dtype=torch.double)
    return Residual.scaled(sample,context,'gap',kind,x,1.,component_ids=ids)


def test_dual_release_and_sample_isolation():
    al=AugmentedLagrangian(rho=2)
    al.update([r('a',[1]),r('b',[3])],adapt_rho=False)
    x=torch.tensor([-.25],dtype=torch.double,requires_grad=True)
    row=r('a',x)
    assert torch.allclose(torch.autograd.grad(al.loss(x.sum()*0,[row]),x)[0],x.new_tensor([1.5]))
    al.update([r('a',[-2])],adapt_rho=False)
    state={tuple(e['key']):e for e in al.state_dict()['entries']}
    assert state[('a','task1','gap')]['dual']['0']==0
    assert state[('b','task1','gap')]['dual']['0']==6
    assert al.loss(x.sum()*0,[r('a',x,context='task2')])==0


def test_component_reorder_checkpoint_and_no_forward_mutation():
    al=AugmentedLagrangian(rho=1)
    al.update([r('a',[1,3],ids=('pair1','pair2'))],adapt_rho=False)
    state=al.state_dict();other=AugmentedLagrangian(rho=1);other.load_state_dict(state)
    a=r('a',[.2,.4],ids=('pair1','pair2'));b=r('a',[.4,.2],ids=('pair2','pair1'))
    assert torch.allclose(al.loss(a.value.sum()*0,[a]),other.loss(b.value.sum()*0,[b]))
    assert al.state_dict()==state
    other.reset(sample_id='a');assert other.state_dict()['entries']==[]


def test_invalid_constraints_fail_atomically():
    al=AugmentedLagrangian()
    with pytest.raises(ValueError):al.update([r('a',[1]),r('b',[float('nan')])])
    assert al.state_dict()['entries']==[]
    with pytest.raises(ValueError):al.loss(torch.tensor(0.),[r('a',[1]),r('a',[2])])
    al.update([r('a',[1])])
    with pytest.raises(ValueError):al.update([r('a',[1],'eq')])


def test_penalty_al_gradients_and_conflicting_constraints():
    # General two-coordinate task: establish x+y=1 while retaining y=0.
    # The prior prefers the origin; AL must coordinate variables without a teacher.
    x=torch.zeros(2,dtype=torch.double,requires_grad=True)
    al=AugmentedLagrangian(rho=2,max_rho=2)
    def rows():return [r('sample',(x.sum()-1).reshape(1),'eq'),Residual.scaled('sample','task1','retain','eq',x[1:2],1.)]
    for outer in range(30):
        for inner in range(80):
            loss=al.loss(.5*x.square().sum(),rows());g=torch.autograd.grad(loss,x)[0]
            with torch.no_grad():x-=.05*g
        al.update(rows(),adapt_rho=False)
    assert torch.allclose(x,torch.tensor([1.,0.],dtype=x.dtype),atol=1e-4)
    z=torch.tensor([.1,-.2],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda v:al.loss(v.square().sum(),[r('other',v)]),(z,))
    assert torch.autograd.gradcheck(lambda v:penalty_loss(v.square().sum(),[r('other',v)],rho=2),(z,))


def test_contact_modes_rotated_plane_and_limits():
    # Different frame/body names are irrelevant to the residual implementation.
    surface=PlaneSurface(torch.zeros(3,dtype=torch.double),torch.tensor([1.,0,0],dtype=torch.double),torch.tensor([[0.,1,0],[0.,0,1]],dtype=torch.double))
    gap=torch.tensor(.03,dtype=torch.double,requires_grad=True)
    task=ContactTask('arbitrary-body','establish',.01,0,.02)
    rows=contact_residuals('s','v',task,surface,support_gap=gap)
    assert rows[0].value.item()==-3 and abs(rows[1].value.item()-1)<1e-12
    keep=contact_residuals('s','v',ContactTask('sphere','keep',.01,0,.02),surface,support_gap=gap,
        anchor_world=torch.tensor([.03,.02,0.],dtype=torch.double),anchor_target=torch.zeros(3,dtype=torch.double))
    assert torch.equal(keep[-1].value,torch.tensor([2.,0.],dtype=torch.double))
    released=contact_residuals('s','v',ContactTask('hand','release',.01,.04,.04),surface,support_gap=gap)
    assert abs(released[0].value.item()-1)<1e-12
    with pytest.raises(ValueError):contact_residuals('s','v',ContactTask('pad','establish',.01,0,.02,coverage=True),surface,support_gap=gap)
    limits=joint_limit_residuals('s','v',torch.tensor([2.]),torch.tensor([-1.]),torch.tensor([1.]),angle_scale=.1,joint_names=('hinge',))
    assert limits[1].value.item()==10


def test_solver_contact_band_does_not_confuse_activation_with_penetration():
    surface=PlaneSurface(torch.zeros(3,dtype=torch.double),torch.tensor([1.,0,0],dtype=torch.double),torch.tensor([[0.,1,0],[0.,0,1]],dtype=torch.double))
    task=ContactTask('arbitrary-body','establish',.01,-.001,.02-1e-6)
    for distance,realized,expected_sign in ((.01,True,0),(-.0005,True,0),(-.09,True,-1),(.03,False,1)):
        gap=torch.tensor(distance,dtype=torch.double,requires_grad=True)
        rows=contact_residuals('s','v',task,surface,support_gap=gap,realized=realized)
        loss=sum(r.value.relu().square().sum() for r in rows)
        gradient=torch.autograd.grad(loss,gap)[0]
        assert gradient.sign().item()==expected_sign
        if realized:assert rows[1].value.item()==0
    # Even a retained AL multiplier must not attract an already active contact.
    gap=torch.tensor(.01,dtype=torch.double,requires_grad=True)
    upper=contact_residuals('s','v',task,surface,support_gap=gap,realized=True)[1]
    al=AugmentedLagrangian(rho=1)
    al.update([Residual.scaled('s','v',upper.constraint_id,'le',gap.new_tensor(1.),.01)])
    assert torch.autograd.grad(al.loss(gap*0,[upper]),gap)[0].item()==0
    # Protection must retain the boundary normal even with zero attraction.
    boundary=contact_residuals('s','v',task,surface,support_gap=gap)[1]
    assert torch.autograd.grad(boundary.value.sum(),gap)[0].item()==100.


def test_environment_and_self_rows_keep_distinct_ids():
    d=torch.tensor([-.1,.2],requires_grad=True)
    rows=[collision_residual('s','v',name,d,distance_scale=.1,component_ids=('shapeA','shapeB')) for name in ('environment','self')]
    loss=penalty_loss(d.sum()*0,rows,rho=1)
    assert torch.allclose(torch.autograd.grad(loss,d)[0],torch.tensor([-20.,0.]))


def test_fields_and_coverage_are_frame_independent():
    from contact_solver.constraint_residuals import OrientedBoxField,SphereField
    center=torch.tensor([2.,-1.,3.],dtype=torch.double)
    rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]],dtype=torch.double)
    box=OrientedBoxField(center,rotation,torch.tensor([1.,2.,3.],dtype=torch.double))
    local=torch.tensor([[.8,0,0],[1.2,0,0]],dtype=torch.double)
    points=(local@rotation.T+center).requires_grad_()
    assert torch.allclose(box.distance(points),torch.tensor([-.2,.2],dtype=torch.double))
    assert torch.autograd.gradcheck(box.distance,(points,))
    sphere=SphereField(center,1.)
    assert torch.allclose(sphere.distance(points),torch.tensor([-.2,.2],dtype=torch.double))
    normal=rotation[:,2];tangents=rotation[:,:2].T
    edges=torch.stack((rotation[:,0],-rotation[:,0],rotation[:,1],-rotation[:,1]))
    surface=PlaneSurface(center,normal,tangents,edges,edges@center+1)
    coverage=surface.coverage(points)
    assert torch.allclose(coverage.amax(-1),torch.tensor([-.2,.2],dtype=torch.double))
    q=ContactTask('pad','establish',.01,0,.02,coverage=True)
    rows=contact_residuals('a','v',q,surface,support_gap=points.new_tensor(.01),coverage_points=points,
        normal_world=normal,normal_target=-normal)
    assert rows[-2].value.norm()>0  # Antiparallel normals must not yield zero.


def test_dual_growth_reset_and_reappearing_components():
    al=AugmentedLagrangian(rho=1,growth=2,max_rho=4)
    al.update([r('a',[1,2],ids=('A','B'))])
    al.update([r('a',[2],ids=('A',))])
    state=al.state_dict()['entries'][0]
    assert state['rho']==2 and state['dual']['B']==2
    restored=AugmentedLagrangian(rho=1,growth=2,max_rho=4)
    restored.load_state_dict(al.state_dict())
    restored.update([r('a',[-3],ids=('B',))],adapt_rho=False)
    assert restored.state_dict()['entries'][0]['dual']['B']==0
    al.reset(sample_id='other');assert len(al.state_dict()['entries'])==1
    al.reset(sample_id='a',context_id='task1');assert not al.state_dict()['entries']
