import copy
import pytest
import torch
from contact_solver.constraint_learning import Residual
from contact_solver.constraint_training import ConstraintBatch, ConstrainedOptimizerStep, StepConfig
from contact_solver.feasibility_filter import ProtectionRule


def setup(dtype=torch.float64, trials=6):
    model = torch.nn.Linear(2, 1, bias=False).to(dtype)
    with torch.no_grad(): model.weight.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.01)
    step = ConstrainedOptimizerStep(model, optimizer, [ProtectionRule('keep', 1e-8)],
        config=StepConfig(max_trials=trials, interior_fraction=0))
    return model, optimizer, step


def rows(q):
    return [Residual.scaled('a','scene','keep','le',q[0:1],1.),
            Residual.scaled('b','scene','keep','le',-q[0:1],1.)]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_shared_batch_projects_actual_adam_increment(dtype):
    model, optimizer, step = setup(dtype)
    def closure():
        q = model.weight[0]; r = rows(q)
        return ConstraintBatch(((q-1)**2).sum(), r, r, {}, 2)
    _, report = step.step(closure)
    assert report['accepted']
    assert abs(float(model.weight[0,0].detach())) < 1e-7
    assert model.weight[0,1] > 0
    assert optimizer.state[model.weight]['step'] == 1
    assert len(step.al.state_dict()['entries']) == 2


def equal_tree(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b


def test_reject_restores_populated_adam_al_buffers_rng():
    model, optimizer, step = setup(trials=2)
    model.register_buffer('counter',torch.tensor(0))
    def closure():
        model.counter.add_(1)
        q=model.weight[0]; r=rows(q)
        return ConstraintBatch(((q-1)**2).sum(),r,r,{'actual':True},2)
    step.step(closure)
    before=copy.deepcopy(model.state_dict()); opt=copy.deepcopy(optimizer.state_dict()); al=step.state_dict()
    rng=torch.random.get_rng_state(); calls=0
    def rejected():
        nonlocal calls
        batch=closure(); calls+=1
        torch.rand(1)
        batch.evidence={'actual':calls==1}
        return batch
    _,report=step.step(rejected)
    assert not report['accepted'] and len(report['trials'])==2
    equal_tree(model.state_dict(),before); equal_tree(optimizer.state_dict(),opt)
    equal_tree(step.state_dict(),al); assert torch.equal(torch.random.get_rng_state(),rng)


def test_exception_restores_proposal():
    model,optimizer,step=setup(); calls=0
    def closure():
        nonlocal calls
        calls+=1
        if calls==2: raise ValueError('missing actual Newton evidence')
        q=model.weight[0];r=rows(q)
        return ConstraintBatch(((q-1)**2).sum(),r,r,{},2)
    with pytest.raises(ValueError,match='Newton'):step.step(closure)
    assert not optimizer.state and torch.count_nonzero(model.weight)==0


def test_checkpoint_and_batch_reordering():
    model,optimizer,step=setup()
    def closure():
        q=model.weight[0];r=rows(q)
        return ConstraintBatch(((q-1)**2).sum(),r,r,{},2)
    step.step(closure)
    state=step.state_dict();new=setup()[2];new.load_state_dict(state)
    r=rows(model.weight[0]);zero=model.weight.sum()*0
    assert torch.equal(new.al.loss(zero,r),new.al.loss(zero,list(reversed(r))))


def test_new_bad_component_cannot_hide_in_batch_mean():
    model,optimizer,step=setup(trials=2);calls=0
    def closure():
        nonlocal calls
        calls+=1;q=model.weight[0];r=rows(q)
        if calls>1:r.append(Residual.scaled('b','scene','keep/new','le',q[1:2],1.))
        return ConstraintBatch(((q-1)**2).sum(),r,r,{},2)
    _,report=step.step(closure)
    assert not report['accepted']
    assert report['trials'][0]['reasons'][0]['kind']=='new_protected_violation'


def test_projected_increment_includes_existing_adam_moments():
    model,optimizer,step=setup()
    optimizer.state[model.weight]=dict(step=torch.tensor(4.),
        exp_avg=torch.tensor([[-.4,.1]],dtype=torch.double),
        exp_avg_sq=torch.tensor([[.1,.2]],dtype=torch.double))
    other=torch.nn.Linear(2,1,bias=False).double()
    other.load_state_dict(model.state_dict())
    opt=torch.optim.AdamW(other.parameters(),lr=.1,weight_decay=.01)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    (((other.weight-1)**2).sum()/2).backward();opt.step()
    expected=other.weight.detach().clone()
    def closure():
        q=model.weight[0];r=rows(q)
        return ConstraintBatch(((q-1)**2).sum(),r,r,{},2)
    _,report=step.step(closure)
    assert report['accepted'] and report['fraction']==1.
    assert torch.allclose(model.weight[0,1],expected[0,1],atol=1e-12,rtol=0)
    assert abs(float(model.weight[0,0].detach()))<1e-8
    equal_tree(optimizer.state_dict(),opt.state_dict())
