import torch
from supermimic.contact_entry_projection import project_entry_step

def test_projects_actual_step_and_keeps_nonconflicting_progress():
    p=torch.nn.Parameter(torch.tensor([1.,-1.]))
    audit=project_entry_step([p],torch.zeros(2),torch.eye(2))
    torch.testing.assert_close(p,torch.tensor([0.,-1.]),atol=1e-6,rtol=0)
    assert audit['predicted_harmed_after']==0
    assert audit['retained_step_ratio']>.7

def test_constraint_is_independent_of_gradient_magnitude():
    proposals=[]
    for gradients in [torch.tensor([[1.,0.],[1.,1.]]),torch.tensor([[100.,0.],[.01,.01]])]:
        p=torch.nn.Parameter(torch.tensor([1.,-2.]))
        project_entry_step([p],torch.zeros(2),gradients)
        assert (gradients@p<=1e-6).all()
        proposals.append(p.detach())
    torch.testing.assert_close(*proposals,atol=1e-6,rtol=0)


def test_active_set_adds_a_constraint_not_initially_violated():
    from supermimic.contact_projection_gpu import solve_active_from_gradients
    rows=torch.tensor([[1.,0.,0.],[-1.,1.,0.]],dtype=torch.float64)
    rows=rows/rows.norm(dim=1,keepdim=True)
    proposed=torch.tensor([1.,.5,-1.],dtype=torch.float64)
    actual,audit=solve_active_from_gradients(rows,proposed,tolerance=1e-9)
    assert audit['passes'][0]['added']==1
    assert audit['full_constraints']==2
    assert (rows@actual<=1e-8).all()
    torch.testing.assert_close(actual,torch.tensor([0.,0.,-1.],dtype=torch.float64),atol=1e-7,rtol=0)


def test_active_set_keeps_a_fully_feasible_step():
    from supermimic.contact_projection_gpu import solve_active_from_gradients
    rows=torch.eye(3,dtype=torch.float64)
    proposed=-torch.ones(3,dtype=torch.float64)
    actual,audit=solve_active_from_gradients(rows,proposed)
    torch.testing.assert_close(actual,proposed,atol=0,rtol=0)
    assert audit['active']==0


def test_cuda_projection_matches_cpu_reference():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    gen=torch.Generator().manual_seed(42)
    gradients=torch.randn(32,64,generator=gen)
    proposed=torch.randn(64,generator=gen)*.001
    cpu=torch.nn.Parameter(proposed.clone())
    project_entry_step([cpu],torch.zeros_like(cpu),gradients.clone())
    gpu=torch.nn.Parameter(proposed.cuda())
    audit=project_entry_step([gpu],torch.zeros_like(gpu),gradients.cuda())
    torch.testing.assert_close(gpu.cpu(),cpu,atol=2e-6,rtol=1e-3)
    assert audit['solver']=='gpu_full'
    assert float((gradients.cuda()@gpu.detach()).max())<2e-6
