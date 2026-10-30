import torch
import pytest
from contact_solver.constraint_direction import project_direction


def test_removes_outward_component_keeps_inward_and_tangent():
    d=torch.tensor([2.,-3.,4.],dtype=torch.double)
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.double)
    result,info=project_direction(d,a)
    torch.testing.assert_close(result,torch.tensor([0.,-3.,4.],dtype=torch.double))
    assert info['converged']


@pytest.mark.parametrize('algorithm',['coordinate','rowspace'])
def test_dependent_and_opposing_constraints(algorithm):
    a=torch.tensor([[1.,1.],[2.,2.],[-1.,-1.]],dtype=torch.double)
    result,info=project_direction(torch.tensor([3.,1.],dtype=torch.double),a,algorithm=algorithm)
    torch.testing.assert_close(result,torch.tensor([1.,-1.],dtype=torch.double))
    assert info['converged']


def test_network_metric_changes_projection():
    k=torch.tensor([[1.,2.],[0.,1.]],dtype=torch.double)
    g=torch.tensor([-1.,-1.],dtype=torch.double)
    a=torch.tensor([[1.,0.]],dtype=torch.double)
    d,_=project_direction(-k.T@g,a@k)
    assert float((a@k@d).max())<1e-8
    naive,_=project_direction(-g,a)
    assert float((a@k@k.T@naive).max())>0


def test_inward_reserve_is_enforced():
    result,info=project_direction(torch.tensor([1.,2.]),torch.tensor([[1.,0.]]),bounds=torch.tensor([-.1]))
    torch.testing.assert_close(result,torch.tensor([-.1,2.],dtype=torch.double))
    assert info['converged']


def test_large_float_parameter_gradients_use_relative_accuracy():
    normals=torch.tensor([[1.,1.,0.],[1.,1.0001,0.]],dtype=torch.float32)
    direction=torch.tensor([1e7,1e7,-1e7],dtype=torch.float32)
    result,info=project_direction(direction,normals)
    assert info['converged']
    assert info['relative_violation']<1e-9
    assert float(result[-1])==-1e7


@pytest.mark.parametrize('scale',[1.,1e7])
def test_rowspace_nearly_opposing_rows_and_reserve(scale):
    # Nearly parallel opposing planes: the coordinate dual converges slowly.
    a=torch.tensor([[1.,0.,0.],[-1.,1e-4,0.],[2.,0.,0.]],dtype=torch.double)
    d=torch.tensor([1.,1.,3.],dtype=torch.double)*scale
    b=torch.tensor([0.,-1e-4,0.],dtype=torch.double)*scale
    result,info=project_direction(d,a,bounds=b,algorithm='rowspace')
    assert info['converged'],info
    torch.testing.assert_close(result,torch.tensor([0.,-1.,3.],dtype=torch.double)*scale,atol=scale*1e-7,rtol=1e-7)


def test_rowspace_infeasible_reserve_is_not_accepted():
    result,info=project_direction(torch.tensor([2.,3.]),torch.tensor([[1.,0.],[-1.,0.]]),bounds=torch.tensor([-1.,-1.]),algorithm='rowspace')
    assert not info['converged']
    assert info['feasibility']['status']==2


def test_rowspace_zero_gradient_with_feasible_reserve():
    result,info=project_direction(torch.zeros(2,dtype=torch.double),torch.tensor([[1.,0.]],dtype=torch.double),bounds=torch.tensor([-.1],dtype=torch.double),algorithm='rowspace')
    assert info['converged'],info
    torch.testing.assert_close(result,torch.tensor([-.1,0.],dtype=torch.double))


def test_rowspace_cone_without_strict_interior():
    generator=torch.Generator().manual_seed(127)
    basis=torch.linalg.qr(torch.randn(40,12,generator=generator,dtype=torch.double)).Q.T
    a=torch.cat((basis,-basis,basis[:6]),0)
    d=torch.randn(40,generator=generator,dtype=torch.double)*1e6
    result,info=project_direction(d,a,algorithm='rowspace')
    assert info['converged'],info
    expected=d-basis.T@(basis@d)
    torch.testing.assert_close(result,expected,atol=1e-7,rtol=1e-9)


def test_recorded_degenerate_cone_regression():
    import json
    import numpy as np
    from pathlib import Path
    case=json.loads((Path(__file__).parent/'data/rowspace_degenerate_projection.json').read_text())
    gram=np.array(case['gram']);e,u=np.linalg.eigh(gram)
    keep=e>np.finfo(float).eps*len(gram)*max(float(e[-1]),1.)
    e,u=e[keep],u[:,keep]
    a=torch.tensor(u*np.sqrt(e));d=torch.tensor((u.T@np.array(case['rhs']))/np.sqrt(e))
    result,info=project_direction(d,a,algorithm='rowspace')
    assert info['converged'],info
    assert info['facial_reduction'] is not None
    assert float((a@result).max())<1e-9
    assert abs(float(result@(d-result)))<1e-8
