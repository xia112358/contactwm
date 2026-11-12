import torch
from supermimic.contact_case_gradient import guarded_case_gradient


def test_conflicting_case_component_is_projected_without_discarding_update():
    g=torch.tensor([[1.,0.,1.],[-2.,1.,1.],[0.,0.,0.]])
    direction,info=guarded_case_gradient(g)
    assert torch.isfinite(direction).all() and direction.norm()>0
    assert direction[2]>0 and info['nonzero']==2
    assert info['conflicts_after']<=info['conflicts_before']


def test_all_zero_case_gradients_produce_finite_zero():
    direction,info=guarded_case_gradient(torch.zeros(3,4))
    assert torch.equal(direction,torch.zeros(4)) and info['nonzero']==0


def test_identical_cases_keep_common_descent():
    direction,info=guarded_case_gradient(torch.tensor([[1.,2.],[1.,2.]]))
    torch.testing.assert_close(direction,torch.tensor([1.,2.]))
    assert info['conflicts_before']==info['conflicts_after']==0


def test_actual_adam_displacement_is_projected_not_wholly_discarded():
    from supermimic.contact_case_gradient import guard_adam_displacement
    gradients=torch.tensor([[1.,0.],[0.,1.]])
    actual=torch.tensor([.1,-.2])
    adjusted,info=guard_adam_displacement(actual,gradients)
    torch.testing.assert_close(adjusted,torch.tensor([0.,-.2]),atol=1e-6,rtol=0)
    assert info['conflicts_before']==1 and info['conflicts_after']==0 and adjusted.norm()>0


def test_actual_update_projection_scales_to_many_conflicting_cases():
    from supermimic.contact_case_gradient import guard_adam_displacement
    generator=torch.Generator().manual_seed(42)
    gradients=torch.randn(151,512,generator=generator)
    displacement=torch.randn(512,generator=generator)*1e-5
    adjusted,info=guard_adam_displacement(displacement,gradients)
    assert adjusted.norm()>0 and info['conflicts_after']==0
    assert torch.isfinite(adjusted).all()
