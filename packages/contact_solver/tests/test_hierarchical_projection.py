import numpy as np
from contact_solver.research.hierarchical_projection import lexicographic_step


def test_feasibility_cannot_be_traded_for_smaller_posture_change():
    x, audit = lexicographic_step([-1.], [[1.]], [0.], [[1.]], [1e6], [-2.], [2.])
    assert abs(x[0]-1.) < 1e-5
    assert audit['minimum_linearized_violation'] == 0


def test_infeasible_trust_region_is_reported_without_violating_bounds():
    x, audit = lexicographic_step([-3.], [[1.]], [0.], [[1.]], [1.], [-1.], [1.])
    np.testing.assert_allclose(x, [1.], atol=1e-5)
    assert abs(audit['minimum_linearized_violation']-2.) < 1e-6


def test_secondary_objective_prefers_cheap_joint_over_expensive_root():
    x, audit = lexicographic_step([-1.], [[1., 1.]], [0., 0.], np.eye(2), [100., 1.], [-2., -2.], [2., 2.])
    np.testing.assert_allclose(x, [1/101, 100/101], atol=1e-5)
    assert audit['achieved_linearized_violation'] < 1e-5
