import itertools
import numpy as np
import pytest
from contact_solver.separation_certificate import vertex_separation


def test_certificate_is_direction_invariant_and_respects_padding():
    a=np.array(list(itertools.product([-1.,1.],repeat=3)));b=a+[3.,0.,0.]
    assert vertex_separation(a,b,[1,0,0])==1.
    assert vertex_separation(a,b,[-3,0,0],.25)==.75
    assert vertex_separation(a,b,[0,1,0])<0
    assert vertex_separation(a,b,[1,0,0],1.1)<0


def test_overlap_is_not_certified_and_invalid_axis_fails():
    a=np.array(list(itertools.product([-1.,1.],repeat=3)))
    assert vertex_separation(a,a+[.5,0,0],[1,0,0])<0
    with pytest.raises(ValueError):vertex_separation(a,a,[0,0,0])
