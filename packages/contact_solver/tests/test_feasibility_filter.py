import torch
import pytest
from contact_solver.constraint_learning import Residual
from contact_solver.feasibility_filter import FeasibilityFilter,ProtectionRule


def rows(value,kind='eq',ids=('a','b')):
    return [Residual.scaled('sample','context','keep',kind,torch.tensor(value,dtype=torch.double),1.,component_ids=ids)]


def test_protects_components_not_just_aggregate():
    f=FeasibilityFilter([ProtectionRule('keep',.1)])
    # Sum of squares improves, but one already-satisfied component is broken.
    result=f.assess(rows([2.,0.]),rows([1.,.2]),required_evidence={},candidate_evidence={})
    assert not result['accepted']
    assert result['reasons'][0]['key'][-1]=='b'


def test_restoration_ratchets_and_identity_survives_reorder():
    f=FeasibilityFilter([ProtectionRule('keep',.1)])
    initial=rows([2.,0.]);better=rows([.05,1.],ids=('b','a'))
    assert f.assess(initial,better,required_evidence={},candidate_evidence={})['accepted']
    assert not f.assess(better,rows([1.1,.05]),required_evidence={},candidate_evidence={})['accepted']


def test_actual_contact_and_missing_rows_are_not_geometry_fallbacks():
    f=FeasibilityFilter([ProtectionRule('keep',.1)])
    assert not f.assess(rows([0.,0.]),rows([0.,0.]),required_evidence={'contact':True},candidate_evidence={})['accepted']
    assert not f.assess(rows([0.,0.]),[],required_evidence={},candidate_evidence={})['accepted']
    assert not f.assess([],rows([1.,0.]),required_evidence={},candidate_evidence={})['accepted']


def test_signed_inequality_accepts_slack_and_invalid_inputs_fail():
    f=FeasibilityFilter([ProtectionRule('keep',0.)])
    assert f.assess(rows([-1.,-1.],'le'),rows([-.2,-2.],'le'),required_evidence={},candidate_evidence={})['accepted']
    with pytest.raises(ValueError):ProtectionRule('keep',-1.)
    with pytest.raises(ValueError):f.assess(rows([0.,0.]),rows([float('nan'),0.]),required_evidence={},candidate_evidence={})
