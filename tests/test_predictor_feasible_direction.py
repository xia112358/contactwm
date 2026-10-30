import torch
from contact_solver.research.feasible_direction import constrained_direction


def test_boundary_direction_preserves_constraint_and_improves_task():
    g=torch.tensor([1.,1.]);a=torch.tensor([[1.,0.]])
    d,report=constrained_direction(g,a,torch.tensor([0.]),learning_rate=.1)
    assert d[0]>=-1e-7 and d[1]<0 and torch.dot(g,d)<0


def test_violated_constraint_gets_recovery_component_without_erasing_task():
    d,_=constrained_direction(torch.tensor([1.,1.]),torch.tensor([[1.,0.]]),torch.tensor([-.1]),learning_rate=.1)
    assert d[0]>=.019999 and d[1]<0
