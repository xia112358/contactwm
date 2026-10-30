"""Explicit six-part permutation at the published PHP tracker boundary."""
import torch

PLANNER_PARTS = ('left_foot','right_foot','left_knee','right_knee','left_hand','right_hand')
PHP_PARTS = ('left_foot','right_foot','left_hand','right_hand','left_knee','right_knee')


def reorder(value, source, target, axis=1):
    if len(source)!=len(set(source)) or set(source)!=set(target) or len(source)!=len(target):
        raise ValueError('Contact schemas must contain the same unique names')
    if value.shape[axis]!=len(source):raise ValueError('Contact tensor does not match schema')
    return value.index_select(axis,torch.tensor([source.index(n) for n in target],device=value.device))


def tracker_contacts(prediction):
    p=reorder(prediction['positions'],PLANNER_PARTS,PHP_PARTS)
    active=reorder(prediction['occupancy'],PLANNER_PARTS,PHP_PARTS)
    return torch.cat((torch.where(active[...,None],p,0).flatten(1),active.float()),-1)
