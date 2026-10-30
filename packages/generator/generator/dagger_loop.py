"""Auditable state invariants for an iterative dataset-aggregation loop."""
import hashlib
import torch


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def append_dataset(previous, new):
    if previous is None:
        return [{k:v.detach().clone() for k,v in group.items()} for group in new]
    if len(previous) != len(new): raise ValueError('Dataset group mismatch')
    merged=[]
    for old, incoming in zip(previous,new):
        if old.keys() != incoming.keys(): raise ValueError('Dataset field mismatch')
        merged.append({k:torch.cat((old[k],incoming[k].detach()),0) for k in old})
    return merged


def verify_iteration(previous, collection_hash, optimizer_step, dataset_size):
    if previous is None: return
    if collection_hash != previous['updated_model_sha256']:
        raise ValueError('DAgger collected with a stale model')
    if optimizer_step != previous['optimizer_step_end']:
        raise ValueError('DAgger optimizer was reset between rounds')
    if dataset_size != previous['dataset_size_after']:
        raise ValueError('DAgger dropped previous labels')
