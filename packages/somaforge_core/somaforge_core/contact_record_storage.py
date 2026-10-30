"""Lossless contact-buffer compaction; does not filter contact candidates."""
import numpy as np


def compact_snapshot(snapshot):
    counts=np.asarray(snapshot['count'])
    if counts.size!=1 or not np.isfinite(counts).all():
        raise ValueError('Expected scalar contact count')
    count=int(counts.reshape(-1)[0])
    if count<0 or count!=counts.reshape(-1)[0]:
        raise ValueError('Invalid contact count')
    result={'count':counts.copy()}
    for name,value in snapshot.items():
        if name=='count':continue
        a=np.asarray(value)
        if a.ndim<1 or len(a)<count:
            raise ValueError(f'Contact count exceeds {name} capacity')
        result[name]=a[:count].copy()  # own allocation: release unused capacity
    return result


def stack_contact_channel(values):
    if not values:raise ValueError('No contact samples')
    shape=values[0].shape[1:];dtype=values[0].dtype
    if any(v.shape[1:]!=shape or v.dtype!=dtype for v in values):
        raise ValueError('Contact channel schema changed')
    width=max(len(v) for v in values)
    result=np.zeros((len(values),width,*shape),dtype=dtype)
    for i,v in enumerate(values):result[i,:len(v)]=v
    return result
