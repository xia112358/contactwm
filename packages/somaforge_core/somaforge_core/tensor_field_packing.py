"""Lossless grouped tensor transport; consumers own cloned buffer storage."""
import math
import torch

SCHEMA='typed_tensor_field_buffers_v1'


def pack_fields(fields):
    groups={};description=[]
    for name,value in fields.items():
        key=(value.device,value.dtype)
        if key not in groups:groups[key]=[]
        groups[key].append((name,value))
    buffers=[]
    for group,values in enumerate(groups.values()):
        offset=0
        for name,value in values:
            description.append((name,group,tuple(value.shape),offset,value.numel()))
            offset+=value.numel()
        buffers.append(torch.cat([value.reshape(-1) for _,value in values]))
    return dict(schema=SCHEMA,buffers=buffers,fields=description)


def clone_fields(packed):
    if packed.get('schema')!=SCHEMA:raise ValueError('Unknown packed tensor schema')
    buffers=[value.clone() for value in packed['buffers']]
    if any(value.ndim!=1 for value in buffers):raise ValueError('Packed buffers must be vectors')
    fields={};offsets=[0]*len(buffers)
    for name,group,shape,offset,count in packed['fields']:
        if name in fields or not 0<=group<len(buffers):raise ValueError('Invalid packed field identity')
        if offset!=offsets[group] or count!=math.prod(shape) or count<0 or offset+count>buffers[group].numel():
            raise ValueError('Invalid packed field extent')
        fields[name]=buffers[group][offset:offset+count].reshape(shape)
        offsets[group]+=count
    if any(offset!=value.numel() for offset,value in zip(offsets,buffers)):
        raise ValueError('Unaccounted packed field data')
    return fields
