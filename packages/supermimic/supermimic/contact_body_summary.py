"""Passive summaries of every native robot-terrain contact, outside task-part labels."""
import numpy as np


def summarize(snapshot, body_world, body_slot, terrain_shapes, world_count, body_count, observed):
    n=int(snapshot['count'])
    active=np.asarray(snapshot['active'][:n],bool)
    allocated=np.asarray(snapshot['constraint_allocated'][:n],bool)
    if np.any(active & ~allocated):raise ValueError('Unallocated active native body contact')
    worlds=np.asarray(snapshot['worldid'][:n],int)
    b0=np.asarray(snapshot['body0'][:n],int);b1=np.asarray(snapshot['body1'][:n],int)
    def robot(b):
        valid=(b>=0)&(b<len(body_slot));safe=np.clip(b,0,len(body_slot)-1)
        return valid & (body_slot[safe]>=0),safe
    r0,s0=robot(b0);r1,s1=robot(b1)
    side=np.where(r0,0,1);body=np.where(r0,s0,s1)
    outside=np.where(r0,snapshot['shape1'][:n],snapshot['shape0'][:n])
    selected=active & (r0 ^ r1) & np.isin(outside,terrain_shapes)
    selected &= (worlds>=0)&(worlds<world_count)&(body_world[body]==worlds)
    selected &= observed[np.clip(worlds,0,world_count-1)]
    rows=np.flatnonzero(selected);world=worlds[rows];slot=body_slot[body[rows]]
    shape=(world_count,body_count)
    count=np.zeros(shape,np.int64);force=np.zeros((*shape,3),np.float32)
    point=np.zeros((*shape,3),np.float32);depth=np.zeros(shape,np.float32)
    if len(rows):
        normals=np.asarray(snapshot['frame_w'])[rows,0]
        if not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals,axis=-1),1.,atol=1e-6):
            raise ValueError('Invalid native normal in all-body contact summary')
        values=np.asarray(snapshot['force_on_body1_w'])[rows]*np.where(side[rows]==1,1.,-1.)[:,None]
        positions=np.asarray(snapshot['position_w'])[rows]
        distances=np.asarray(snapshot['dist'])[rows]
        if not all(np.isfinite(x).all() for x in (values,positions,distances)):
            raise ValueError('Nonfinite native body contact evidence')
        np.add.at(count,(world,slot),1);np.add.at(force,(world,slot),values)
        np.add.at(point,(world,slot),positions)
        np.maximum.at(depth,(world,slot),np.maximum(-distances,0))
    point/=np.maximum(count,1)[...,None]
    return dict(body_contact_count=count,body_contact_active=count>0,body_contact_force=force,
                body_contact_point=point,body_contact_penetration=depth,
                body_contact_known=np.broadcast_to(observed[:,None],shape).copy())
