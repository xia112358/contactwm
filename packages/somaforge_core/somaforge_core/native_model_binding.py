"""Read-only inspection and surface binding for an initialized Newton model.

These helpers consume the realized model arrays. They do not import Newton,
initialize a simulator or modify collision settings. SciPy is needed only when
inspecting or transforming mesh geometry.
"""
from __future__ import annotations

import hashlib
import json


def inspect_model(model):
    """Fingerprint realized collision settings, geometry and joint mapping."""
    import numpy as np
    digest = hashlib.sha256()
    result = dict(body_labels=list(model.body_label), shape_labels=list(model.shape_label),
                  joint_labels=list(model.joint_label))
    for key in ('shape_body', 'shape_type', 'shape_transform', 'shape_scale', 'shape_flags',
                'shape_margin', 'shape_gap', 'shape_collision_group', 'joint_q_start', 'joint_type',
                'joint_parent', 'joint_child', 'joint_X_p', 'joint_X_c'):
        value = getattr(model, key, None)
        if value is None:
            raise ValueError(f'Cannot fingerprint missing Newton model field {key}')
        array = value.numpy()
        digest.update(key.encode()); digest.update(array.tobytes())
        result[key] = array.tolist()
    digest.update(json.dumps(result, sort_keys=True).encode())
    digest.update(repr(sorted(model.shape_collision_filter_pairs)).encode())
    for source in model.shape_source:
        if source is not None and hasattr(source, 'vertices'):
            digest.update(np.asarray(source.vertices).tobytes())
            digest.update(np.asarray(source.indices).tobytes())
    result['model_fingerprint'] = digest.hexdigest()
    result['world_count'] = int(model.world_count)
    for key in ('body_world', 'shape_world', 'joint_world'):
        value = getattr(model, key, None)
        if value is not None:
            result[key] = value.numpy().tolist()
    from scipy.spatial.transform import Rotation
    result['terrain_mesh_bounds'] = []
    for i, source in enumerate(model.shape_source):
        if int(result['shape_body'][i]) >= 0 or source is None or not hasattr(source, 'vertices'):
            continue
        transform = np.asarray(result['shape_transform'][i])
        vertices = np.asarray(source.vertices)*result['shape_scale'][i]
        vertices = Rotation.from_quat(transform[3:]).apply(vertices)+transform[:3]
        result['terrain_mesh_bounds'].append(dict(shape=i, minimum=vertices.min(0).tolist(),
            maximum=vertices.max(0).tolist(), z_levels=np.unique(np.round(vertices[:,2],5)).tolist()))
    return result


def native_binding(model, info):
    """Surface planes from actual static triangle meshes, not a second model."""
    import numpy as np
    import re
    from scipy.spatial.transform import Rotation
    body_env = {i: int(info.get('body_world', [])[i]) if 'body_world' in info else int(match[1]) for i,label in enumerate(model.body_label)
                if (match := re.search(r'/env_(\d+)/Robot/',label))}
    if set(body_env.values()) != set(range(max(1, int(model.world_count)))):
        raise ValueError('Native worker requires one canonical robot in each Newton world')
    planes_by_shape = {}; all_planes = set(); triangles_by_shape = {}; indices_by_shape = {}
    for i,body in enumerate(info['shape_body']):
        if body in body_env:
            continue
        source = model.shape_source[i]
        if body >= 0 or source is None or not hasattr(source,'vertices'):
            raise ValueError('Native catalog currently requires static triangle-mesh terrain')
        transform=np.asarray(info['shape_transform'][i])
        v=Rotation.from_quat(transform[3:]).apply(np.asarray(source.vertices)*info['shape_scale'][i])+transform[:3]
        triangles=v[np.asarray(source.indices).reshape(-1,3)]
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        length=np.linalg.norm(normal,axis=-1);valid=length>1e-10
        normal=normal[valid]/length[valid,None]
        offset=(normal*triangles[valid,0]).sum(-1)
        keys=[tuple(np.round(np.r_[n,d],8)) for n,d in zip(normal,offset)]
        planes=set(keys)
        grouped={p:[] for p in planes}
        grouped_indices={p:[] for p in planes}
        for plane,triangle,index in zip(keys,triangles[valid],np.flatnonzero(valid)):
            grouped[plane].append(triangle.tolist());grouped_indices[plane].append(int(index))
        triangles_by_shape[i]=grouped
        indices_by_shape[i]=grouped_indices
        planes_by_shape[i]=planes;all_planes.update(planes)
    # Stable ground/top/other ordering. Multiple top heights stay distinct.
    ordered=sorted(all_planes,key=lambda p:(0 if p==(0.,0.,1.,0.) else 1 if p[:3]==(0.,0.,1.) else 2,p))
    ids={plane:i for i,plane in enumerate(ordered)}
    return dict(model_fingerprint=info['model_fingerprint'],scene=None,
        body_env=body_env,shape_surface={i:[dict(surface=ids[p],normal_w=p[:3],plane_offset=p[3],
            triangles_w=triangles_by_shape[i][p],triangle_indices=indices_by_shape[i][p],
            attribution_schema='native_triangle_nearest_v1')
            for p in sorted(planes)] for i,planes in planes_by_shape.items()},
        surface_catalog=[dict(surface=ids[p],normal_w=p[:3],plane_offset=p[3]) for p in ordered])
