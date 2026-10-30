"""Passive capture of the source key already written by Newton's narrow phase.

Does not enable deterministic sorting, modify geometry, or select contacts.
The currently validated path is unsorted, without contact matching. Other
layouts are rejected until their permutations have an audited mapping.
"""
import numpy as np
from .contact_source_geometry import (SOURCE_NORMAL_FAN_SCHEMA, project_source_triangle,
                                      incident_witness_faces)

SCHEMA = 'newton_narrow_phase_same_pass_source_v2'


def capture_constraint_snapshot(pipeline,state,contacts,solver,snapshot,capture=None):
    """One collision pass owns BOTH physical contacts and source metadata.

Unsorted Newton can produce distinct valid manifolds on repeated identical q.
Cross-run bit equality is not an instrumentation integrity test. Instead verify
the raw-to-solver mapping from this pass and that attaching metadata does not
mutate any physical snapshot value. No sorting/matching or activation changes.
"""
    if capture is None:
        pipeline.collide(state,contacts)
        return snapshot()
    keys=capture.collide(state,contacts)
    raw=snapshot()
    before=physical_contact_signature(raw)
    attach_solver_sources(raw,solver,contacts,keys)
    if physical_contact_signature(raw)!=before:
        raise ValueError('Source metadata attachment changed physical snapshot')
    raw['same_pass_source_mapping_verified']=True
    if hasattr(capture,'encoding'):
        raw['source_key_encoding']=dict(capture.encoding)
    return raw


def physical_contact_signature(snapshot):
    """Order-independent exact physical values, excluding label/row ordering.

Addresses may permute with atomic contact insertion; allocation validity and
required constraint dimensions must not. No tolerance hides changed geometry.
"""
    import hashlib
    import json
    rows=[]
    for i in range(int(snapshot['count'])):
        row=[]
        for name in ('worldid','shape0','shape1','body0','body1','type','dist','includemargin',
                     'active','constraint_allocated','constraint_rows','position_w','frame_w',
                     'geometry_point0_w','geometry_point1_w'):
            row.append(np.asarray(snapshot[name][i]).tolist())
        rows.append(json.dumps(row,separators=(',',':')))
    return hashlib.sha256('\n'.join(sorted(rows)).encode()).hexdigest()


def decode_mesh_triangle_key(key, shape0, shape1):
    key = int(key)
    if key < 0:raise ValueError('Newton did not write a contact source key')
    a, b, sub = (key >> 43) & 0xfffff, (key >> 23) & 0xfffff, key & 0x7fffff
    if (a,b) != (int(shape0),int(shape1)):
        raise ValueError('Newton source key/raw shape mapping mismatch')
    # narrow_phase mesh triangle: (tri_idx << 1) | 1;
    # convex single/multi contact: template << 3 | manifold_point.
    # Only call the triangle interpretation for a verified mesh-triangle path.
    return dict(key_hex=hex(key),shape0=a,shape1=b,sub_key=sub,
                triangle_index=(sub >> 4) if sub & 8 else None,
                manifold_point=sub & 7)


def canonical_source_keys(keys, shape_bits, sub_key_bits):
    """Lossless versioned layout conversion; no geometry/contact inference."""
    keys = np.asarray(keys, dtype=np.int64)
    if not (1 <= shape_bits <= 20 and 1 <= sub_key_bits <= 23):
        raise ValueError('Unsupported Newton contact source-key layout')
    if np.any(keys < 0) or np.any(keys >> (sub_key_bits + 2*shape_bits)):
        raise ValueError('Newton source key exceeds declared layout')
    mask = (1 << shape_bits)-1
    a = (keys >> (sub_key_bits+shape_bits)) & mask
    b = (keys >> sub_key_bits) & mask
    sub = keys & ((1 << sub_key_bits)-1)
    return (a << 43) | (b << 23) | sub


def expand_native_sphere_triangle_keys(keys, shape_types):
    """Main's reduced mesh/sphere analytic path omits the manifold <<3."""
    keys = np.asarray(keys,dtype=np.int64).copy()
    a = (keys >> 43) & 0xfffff
    b = (keys >> 23) & 0xfffff
    types = np.asarray(shape_types)
    mask = (types[a] == 8) & (types[b] == 3)
    sub = keys[mask] & 0x7fffff
    if np.any((sub & 1) != 1) or np.any(sub >= (1 << 20)):
        raise ValueError('Unsupported native analytic mesh/sphere source encoding')
    keys[mask] = (keys[mask] & ~np.int64(0x7fffff)) | (sub << 3)
    return keys


class PassiveContactSourceCapture:
    def __init__(self, pipeline, contacts, *, model=None):
        import warp as wp
        if pipeline.deterministic or getattr(pipeline,'_contact_matcher',None) is not None:
            raise ValueError('Source capture needs an audited unsorted/unmatched collision pipeline')
        if not hasattr(pipeline,'_sort_key_array') or pipeline._sort_key_array.shape[0] != 0:
            raise ValueError('Unexpected existing Newton source-key buffer')
        self.pipeline = pipeline
        from .newton_collision_compat import SCHEMA as numeric_schema, NATIVE_SCHEMA
        if numeric_schema == NATIVE_SCHEMA:
            # Read the writer's actual widths, never infer them from row IDs.
            self.shape_bits = int(pipeline._contact_sort_shape_index_bits)
            self.sub_key_bits = int(pipeline._contact_sort_sub_key_bits)
            if model is None:
                raise ValueError('Native source capture requires actual model shape types')
            self.analytic_sphere_types = (model.shape_type.numpy()
                if pipeline.narrow_phase.reduce_contacts else None)
        else:
            self.shape_bits, self.sub_key_bits = 20, 23
            self.analytic_sphere_types = None
        self.encoding = dict(raw_shape_bits=self.shape_bits, raw_sub_key_bits=self.sub_key_bits,
                             stored_shape_bits=20, stored_sub_key_bits=23,
                             schema='newton_source_key_lossless_layout_v1')
        self.encoding['native_analytic_sphere_subkey_expanded'] = self.analytic_sphere_types is not None
        self.keys = wp.zeros(contacts.rigid_contact_max,dtype=wp.int64,device=pipeline.device)

    def collide(self, state, contacts):
        self.collide_device(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count > len(self.keys):raise RuntimeError('Source capture contact buffer overflow')
        result = self.keys.numpy()[:count].copy()
        if np.any(result < 0):raise ValueError('Missing narrow-phase contact source keys')
        result = canonical_source_keys(result,self.shape_bits,self.sub_key_bits)
        if self.analytic_sphere_types is not None:
            result = expand_native_sphere_triangle_keys(result,self.analytic_sphere_types)
        return result

    def collide_device(self, state, contacts):
        """Capture native keys on device; validity is checked by the reader."""
        original = self.pipeline._sort_key_array
        if original.shape[0] != 0 or self.pipeline.deterministic:
            raise ValueError('Collision source capture configuration changed')
        self.keys.fill_(-1)
        self.pipeline._sort_key_array = self.keys
        try:
            self.pipeline.collide(state,contacts)
        finally:
            self.pipeline._sort_key_array = original
        return self.keys


def attach_solver_sources(snapshot, solver, contacts, keys):
    """Match source keys to solver rows via the actual raw-to-solver mapping."""
    count = int(snapshot['count'])
    cid = solver._contact_tid_to_cid.numpy()[:len(keys)]
    output = np.full(len(snapshot['dist']),-1,dtype=np.int64)
    shape0, shape1 = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
    for tid,row in enumerate(cid):
        if row < 0:continue
        if row >= count or output[row] != -1:raise ValueError('Non-unique raw-to-solver source mapping')
        decode_mesh_triangle_key(keys[tid],shape0[tid],shape1[tid])
        if (shape0[tid],shape1[tid]) != (snapshot['shape0'][row],snapshot['shape1'][row]):
            raise ValueError('Contact source/solver shape mismatch')
        output[row] = keys[tid]
    if np.any(output[:count]<0):raise ValueError('Missing solver contact sources')
    snapshot['source_key'] = output


def source_face_metadata(key, terrain_shape, shape0, shape1, binding, normal=None, terrain_point=None):
    decoded = decode_mesh_triangle_key(key,shape0,shape1)
    result = dict(schema=SCHEMA,**decoded,terrain_shape=int(terrain_shape),
                  source_surface=None,status='unsupported_source_path')
    # Current mesh-triangle narrow phase always places that mesh on side A.
    if decoded['shape0'] != int(terrain_shape) or decoded['triangle_index'] is None:return result
    if not all('triangle_indices' in face for face in binding):
        raise ValueError('Source attribution requires original mesh triangle indices')
    matches = [face for face in binding if decoded['triangle_index'] in face['triangle_indices']]
    if len(matches) != 1:raise ValueError('Source triangle has no unique actual face mapping')
    face = matches[0];index = face['triangle_indices'].index(decoded['triangle_index'])
    triangle = np.asarray(face['triangles_w'][index])
    # Record incident faces by exact shared mesh vertices. This is topology,
    # not a claim that every adjacent face is currently touched.
    vertices = {tuple(v) for v in triangle}
    adjacent = []
    for other in binding:
        if other['surface']==face['surface']:continue
        if any(len(vertices & {tuple(v) for v in t})>=2 for t in other['triangles_w']):
            adjacent.append(int(other['surface']))
    result.update(status='mesh_triangle',source_surface=int(face['surface']),
                  source_triangle_w=triangle.tolist(),edge_adjacent_surfaces=sorted(set(adjacent)))
    if normal is not None:
        result.update(source_normal_fan(triangle,face,binding,normal,terrain_point))
    return result


def source_normal_fan(triangle,source_face,binding,normal,terrain_point):
    """Task-face ownership from a source triangle's local normal fan.

Support vertices identify candidate incident mesh faces. Only faces containing
the actual source witness feature may compete by normal alignment. A side
interior cannot acquire a remote top because its triangle has a top vertex.
Raw witnesses, activation and allocation remain unchanged.
"""
    triangle=np.asarray(triangle,np.float64);normal=np.asarray(normal,np.float64)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)==0:
        raise ValueError('Missing/invalid Newton source normal')
    if terrain_point is None:
        raise ValueError('Missing actual source-triangle terrain witness')
    projected = project_source_triangle(terrain_point, triangle)
    score=(triangle-triangle[0])@normal
    support={tuple(v) for v in triangle[score==score.max()]}
    candidates=[source_face]
    for face in binding:
        if face['surface']==source_face['surface']:continue
        if any(support & {tuple(v) for v in t} for t in face['triangles_w']):candidates.append(face)
    normals = np.asarray([face['normal_w'] for face in candidates])
    vertices = [np.asarray(face['triangles_w']).reshape(-1, 3) for face in candidates]
    offsets = np.asarray([v[0]@n for v,n in zip(vertices,normals)])
    extents = np.asarray([np.ptp(v,axis=0).max() for v in vertices])
    candidates = [face for face,keep in zip(candidates,incident_witness_faces(projected,normals,offsets,extents)) if keep]
    if not candidates:
        raise ValueError('Actual source witness belongs to no incident native face')
    candidates.sort(key=lambda face:(-float(np.dot(normal,face['normal_w'])),int(face['surface'])))
    return dict(normal_fan_schema=SOURCE_NORMAL_FAN_SCHEMA,
        primary_surface=int(candidates[0]['surface']),
        incident_surface_candidates=[int(face['surface']) for face in candidates],
        support_vertices_w=[list(v) for v in sorted(support)],
        source_feature_projection_w=projected.tolist())
