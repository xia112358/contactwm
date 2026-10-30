"""Newton/MJWarp contact truth shared by recording, labels and query clients.

No force or geometric-proximity fallback. Intentions never gate observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

from .contact_schema import CONTACT_BODY_NAMES_BY_PART

SCHEMA = 'newton_mjwarp_constraint_snapshot_v1'
PARTS = ('left_foot', 'right_foot', 'left_hand', 'right_hand', 'left_knee', 'right_knee')
FIELDS = ('count', 'active', 'constraint_allocated', 'dist', 'includemargin',
          'type', 'worldid', 'shape0', 'shape1', 'body0', 'body1', 'position_w', 'efc_address', 'frame_w', 'constraint_rows')


def constraint_decision(dist, includemargin, kind, address, count, row_count=None):
    dist, includemargin, kind, address = map(np.asarray, (dist, includemargin, kind, address))
    if dist.ndim != 1 or includemargin.shape != dist.shape or kind.shape != dist.shape:
        raise ValueError('Invalid Newton contact field shapes')
    if address.ndim != 2 or address.shape[0] != len(dist) or address.shape[1] < 1:
        raise ValueError('Invalid Newton constraint address shape')
    if int(count) != count or not 0 <= count <= len(dist):
        raise ValueError('Invalid Newton contact count/capacity')
    if not np.isfinite(dist[:count]).all() or not np.isfinite(includemargin[:count]).all():
        raise ValueError('Nonfinite Newton contact distance/margin')
    active = (np.arange(len(dist)) < count) & ((kind & 1) != 0) & (dist < includemargin)
    rows = np.ones(len(dist),int) if row_count is None else np.asarray(row_count)
    if rows.shape != dist.shape or np.any(rows[:count]<1) or np.any(rows[:count]>address.shape[1]):
        raise ValueError('Invalid expected contact constraint row count')
    required = np.arange(address.shape[1])[None] < rows[:,None]
    return active, active & ((address >= 0) | ~required).all(-1)


def snapshot_solver_contacts(solver, model, *, state=None, contacts=None, static_mapping_cache=None,
                             include_support=False):
    contact = solver.mjw_data.contact
    values = {k: np.asarray(getattr(contact, attr).numpy()).copy() for k, attr in
              [('dist', 'dist'), ('includemargin', 'includemargin'), ('type', 'type'),
               ('efc_address', 'efc_address'), ('worldid', 'worldid'), ('position_w', 'pos'), ('frame_w', 'frame')]}
    count = int(solver.mjw_data.nacon.numpy().reshape(-1)[0])
    dim = contact.dim.numpy()
    cone = int(solver.mjw_model.opt.cone)
    if cone not in (0,1):
        raise ValueError('Unknown MJWarp contact cone')
    values['constraint_rows'] = np.where(dim == 1,1,2*(dim-1)) if cone == 0 else dim.copy()
    active, allocated = constraint_decision(values['dist'], values['includemargin'],
                                           values['type'], values['efc_address'], count, values['constraint_rows'])
    geom = contact.geom.numpy()
    # Only dedicated immutable-scene queries opt in. Dynamic simulation callers
    # retain fresh reads; margins and all contact state are always read afresh.
    if static_mapping_cache is None:
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        shape_body = model.shape_body.numpy()
    else:
        owner=(id(model),id(solver),id(solver.mjc_geom_to_newton_shape),id(model.shape_body))
        if static_mapping_cache.get('owner') != owner:
            static_mapping_cache.clear()
            static_mapping_cache.update(owner=owner,
                mapping=solver.mjc_geom_to_newton_shape.numpy().copy(),
                shape_body=model.shape_body.numpy().copy())
        mapping=static_mapping_cache['mapping'];shape_body=static_mapping_cache['shape_body']
    shapes = np.full((len(active), 2), -1, dtype=np.int32)
    bodies = shapes.copy()
    for row in range(count):
        for side in range(2):
            g, w = int(geom[row, side]), int(values['worldid'][row])
            if not (0 <= w < mapping.shape[0] and 0 <= g < mapping.shape[1]):
                raise ValueError('Invalid solver contact world/geom mapping')
            shape = int(mapping[w, g])
            if not 0 <= shape < len(shape_body):
                raise ValueError('Invalid solver contact shape mapping')
            shapes[row, side], bodies[row, side] = shape, shape_body[shape]
    if state is not None and contacts is not None:
        # Exact raw -> solver correspondence. Recover each geometry point with
        # the same margin stripping as Newton's conversion kernel, not by
        # projecting its midpoint along a normal (wrong at edges/corners).
        from scipy.spatial.transform import Rotation
        raw_count = int(contacts.rigid_contact_count.numpy()[0])
        cid = solver._contact_tid_to_cid.numpy()[:raw_count]
        transforms = state.body_q.numpy().reshape(-1,7)
        margins = model.shape_margin.numpy()
        covered = np.zeros(count, bool)
        for side in (0,1):
            points = np.full((len(active),3), np.nan, np.float32)
            raw_points = getattr(contacts,f'rigid_contact_point{side}').numpy()
            raw_offsets = getattr(contacts,f'rigid_contact_offset{side}').numpy()
            raw_margins = getattr(contacts,f'rigid_contact_margin{side}').numpy()
            raw_shapes = getattr(contacts,f'rigid_contact_shape{side}').numpy()
            for tid, row in enumerate(cid):
                if row < 0:
                    continue
                if row >= count or raw_shapes[tid] != shapes[row,side]:
                    raise ValueError('Raw/solver contact correspondence mismatch')
                shape = int(raw_shapes[tid]); body = int(shape_body[shape])
                scale = (raw_margins[tid]-margins[shape])/raw_margins[tid] if raw_margins[tid] != 0 else 0.
                point = raw_points[tid]+raw_offsets[tid]*scale
                if body >= 0:
                    point = Rotation.from_quat(transforms[body,3:]).apply(point)+transforms[body,:3]
                points[row] = point; covered[row] = True
            values[f'geometry_point{side}_w'] = points
        if not covered.all():
            raise ValueError('Missing raw pair for solver contact')
    if include_support:
        from .newton_support import snapshot_support
        values.update(snapshot_support(solver,dict(values,count=count,body0=bodies[:,0],body1=bodies[:,1])))
    return dict(values, count=np.asarray(count, dtype=np.int32), active=active,
                constraint_allocated=allocated, shape0=shapes[:, 0], shape1=shapes[:, 1],
                body0=bodies[:, 0], body1=bodies[:, 1])


@dataclass
class PartContacts:
    active: np.ndarray
    unallocated: np.ndarray
    position_w: np.ndarray
    surface: np.ndarray
    # Preserve every active pair; a six-part view must not erase multi-surface contact.
    pairs: list[dict]
    candidate_pairs: list[dict] = field(default_factory=list)


def triangle_distance_squared(point, triangles):
    """Exact point-to-finite-triangle distance; only for surface attribution.

    This never defines activation, never moves a saved contact point, and never
    substitutes an XY box or an infinite plane for a bounded physical face.
    """
    t=np.asarray(triangles,np.float64)
    if t.ndim!=3 or t.shape[1:]!=(3,3) or len(t)==0 or not np.isfinite(t).all():
        raise ValueError('Missing/invalid actual triangle geometry')
    a,b,c=t[:,0],t[:,1],t[:,2];u=b-a;v=c-a;w=np.asarray(point)-a
    uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
    wu=np.einsum('ij,ij->i',w,u);wv=np.einsum('ij,ij->i',w,v);den=uu*vv-uv*uv
    if (den<=0).any():raise ValueError('Degenerate triangle in contact surface binding')
    s=(wu*vv-wv*uv)/den;r=(wv*uu-wu*uv)/den
    projection=a+s[:,None]*u+r[:,None]*v
    distance=np.where((s>=0)&(r>=0)&(s+r<=1),np.sum((projection-point)**2,axis=-1),np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;parameter=np.sum((point-start)*edge,axis=-1)/np.sum(edge*edge,axis=-1)
        nearest=start+np.clip(parameter,0,1)[:,None]*edge
        distance=np.minimum(distance,np.sum((nearest-point)**2,axis=-1))
    return distance


def actual_contact_faces(point, normal, binding):
    if any('triangles_w' in face for face in binding):
        if not all('triangles_w' in face for face in binding):raise ValueError('Incomplete actual face geometry')
        distances=np.array([triangle_distance_squared(point,face['triangles_w']).min() for face in binding])
        # Numerical tie handling, not a contact-distance gate. Normal determines
        # the primary face only among equally nearest finite physical faces.
        tied=np.isclose(distances,distances.min(),rtol=1e-9,atol=1e-12)
        matching=[face for face,keep in zip(binding,tied) if keep]
    else:
        # Legacy plane bindings work only for exact unambiguous witnesses.
        # Off-plane observations require an upgraded native triangle binding.
        matching=[face for face in binding if ('plane_offset' not in face or np.isclose(
            np.dot(point,face['normal_w']),face['plane_offset'],atol=1e-4,rtol=0))]
    if not matching:raise ValueError(f'Actual contact face absent: point={point.tolist()}, normal={normal.tolist()}')
    return sorted(matching,key=lambda face:(-float(np.dot(normal,face['normal_w'])),int(face['surface'])))


def actual_primary_face_indices(points, normals, binding, *, batch_size=256):
    """Batch the same finite-triangle attribution and normal/ID tie ordering.

    Contact activation is already known. This performs no distance gating,
    projection, candidate promotion or geometry modification.
    """
    points, normals = np.asarray(points, np.float64), np.asarray(normals, np.float64)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3 or batch_size < 1:
        raise ValueError('Invalid batched contact face inputs')
    if not binding or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Missing binding or nonfinite contact geometry')
    if not all('triangles_w' in face for face in binding):
        return np.asarray([next(i for i,f in enumerate(binding) if f is actual_contact_faces(p,n,binding)[0])
                           for p,n in zip(points,normals)],np.int64)
    geometry=[]
    for face in binding:
        triangles=np.asarray(face['triangles_w'],np.float64)
        if triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or not len(triangles) or not np.isfinite(triangles).all():
            raise ValueError('Missing/invalid actual triangle geometry')
        a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
        u,v=b-a,c-a
        uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
        den=uu*vv-uv*uv
        if (den<=0).any():raise ValueError('Degenerate triangle in contact surface binding')
        geometry.append((a,b,c,u,v,uu,uv,vv,den))
    face_normals=np.asarray([face['normal_w'] for face in binding],float)
    id_order=np.argsort([int(face['surface']) for face in binding],kind='stable')
    output=[]
    for start in range(0,len(points),batch_size):
        p=points[start:start+batch_size]; distances=[]
        for a,b,c,u,v,uu,uv,vv,den in geometry:
            w=p[:,None]-a
            wu=np.einsum('ptj,tj->pt',w,u);wv=np.einsum('ptj,tj->pt',w,v)
            s=(wu*vv-wv*uv)/den;r=(wv*uu-wu*uv)/den
            projection=a+s[...,None]*u+r[...,None]*v
            distance=np.where((s>=0)&(r>=0)&(s+r<=1),np.sum((projection-p[:,None])**2,axis=-1),np.inf)
            for left,right in ((a,b),(b,c),(c,a)):
                edge=right-left
                parameter=np.sum((p[:,None]-left)*edge,axis=-1)/np.sum(edge*edge,axis=-1)
                nearest=left+np.clip(parameter,0,1)[...,None]*edge
                distance=np.minimum(distance,np.sum((nearest-p[:,None])**2,axis=-1))
            distances.append(distance.min(-1))
        distances=np.stack(distances,-1)
        tied=np.isclose(distances,distances.min(-1,keepdims=True),rtol=1e-9,atol=1e-12)
        dot=np.sum(normals[start:start+len(p),None]*face_normals[None],axis=-1)
        score=np.where(tied,dot,-np.inf)[:,id_order]
        output.append(id_order[score.argmax(-1)])
    return np.concatenate(output) if output else np.empty(0,np.int64)


def full_robot_separation(snapshot, *, body_env, shape_surface, env_id=0):
    """Signed Newton distances BEFORE six-part/task-face filtering.

Reports terrain and enabled self-collision pairs, including non-contact limbs.
Zero reported penetration is not a proof about collision-filtered pairs or
undetected deep mesh containment. No margin is subtracted from geometry dist.
"""
    result=dict(schema='newton_full_robot_signed_separation_v1',terrain_rows=0,self_rows=0,
                terrain_penetration_m=0.,self_penetration_m=0.,worst_terrain=None,worst_self=None,
                closest_terrain=None,closest_self=None,
                invalid_penetrating_witnesses=[],
                scope='solver-detected pairs; configured collision filtering unchanged')
    for row in range(int(snapshot['count'])):
        if int(snapshot['worldid'][row])!=env_id or not int(snapshot['type'][row])&1:continue
        bodies=[int(snapshot[f'body{s}'][row]) for s in (0,1)]
        sides=[s for s,b in enumerate(bodies) if b in body_env]
        if len(sides)==1 and body_env[bodies[sides[0]]]==env_id:
            if int(snapshot[f'shape{1-sides[0]}'][row]) not in shape_surface:
                raise ValueError('Unmapped external shape in full-robot separation')
            kind='terrain'
        elif len(sides)==2 and all(body_env[b]==env_id for b in bodies):kind='self'
        else:continue
        dist=float(snapshot['dist'][row])
        if not np.isfinite(dist):raise ValueError('Nonfinite full-robot separation')
        if snapshot['active'][row] and not snapshot['constraint_allocated'][row]:
            raise ValueError('Unallocated active full-robot constraint')
        result[kind+'_rows']+=1
        depth=max(-dist,0.)
        witness=dict(row=row,body0=bodies[0],body1=bodies[1],dist=dist,
                shape0=int(snapshot['shape0'][row]),shape1=int(snapshot['shape1'][row]))
        if all(k in snapshot for k in ('geometry_point0_w','geometry_point1_w','frame_w')):
                witness.update(point0_w=np.asarray(snapshot['geometry_point0_w'][row]).tolist(),
                    point1_w=np.asarray(snapshot['geometry_point1_w'][row]).tolist(),
                    normal_w=np.asarray(snapshot['frame_w'][row])[0].tolist())
        if depth > 0:
            normal=np.asarray(witness.get('normal_w', [np.nan]*3))
            if not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal),1,atol=1e-4,rtol=0):
                result['invalid_penetrating_witnesses'].append(dict(witness,kind=kind))
        if result['closest_'+kind] is None or dist<result['closest_'+kind]['dist']:
            result['closest_'+kind]=dict(witness)
        if depth>result[kind+'_penetration_m']:
            result[kind+'_penetration_m']=depth;result['worst_'+kind]=dict(witness)
    return result


def reduce_snapshot(snapshot, *, body_labels, body_env, shape_surface, env_id=0, parts=PARTS, include_candidates=False):
    """Use explicit mappings, not regex guesses about unknown terrain bodies.

    body_env: robot body ID -> environment ID, including non-contact robot parts.
    shape_surface: terrain shape ID -> surface ID. Unmapped external pairs error.
    position_w is the solver contact position, not a projected surface anchor.
    """
    count = int(snapshot['count'])
    actual, allocated = constraint_decision(snapshot['dist'], snapshot['includemargin'],
        snapshot['type'], snapshot['efc_address'], count, snapshot.get('constraint_rows'))
    for key, value in [('active', actual), ('constraint_allocated', allocated)]:
        if not np.array_equal(snapshot[key], value):
            raise ValueError(f'Inconsistent saved Newton {key}')
    active = np.zeros(len(parts), bool); failed = active.copy()
    position = np.zeros((len(parts), 3), np.float32)
    surfaces = np.full(len(parts), -1, np.int32)
    representatives = np.full(len(parts), np.inf)
    pairs = []
    candidate_pairs = []
    names = [str(x).rsplit('/', 1)[-1] for x in body_labels]
    selected_rows = ((np.arange(len(actual)) < int(snapshot['count'])) & ((np.asarray(snapshot['type']) & 1)!=0)) if include_candidates else actual
    for row in np.flatnonzero(selected_rows):
        if int(snapshot['worldid'][row]) != env_id:
            continue
        bodies = [int(snapshot[f'body{s}'][row]) for s in (0, 1)]
        if any(b < -1 or b >= len(names) for b in bodies):
            raise ValueError('Invalid solver body mapping')
        robot_sides = [s for s, b in enumerate(bodies) if b in body_env]
        if len(robot_sides) != 1:
            continue  # self/cross-robot or terrain/terrain
        side = robot_sides[0]; body = bodies[side]
        if body_env[body] != env_id:
            continue
        matched = [p for p, part in enumerate(parts) if names[body] in CONTACT_BODY_NAMES_BY_PART[part]]
        if not matched:
            continue
        shape = int(snapshot[f'shape{1-side}'][row])
        if shape not in shape_surface:
            raise ValueError(f'Unmapped external contact shape {shape}')
        binding = shape_surface[shape]
        normal = np.asarray(snapshot['frame_w'][row])[0] * (1 if side == 1 else -1)
        contact_source = None
        if isinstance(binding, list):
            # Classify the actual contact face by its outward terrain normal;
            # a mesh shape ID alone cannot distinguish box top from its sides.
            normal = np.asarray(snapshot['frame_w'][row])[0] * (1 if side == 1 else -1)
            terrain_point = np.asarray(snapshot[f'geometry_point{1-side}_w'][row])
            if 'source_key' in snapshot:
                from .newton_contact_sources import source_face_metadata
                contact_source = source_face_metadata(snapshot['source_key'][row],shape,
                    snapshot['shape0'][row],snapshot['shape1'][row],binding,normal,terrain_point)
                if contact_source['status'] != 'mesh_triangle':
                    raise ValueError('Unsupported actual contact source; no nearest-face fallback')
                by_surface = {int(face['surface']):face for face in binding}
                matching = [by_surface[s] for s in contact_source['incident_surface_candidates']]
                if not snapshot.get('same_pass_source_mapping_verified',False):
                    raise ValueError('Unverified same-pass contact source mapping')
                contact_source['same_pass_source_mapping_verified'] = True
            else:
                matching = actual_contact_faces(terrain_point, normal, binding)
            surface = int(matching[0]['surface'])
            surface_candidates = [int(face['surface']) for face in matching]
        else:
            surface = int(binding)  # recording physical-shape IDs, not top classes
            surface_candidates = [surface]
        pos = np.asarray(snapshot.get(f'geometry_point{side}_w', snapshot['position_w'])[row])
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError('Invalid solver contact position')
        for p in matched:
            pair=dict(part=p, body=body, surface=surface, terrain_shape=shape,
                surface_candidates=surface_candidates,
                robot_shape=int(snapshot[f'shape{side}'][row]), allocated=bool(allocated[row]),
                position_w=pos.tolist(), dist=float(snapshot['dist'][row]),
                solver_position_w=np.asarray(snapshot['position_w'][row]).tolist(),
                normal_w=normal.tolist(),
                includemargin=float(snapshot['includemargin'][row]))
            # Preserve solver provenance and both witnesses for local loss
            # linearization. Missing geometry is explicit, never reconstructed
            # from the solver midpoint (which is wrong at edges).
            pair.update(witness_schema='newton_geometry_pair_v1',
                body_name=names[body], solver_row=int(row),
                constraint_type=int(snapshot['type'][row]),
                constraint_active=bool(actual[row]),
                efc_address=np.asarray(snapshot['efc_address'][row]).reshape(-1)[:int(
                    snapshot['constraint_rows'][row]) if 'constraint_rows' in snapshot else 1].tolist(),
                geometry_witness_available=all(f'geometry_point{s}_w' in snapshot for s in (0,1)),
                terrain_position_w=(np.asarray(snapshot[f'geometry_point{1-side}_w'][row]).tolist()
                    if f'geometry_point{1-side}_w' in snapshot else None))
            if isinstance(binding,list):
                pair['surface_attribution']=matching[0].get('attribution_schema','legacy_exact_plane')
                if contact_source is not None:
                    pair['contact_source'] = dict(contact_source)
                    pair['surface_attribution'] = contact_source['normal_fan_schema']
            if include_candidates:candidate_pairs.append(dict(pair,constraint_active=bool(actual[row])))
            if not actual[row]:continue
            pairs.append(pair)
            if not allocated[row]:
                failed[p] = True
                continue
            active[p] = True
            score = float(snapshot['dist'][row]-snapshot['includemargin'][row])
            if score < representatives[p]:
                representatives[p] = score; position[p] = pos; surfaces[p] = surface
    return PartContacts(active, failed, position, surfaces, pairs, candidate_pairs)


def extract_recording_contacts(arrays, metadata, env_id, *, parts=PARTS):
    missing = [f'solver_contact_{k}' for k in FIELDS if f'solver_contact_{k}' not in arrays]
    if missing:
        raise ValueError('Recording has no verified solver activation: '+', '.join(missing))
    if metadata.get('solver_contact_semantics', {}).get('schema') != SCHEMA:
        raise ValueError('Unknown contact snapshot semantics')
    labels = metadata['newton_body_labels']
    body_env = {}
    for i, label in enumerate(labels):
        match = re.search(r'/env_(\d+)/Robot/', label)
        if match:
            body_env[i] = int(match[1])
    if env_id not in body_env.values():
        raise ValueError('Recording has no robot body mapping for selected environment')
    shape_body = metadata['newton_shape_body']
    # These are physical shape IDs, not inferred "top" classes. Training
    # surface catalogs must explicitly bind them before using the descriptors.
    surfaces = {i: i for i, body in enumerate(shape_body) if int(body) not in body_env}
    rows = [reduce_snapshot({k: arrays[f'solver_contact_{k}'][f] for k in FIELDS},
        body_labels=labels, body_env=body_env, shape_surface=surfaces, env_id=env_id, parts=parts)
        for f in range(len(arrays['solver_contact_count']))]
    return {k: np.stack([getattr(row, k) for row in rows])
            for k in ('active', 'unallocated', 'position_w', 'surface')}
