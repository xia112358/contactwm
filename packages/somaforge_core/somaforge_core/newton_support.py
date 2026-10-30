"""Force-bearing support motion from one MJWarp constraint evaluation.

Contact activation is supplied separately. A positive normal force localizes
load within already verified contacts; it never invents contact truth. Body
net forces or zero-filled Newton raw forces cannot replace per-constraint force.
"""
import numpy as np

SUPPORT_SCHEMA = 'newton_mjwarp_force_bearing_point_motion_v3'
SUPPORT_FIELDS = ('force_on_body1_w', 'torque_on_body1_w', 'point_velocity0_w', 'point_velocity1_w',
                  'pre_point_velocity0_w', 'pre_point_velocity1_w',
                  'application_point0_local', 'application_point1_local')


def observed_support(snapshot, binding, body_labels, num_envs, *, selection=None):
    """Assess a complete native solve; zero load and missing evidence differ.

    This is a passive read. It does not change physics, poses, margins or policy.
    The six-part arrays retain solve sampling, not post-integration pose time.
    """
    selected = selection if selection is not None else select_support_contacts(snapshot, binding, body_labels)
    measured = support_motion(snapshot, eligible=selected['eligible'],
                              robot_side=selected['robot_side'], groups=selected['groups'],
                              group_count=num_envs * 6)
    return {key: np.asarray(measured[key]).reshape(num_envs, 6) for key in
            ('contact_activated', 'load_known', 'load_bearing', 'load_state',
             'normal_force_n', 'loaded_motion_known', 'rms_tangent_speed_m_s', 'slip_state')}


def select_support_contacts(snapshot, binding, body_labels, *, face_indexes=None,
                            part_names=None, body_names_by_part=None):
    """Resolve actual force application points to native primary mesh faces.

This attributes an existing active constraint, never creates contact from
distance. All faces, including sides, compete before shared face eligibility.
"""
    from .contact_schema import CONTACT_BODY_NAMES_BY_PART
    from .contact_face_selection import select_contact_pairs
    from .newton_contacts import PARTS, actual_primary_face_indices, constraint_decision
    parts = PARTS if part_names is None else tuple(part_names)
    body_names_by_part = CONTACT_BODY_NAMES_BY_PART if body_names_by_part is None else body_names_by_part
    n=int(snapshot['count']);body_env={int(k):v for k,v in binding['body_env'].items()}
    shape_faces={int(k):v for k,v in binding['shape_surface'].items()}
    active,allocated=constraint_decision(snapshot['dist'],snapshot['includemargin'],
        snapshot['type'],snapshot['efc_address'],n,snapshot['constraint_rows'])
    if not np.array_equal(active[:n],snapshot['active'][:n]) or not np.array_equal(allocated[:n],snapshot['constraint_allocated'][:n]):
        raise ValueError('Inconsistent saved solver activation/allocation')
    side=np.full(n,-1,int);group=np.full(n,-1,int);surface=np.full(n,-1,int);records=[]
    pending={}
    for i in np.flatnonzero(active[:n]):
        if not allocated[i]:raise ValueError('Unallocated active support constraint')
        bodies=[int(snapshot[f'body{k}'][i]) for k in (0,1)]
        robot=[k for k in (0,1) if bodies[k] in body_env]
        if len(robot)!=1:continue  # self and cross-robot contact are not terrain
        k=robot[0]
        if body_env[bodies[k]]!=int(snapshot['worldid'][i]):continue
        body=body_labels[bodies[k]].rsplit('/',1)[-1]
        part=next((p for p,name in enumerate(parts) if body in body_names_by_part[name]),None)
        if part is None:continue
        external=int(snapshot[f'shape{1-k}'][i])
        if external not in shape_faces:raise ValueError('Unknown actual external shape')
        position=np.asarray(snapshot['position_w'][i]);normal=np.asarray(snapshot['frame_w'][i,0])*(1 if k==1 else -1)
        side[i]=k;group[i]=int(snapshot['worldid'][i])*len(parts)+part
        pending.setdefault(external,[]).append((i,part,position,normal))
    resolved={}
    for external, rows in pending.items():
        faces=shape_faces[external]
        points, normals = [r[2] for r in rows], [r[3] for r in rows]
        chosen=(face_indexes[external].query(points, normals) if face_indexes is not None
                else actual_primary_face_indices(points,normals,faces))
        for (i,part,position,normal),face_index in zip(rows,chosen):
            surface[i]=int(faces[int(face_index)]['surface'])
            resolved[i]=(part,position)
    for i in sorted(resolved):
        part,position=resolved[i]
        records.append(dict(row=i,part=part,surface=int(surface[i]),allocated=True,
            position_w=position.tolist(),dist=float(snapshot['dist'][i]),
            includemargin=float(snapshot['includemargin'][i]),
            position_semantics='solver_force_application_point',surface_attribution='native_finite_triangle_primary_at_force_application_point'))
    selected=select_contact_pairs([records],binding['surface_catalog'],part_count=len(parts))
    eligible=np.zeros(n,bool)
    eligible[[p['row'] for p in selected['contact_pairs'][0]]]=True
    return dict(eligible=eligible,robot_side=side,groups=group,surfaces=surface,
                excluded_rows=[p['row'] for p in selected['abnormal_contact_pairs'][0]])


class SupportContactSelector:
    """Batch the same selection for one immutable realized native binding.

    Static body/part and face eligibility maps are compiled once. Actual solver
    activation, allocation, world identity and force-point primary faces are
    still checked on every read. No post-integration pose participates.
    """
    def __init__(self, binding, body_labels, *, face_indexes=None,
                 part_names=None, body_names_by_part=None):
        from .contact_schema import CONTACT_BODY_NAMES_BY_PART
        from .contact_face_selection import select_contact_pairs, upward_face_mask
        from .native_surface_index import NativeSurfaceIndex
        from .newton_contacts import PARTS
        parts = PARTS if part_names is None else tuple(part_names)
        names = CONTACT_BODY_NAMES_BY_PART if body_names_by_part is None else body_names_by_part
        self.part_count = len(parts)
        # Validate the same catalog/normal contract once, before compiling it.
        select_contact_pairs([[]], binding['surface_catalog'], part_count=self.part_count)
        catalog = {int(face['surface']): face for face in binding['surface_catalog']}
        upward = {int(face['surface']): bool(keep) for face, keep in zip(
            binding['surface_catalog'], upward_face_mask([f['normal_w'] for f in binding['surface_catalog']]))}
        body_env = {int(k): int(v) for k, v in binding['body_env'].items()}
        size = max(body_env, default=-1)+1
        self.body_world = np.full(size, -1, np.int64)
        self.body_part = np.full(size, -1, np.int64)
        part_by_name = {}
        for part, name in enumerate(parts):
            for body in names[name]:
                part_by_name.setdefault(body, part)
        for body, world in body_env.items():
            if body < 0 or world < 0:
                raise ValueError('Invalid native robot body/world binding')
            self.body_world[body] = world
            self.body_part[body] = part_by_name.get(body_labels[body].rsplit('/', 1)[-1], -1)
        self.indexes, self.face_ids, self.face_upward = {}, {}, {}
        for shape, faces in binding['shape_surface'].items():
            shape = int(shape)
            ids = np.asarray([int(face['surface']) for face in faces], np.int64)
            if any(int(sid) not in catalog for sid in ids):
                raise ValueError('Unknown actual surface')
            self.face_ids[shape] = ids
            self.face_upward[shape] = np.asarray([upward[int(sid)] for sid in ids], bool)
            self.indexes[shape] = face_indexes[shape] if face_indexes is not None else NativeSurfaceIndex(faces)

    def __call__(self, snapshot):
        from .newton_contacts import constraint_decision
        n = int(snapshot['count'])
        active, allocated = constraint_decision(snapshot['dist'], snapshot['includemargin'],
            snapshot['type'], snapshot['efc_address'], n, snapshot['constraint_rows'])
        if (not np.array_equal(active[:n], snapshot['active'][:n])
                or not np.array_equal(allocated[:n], snapshot['constraint_allocated'][:n])):
            raise ValueError('Inconsistent saved solver activation/allocation')
        if np.any(active[:n] & ~allocated[:n]):
            raise ValueError('Unallocated active support constraint')
        bodies = np.stack((snapshot['body0'][:n], snapshot['body1'][:n]), axis=1)
        valid = (bodies >= 0) & (bodies < len(self.body_world))
        worlds = np.full(bodies.shape, -1, np.int64)
        parts = np.full(bodies.shape, -1, np.int64)
        worlds[valid] = self.body_world[bodies[valid]]
        parts[valid] = self.body_part[bodies[valid]]
        robot = worlds >= 0
        robot_side = robot[:, 1].astype(np.int64)
        row = np.arange(n)
        own_world, own_part = worlds[row, robot_side], parts[row, robot_side]
        choose = active[:n] & (robot.sum(1) == 1) & (own_part >= 0)
        choose &= own_world == np.asarray(snapshot['worldid'][:n], np.int64)
        rows = np.flatnonzero(choose)
        side = np.full(n, -1, np.int64); group = np.full(n, -1, np.int64)
        surface = np.full(n, -1, np.int64); eligible = np.zeros(n, bool)
        side[rows] = robot_side[rows]
        group[rows] = own_world[rows]*self.part_count+own_part[rows]
        external = np.where(robot_side[rows] == 0, snapshot['shape1'][rows], snapshot['shape0'][rows])
        depth = (np.asarray(snapshot['dist'][rows], np.float64)
                 -np.asarray(snapshot['includemargin'][rows], np.float64))
        if not np.isfinite(depth).all() or (depth >= 0).any():
            raise ValueError('Invalid activated Newton pair')
        for shape in np.unique(external):
            shape = int(shape)
            if shape not in self.indexes:
                raise ValueError('Unknown actual external shape')
            selected = rows[external == shape]
            normal = snapshot['frame_w'][selected, 0]*np.where(robot_side[selected] == 1, 1., -1.)[:, None]
            face = self.indexes[shape].query(snapshot['position_w'][selected], normal)
            surface[selected] = self.face_ids[shape][face]
            eligible[selected] = self.face_upward[shape][face]
        return dict(eligible=eligible, robot_side=side, groups=group, surfaces=surface,
                    excluded_rows=rows[~eligible[rows]].tolist())


def snapshot_support(solver, snapshot):
    """Read force and both incoming/resolved velocities at solve geometry."""
    import warp as wp
    import mujoco_warp as mjw
    m,d=solver.mjw_model,solver.mjw_data
    count=int(snapshot['count']); capacity=len(snapshot['dist'])
    result={name:np.zeros((capacity,3),np.float32) for name in SUPPORT_FIELDS}
    if count==0:return result
    # mjw.contact_force decodes the actual pyramidal/elliptic efc rows. Newton
    # raw rigid_contact_force is not populated by this constraint solver.
    device=d.contact.pos.device
    ids=wp.array(np.arange(count,dtype=np.int32),dtype=wp.int32,device=device)
    wrench=wp.zeros(count,dtype=wp.spatial_vector,device=device)
    mjw.contact_force(m,d,ids,True,wrench)
    values=wrench.numpy()
    if values.shape!=(count,6) or not np.isfinite(values).all():
        raise ValueError('Invalid MJWarp per-contact wrench')
    result['force_on_body1_w'][:count]=values[:,:3]
    result['torque_on_body1_w'][:count]=values[:,3:]
    geom=d.contact.geom.numpy()[:count]
    world=np.asarray(snapshot['worldid'][:count],int)
    geom_body=m.geom_bodyid.numpy();root=m.body_rootid.numpy()
    # cvel is rotation:translation, centered at the root subtree COM. It is
    # retained from the velocity evaluation that constructed these constraints.
    velocity=d.cvel.numpy();com=d.subtree_com.numpy()
    # Integration updates qvel after cvel was evaluated. Reconstruct the same
    # body spatial Jacobian using actual ancestry and cdof, without mutating the
    # policy's solver Data or rerunning any physics/kinematic kernel.
    cache=getattr(solver,'_somaforge_support_velocity_tree',None)
    if cache is None or cache[0]!=id(m):
        parent=m.body_parentid.numpy();dof_body=m.dof_bodyid.numpy()
        ancestry=np.zeros((len(parent),len(dof_body)),bool)
        for b in range(1,len(parent)):
            j=b;seen=set()
            while j:
                if j in seen:raise ValueError('Cyclic MJWarp body tree')
                seen.add(j);ancestry[b] |= dof_body==j;j=int(parent[j])
        cache=(id(m),ancestry);solver._somaforge_support_velocity_tree=cache
    resolved=np.einsum('bv,wv,wvj->wbj',cache[1],d.qvel.numpy(),d.cdof.numpy(),optimize=True)
    point=np.asarray(snapshot['position_w'][:count])
    # MuJoCo body frames are copied directly into Newton body_q by the solver's
    # convert_body_xforms_to_warp_kernel. Read the retained solve transform,
    # not the post-integration state used by the trajectory recorder.
    body_position=d.xpos.numpy();body_rotation=d.xmat.numpy().reshape(*body_position.shape[:2],3,3)
    body_mapping=solver.mjc_body_to_newton.numpy()
    for side in (0,1):
        body=geom_body[geom[:,side]]
        if not np.array_equal(body_mapping[world,body],snapshot[f'body{side}'][:count]):
            raise ValueError('Support point body frame mapping differs from Newton shape body')
        result[f'application_point{side}_local'][:count]=np.einsum(
            'nji,nj->ni',body_rotation[world,body],point-body_position[world,body])
        spatial=velocity[world,body]
        origin=com[world,root[body]]
        result[f'pre_point_velocity{side}_w'][:count]=spatial[:,3:]+np.cross(spatial[:,:3],point-origin)
        spatial=resolved[world,body]
        result[f'point_velocity{side}_w'][:count]=spatial[:,3:]+np.cross(spatial[:,:3],point-origin)
    return result


def support_motion(snapshot, *, eligible, robot_side, groups, group_count):
    """Force-weighted RMS slip speed at actual force application points.

eligible must already verify activation, allocation and task primary faces.
RMS of local velocities is intentional: motion at different loaded points must
not cancel at a fictitious center of pressure. Zero load is known unloaded;
motion of loaded points is then unknown, never an automatic static certificate.
"""
    count=int(snapshot['count']);eligible=np.asarray(eligible,bool)
    side=np.asarray(robot_side,int);groups=np.asarray(groups,int)
    if any(x.shape!=(count,) for x in (eligible,side,groups)):
        raise ValueError('Support mapping dimensions differ from contact count')
    if np.any(eligible & ~np.isin(side,[0,1])) or np.any(eligible & ((groups<0)|(groups>=group_count))):
        raise ValueError('Invalid robot side/support group')
    if not all(key in snapshot for key in SUPPORT_FIELDS):
        raise ValueError('Missing per-constraint support force/velocity; support location unknown')
    normal=np.asarray(snapshot['frame_w'][:count,0],float)
    force=np.asarray(snapshot['force_on_body1_w'][:count],float)
    relative=np.asarray(snapshot['point_velocity1_w'][:count],float)-np.asarray(snapshot['point_velocity0_w'][:count],float)
    if not np.isfinite(np.concatenate((normal,force,relative))).all():
        raise ValueError('Nonfinite solver support fields')
    if not np.allclose(np.linalg.norm(normal[eligible],axis=-1),1.,atol=1.e-6):
        raise ValueError('Invalid constraint normals')
    if np.any(eligible & (~np.asarray(snapshot['active'][:count],bool) |
                          ~np.asarray(snapshot['constraint_allocated'][:count],bool))):
        raise ValueError('Support includes inactive or unallocated constraint')
    load=np.maximum(np.einsum('ij,ij->i',force,normal),0.)
    tangent=relative-np.einsum('ij,ij->i',relative,normal)[:,None]*normal
    speed=np.linalg.norm(tangent,axis=-1)
    select=eligible & (load>0)
    from .contact_motion import loaded_motion_statistics
    statistics=loaded_motion_statistics(speed[select],load[select],groups[select],group_count)
    from .support_semantics import assess_support
    contact=np.zeros(group_count,bool)
    contact[groups[eligible]]=True
    assessment=assess_support(contact,statistics['normal_force_n'],statistics['rms_tangent_speed_m_s'],
                              load_known=np.ones(group_count,bool))
    return dict(schema=SUPPORT_SCHEMA,**statistics,
                assessment_schema=assessment['schema'],contact_activated=assessment['contact_activated'],
                load_known=assessment['load_known'],load_state=assessment['load_state'],slip_state=assessment['slip_state'],
                loaded_contact_rows=np.flatnonzero(select),point_tangent_speed_m_s=speed[select],
                point_normal_force_n=load[select],point_positions_w=np.asarray(snapshot['position_w'][:count])[select])
