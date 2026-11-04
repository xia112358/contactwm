"""Passive native constraint evidence for primitive-box scenes.

Direct mode reduces allocated simulator contacts by body part, without surface
classification. Legacy face attribution remains available for old recordings.
"""
from __future__ import annotations

import re
import numpy as np
import torch


class NativeContacts:
    def __init__(self, env, *, direct=False, primary_faces=False):
        self.direct=direct
        self.primary_faces=primary_faces
        self.env = env
        from somaforge_core.collision_profiles import contact_body_names
        self.collision_profile = getattr(env, 'collision_profile', 'legacy_v1')
        self.body_names = contact_body_names(self.collision_profile)
        self.model = env.sim.backend_manager.get_model()
        self.solver = env.sim.backend_manager._solver
        model, solver = self.model, self.solver
        if direct and not primary_faces:
            self._init_direct()
            return
        from scipy.spatial.transform import Rotation
        from somaforge_core.native_surface_index import NativeSurfaceIndex

        body_world = model.body_world.numpy()
        body_env = {i: int(body_world[i]) for i, label in enumerate(model.body_label)
                    if re.search(r'/env_\d+/Robot/', label)}
        if set(body_env.values()) != set(range(env.N)):
            raise ValueError('Expected one canonical robot per native world')
        mapping = solver.mjc_geom_to_newton_shape.numpy()
        m = solver.mjw_model
        positions, sizes, quats = (getattr(m, name).numpy() for name in ('geom_pos', 'geom_size', 'geom_quat'))
        types, bodies = m.geom_type.numpy(), m.geom_bodyid.numpy()
        faces, catalog = {}, []
        self.world_faces = []
        self.surface_offsets = []
        # Explicit right-handed corner order for all six bounded box faces.
        signs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                          [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
        quads = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
        for world in range(env.N):
            self.surface_offsets.append(len(catalog))
            world_faces = []
            for geom in range(mapping.shape[1]):
                shape = int(mapping[world, geom])
                if shape < 0 or int(bodies[geom]) != 0:
                    continue
                if int(types[geom]) != 6:
                    raise ValueError('Contact experiment requires actual static box terrain')
                pick = lambda x: x[min(world, len(x)-1), geom]
                rotation = Rotation.from_quat(pick(quats)[[1,2,3,0]])
                vertices = rotation.apply(signs * pick(sizes)) + pick(positions)
                actual = []
                for quad in quads:
                    triangles = vertices[np.array([[quad[0],quad[1],quad[2]], [quad[0],quad[2],quad[3]]])]
                    normal = np.cross(triangles[0,1]-triangles[0,0], triangles[0,2]-triangles[0,0])
                    normal /= np.linalg.norm(normal)
                    surface = len(catalog)
                    item = dict(surface=surface, normal_w=normal.tolist(),
                                plane_offset=float(normal @ triangles[0,0]), triangles_w=triangles.tolist())
                    catalog.append({k:v for k,v in item.items() if k != 'triangles_w'})
                    actual.append(item)
                    world_faces.append(item)
                if shape in faces:
                    raise ValueError('External shape unexpectedly shared across worlds')
                faces[shape] = actual
            self.world_faces.append(world_faces)
        self.binding = dict(body_env=body_env, shape_surface=faces, surface_catalog=catalog)
        names=sorted({model.body_label[i].rsplit('/',1)[-1] for i in body_env})
        slots={name:i for i,name in enumerate(names)}
        self._summary_world=np.full(len(model.body_label),-1,np.int64)
        self._summary_slot=np.full(len(model.body_label),-1,np.int64)
        for body, world in body_env.items():
            self._summary_world[body]=world
            self._summary_slot[body]=slots[model.body_label[body].rsplit('/',1)[-1]]
        self.body_contact_metadata=dict(body_contact_schema='native_all_robot_terrain_constraints_v1',
            body_contact_names=names,
            body_contact_scope='all active allocated robot-terrain contacts including side faces; separate from task-part eligibility')
        self.indexes = {shape: NativeSurfaceIndex(value) for shape,value in faces.items()}
        self.last_snapshot = None
        self.last_evidence = None
        self.surface_offsets_device = torch.tensor(self.surface_offsets, device=env.device)
        if direct:self._init_direct()

    def set_observed_worlds(self, world_ids):
        """Limit expensive face attribution; retain all original native rows.

        Unobserved worlds are unknown, never reported as known no-contact.
        Physics, sensors and the teacher policy are completely unaffected.
        """
        if self.direct:
            raise ValueError('World scoping currently applies to classified capture only')
        ids = torch.as_tensor(world_ids).detach().cpu().numpy()
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Observed worlds must be a one-dimensional integer list')
        if not len(ids) or np.any(ids < 0) or np.any(ids >= self.env.N) or len(np.unique(ids)) != len(ids):
            raise ValueError('Observed worlds must be nonempty, distinct, in-range IDs')
        if not hasattr(self, '_all_body_env'):
            self._all_body_env = dict(self.binding['body_env'])
        mask = np.zeros(self.env.N, dtype=bool)
        mask[ids] = True
        self._observed_world_mask = mask
        self.binding['body_env'] = {body: world for body, world in self._all_body_env.items() if mask[world]}
        for name in ('evidence_binding', 'device_selector'):
            if hasattr(self, name):
                delattr(self, name)

    def _known_mask(self, *, solved=False):
        """Only a post-reset solve in an observed world is measured evidence."""
        observed = torch.as_tensor(getattr(self, '_observed_world_mask',
                                           np.ones(self.env.N, bool)), device=self.env.device)
        return (((self.env.ep_step > 0) | solved) & observed)[:, None].expand(-1, 6)

    @staticmethod
    def _normal_error(snapshot):
        """Validate all active native rows, including non-task body contacts."""
        dist = torch.as_tensor(snapshot['dist'])
        margin = torch.as_tensor(snapshot['includemargin'], device=dist.device)
        kind = torch.as_tensor(snapshot['type'], device=dist.device)
        normal = torch.as_tensor(snapshot['frame_w'], device=dist.device)[:, 0]
        count = torch.as_tensor(snapshot['count'], device=dist.device)
        if count.numel() != 1:
            raise ValueError('Native contact count must be scalar')
        valid = torch.arange(len(dist), device=dist.device) < count.reshape(())
        active = valid & ((kind.long() & 1) != 0) & (dist < margin)
        length = normal.norm(dim=-1)
        unit = torch.isclose(length, torch.ones_like(length), atol=1e-6, rtol=1e-5)
        return (active & (~torch.isfinite(normal).all(-1) | ~unit)).any()

    def _mask_unknown(self, result, *, known=None):
        """Explicit sentinels for unknown summaries; raw evidence is untouched."""
        if known is None:known = self._known_mask()
        result['known'] = known
        for key, value in list(result.items()):
            if key == 'known':
                continue
            mask = known[:, :1] if key.startswith('body_contact_') else known
            if key == 'body_contact_known':
                result[key] = mask.expand_as(value).clone()
                continue
            if value.ndim > mask.ndim:
                mask = mask[..., None]
            sentinel = float('nan') if key == 'slip' else -1 if key in ('surface', 'surface_local') else 0
            result[key] = torch.where(mask, value, torch.full_like(value, sentinel))
        return result

    def _init_direct(self):
        from .contact_control import PARTS
        model=self.model;mapping=self.solver.mjc_geom_to_newton_shape.numpy()
        shape_body=model.shape_body.numpy();world=model.body_world.numpy()
        by_name={name:i for i,part in enumerate(PARTS) for name in self.body_names[part]}
        robot=np.array([bool(re.search(r'/env_\d+/Robot/',label)) for label in model.body_label])
        part=np.array([by_name.get(label.rsplit('/',1)[-1],-1) for label in model.body_label])
        safe_shape=np.clip(mapping,0,len(shape_body)-1);body=shape_body[safe_shape]
        safe_body=np.clip(body,0,len(robot)-1)
        valid=(mapping>=0)&(mapping<len(shape_body))&(body>=0)&(body<len(robot))
        is_robot=valid&robot[safe_body]
        same_world=world[safe_body]==np.arange(mapping.shape[0])[:,None]
        self.geom_robot=torch.as_tensor(is_robot,device=self.env.device)
        self.geom_part=torch.as_tensor(np.where(is_robot&same_world,part[safe_body],-1),device=self.env.device)
        from somaforge_core.contact_system.evidence import BodyBinding
        self.evidence_binding=BodyBinding.from_model(model,PARTS,self.body_names,self.env.device)
        import hashlib
        from importlib.metadata import version
        from somaforge_core.robot_assets import canonical_g1_asset_metadata
        digest=hashlib.sha256()
        fingerprint_fields=[]
        for owner,prefix,fields in ((model,'newton',('shape_body','shape_type','shape_transform','shape_scale','shape_margin','shape_gap')),
                (self.solver.mjw_model,'mjwarp',('geom_size','geom_pos','geom_quat','geom_friction','geom_margin','geom_solref','geom_solimp','body_mass','body_inertia'))):
            for name in fields:
                value=getattr(owner,name,None)
                if value is not None:
                    key=prefix+'.'+name;fingerprint_fields.append(key)
                    digest.update(key.encode());digest.update(value.numpy().tobytes())
        digest.update(mapping.tobytes())
        self.evidence_provenance=dict(backend='supermimic_mjwarp',selection='primary_horizontal_upward_faces_v1' if self.primary_faces else 'one_robot_no_face_filter',
            parts=PARTS,robot_asset=canonical_g1_asset_metadata(collision_profile=self.collision_profile),model_fingerprint=digest.hexdigest(),
            fingerprint_fields=fingerprint_fields,
            contact_response=getattr(self.env,'contact_response','legacy'),
            solver_numerics=getattr(self.env,'contact_solver_numerics',{}).get('contract','upstream'),
            runtime={name:version(name) for name in ('newton','mujoco','mujoco-warp','warp-lang')})
        from .contact_control import LINKS
        proto=self.solver.mjc_body_to_newton.numpy()[0]
        self.part_body_ids=[]
        for name in LINKS:
            ids=[i for i,b in enumerate(proto) if b>=0 and model.body_label[int(b)].rsplit('/',1)[-1]==name]
            if len(ids)!=1:raise ValueError('Missing/ambiguous native part frame: '+name)
            self.part_body_ids.append(ids[0])
        self.last_evidence=None
        self.targets=None

    def export_collision_model(self, path):
        """Serialize the actual solver geometry for the matching objective atlas."""
        import mujoco, hashlib, json
        from pathlib import Path
        from somaforge_core.robot_assets import canonical_g1_asset_metadata
        from somaforge_core.collision_profiles import audit_surface_mujoco
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + '.json')
        if path.exists() or sidecar.exists():
            raise FileExistsError('Collision model export must use a fresh path: ' + str(path))
        # The CPU compilation template does not follow runtime GPU parameter
        # changes (including the actual solver dt). Export an independent,
        # synchronized snapshot, never mutate the running solver template.
        import copy
        from .contact_response import contract as response_contract
        model = copy.copy(self.solver.mj_model)
        fields = ('geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap', 'geom_friction')
        differing_world_fields = []
        for field in fields:
            actual = getattr(self.solver.mjw_model, field).numpy()
            getattr(model, field)[:] = actual[0]
            if len(actual) > 1 and np.any(actual != actual[:1]):
                differing_world_fields.append(field)
        model.opt.timestep = self.env.sim.backend_manager.get_solver_dt()
        counts = audit_surface_mujoco(model) if self.collision_profile == 'surface_v2' else None
        mujoco.mj_saveModel(model, str(path))
        meta = dict(schema='somaforge_solver_collision_model_v1',
            robot_asset=canonical_g1_asset_metadata(collision_profile=self.collision_profile),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), colliders=counts,
            source='actual_running_newton_mjwarp_solver',
            exported_world=0,
            world_scope='representative world 0 only; other worlds are not serialized',
            synchronized_live_fields=list(fields),
            per_world_parameter_differences=differing_world_fields,
            solver_dt=float(model.opt.timestep),
            contact_response=getattr(self.env, 'contact_response', 'legacy'),
            contact_response_contract=response_contract(getattr(self.env, 'contact_response', 'legacy')))
        sidecar.write_text(json.dumps(meta, indent=2) + '\n')
        return path

    def set_targets(self,points,ids):
        if self.direct:
            if self.targets is None:self.targets=torch.zeros((self.env.N,6,3),device=self.env.device)
            else:self.targets.zero_()
            self.targets[ids]=points+self.env.scene.env_origins[ids,None]

    def update_targets(self,points,ids):
        """Replace only handed-off worlds; retain every other world's goal."""
        if self.direct:
            if self.targets is None:
                self.set_targets(points,ids)
            else:
                self.targets[ids]=points+self.env.scene.env_origins[ids,None]

    def _read_direct(self,*,substep=None,same_solve=False):
        """Owned point-force evidence; nearest-point summary is compatibility only."""
        from somaforge_core.contact_system.evidence import ContactEvidence,Sampling,solver_snapshot
        from .motion import CONTROL_DT
        manager=self.env.sim.backend_manager
        snapshot=solver_snapshot(self.solver,self.model)
        known=self._known_mask(solved=same_solve)
        evidence=ContactEvidence(snapshot,sampling=Sampling(CONTROL_DT,float(manager.get_solver_dt()),
            substep=int(manager._num_substeps)-1 if substep is None else substep,
            pose_contact_same_state=same_solve,pose_time='constraint_evaluation' if same_solve else 'post_integration'),known=known,control_step=self.env.ep_step,
            provenance=self.evidence_provenance)
        if self._normal_error(snapshot):
            raise ValueError('Invalid active native constraint normal')
        self.last_evidence=evidence
        import warp as wp
        positions=wp.to_torch(self.solver.mjw_data.xpos)
        rotations=wp.to_torch(self.solver.mjw_data.xmat).reshape(*positions.shape[:2],3,3)
        # Advanced indexing already owns its storage; no second pose copy is
        # needed to retain this solve across the next simulator step.
        self.last_part_poses=(positions[:,self.part_body_ids],rotations[:,self.part_body_ids])
        eligible=None
        if self.primary_faces:
            from somaforge_core.live_contact_device import DeviceSupportContactSelector
            if not hasattr(self,'device_selector'):
                self.device_selector=DeviceSupportContactSelector(self.binding,self.model.body_label,self.env.device,
                    body_names_by_part=self.body_names)
            selected=self.device_selector(dict(evidence.rows,count=evidence.count))
            self.device_selector.validate(selected)
            self.last_primary_selection=selected
            eligible=selected['eligible']
        view=evidence.terrain_view(self.evidence_binding,eligible=eligible)
        self.last_view=view
        reduced=view.reduce();n=evidence.count
        index,group=view.rows,view.groups
        point=evidence.rows['position_w'];total=self.env.N*6
        if self.targets is None:score=(evidence.rows['dist']-evidence.rows['includemargin'])[index]
        else:score=(point[index]-self.targets.reshape(-1,3)[group]).square().sum(-1)
        minimum=torch.full((total,),float('inf'),device=point.device,dtype=point.dtype)
        minimum.scatter_reduce_(0,group,score,reduce='amin',include_self=True)
        winner=torch.full((total,),n,device=point.device,dtype=torch.long)
        winner.scatter_reduce_(0,group,torch.where(score==minimum[group],index,n),reduce='amin',include_self=True)
        present=winner<n
        # A sentinel row avoids two CUDA boolean compactions/host syncs. It also
        # handles an entirely empty solve without a special device-side branch.
        points=torch.cat((point,point.new_zeros((1,3))),0)[winner]
        # Already-computed body sensor forces are diagnostic only, never activation.
        force=self.env.xi[:,34:52].reshape(self.env.N,6,3)*self.env.force_scale
        self.direct_counts=dict(native_rows=n,capacity=len(self.solver.mjw_data.contact.dist),selected_rows=len(index))
        return self._mask_unknown(dict(active=present.reshape(-1,6),known=known,
            point=points.reshape(-1,6,3)-self.env.scene.env_origins[:,None],
            surface_local=torch.full((self.env.N,6),-1,device=point.device,dtype=torch.long),
            force=force.clone(),load=force[...,2].clamp_min(0),
            paired_force_w=reduced['force_w'],contact_count=reduced['contact_count']),known=known)

    def surface_features(self,atlas,*,bandwidth=.01):
        """Project retained evidence using its retained constraint-evaluation poses."""
        if self.last_evidence is None or tuple(atlas.parts)!=self.evidence_binding.parts:
            raise ValueError('Surface projection requires a matching retained native solve')
        return atlas.project(self.last_view,
                             *self.last_part_poses,bandwidth=bandwidth)

    def _read_direct_reference(self):
        """Independent CPU reduction of the same native rows, for explicit audits."""
        import warp as wp
        torch.cuda.synchronize()
        c=self.solver.mjw_data.contact
        get=lambda key:wp.to_torch(getattr(c,key)).detach().cpu().numpy()
        n=int(wp.to_torch(self.solver.mjw_data.nacon).reshape(-1)[0])
        world,geom,dist,margin,kind,dim,address,pos=[get(k) for k in ('worldid','geom','dist','includemargin','type','dim','efc_address','pos')]
        if not 0 <= n <= len(dist):
            raise ValueError('Native contact capacity exceeded')
        if self._normal_error(dict(dist=dist,includemargin=margin,type=kind,frame_w=get('frame'),count=n)):
            raise ValueError('Invalid active native constraint normal')
        robot=self.geom_robot.cpu().numpy();part=self.geom_part.cpu().numpy()
        targets=None if self.targets is None else self.targets.cpu().numpy()
        active=np.zeros((self.env.N,6),bool);points=np.zeros((self.env.N,6,3),np.float32)
        best=np.full((self.env.N,6),np.inf)
        for i in range(n):
            if not (int(kind[i])&1) or not dist[i]<margin[i]:continue
            w=int(world[i]);g0,g1=map(int,geom[i]);r0,r1=bool(robot[w,g0]),bool(robot[w,g1])
            if r0==r1:continue
            p=int(part[w,g0 if r0 else g1])
            if p<0:continue
            rows=(1 if dim[i]==1 else 2*(dim[i]-1)) if int(self.solver.mjw_model.opt.cone)==0 else dim[i]
            if (address[i,:rows]<0).any():raise ValueError('Unallocated active contact')
            score=float(dist[i]-margin[i]) if targets is None else float(((pos[i]-targets[w,p])**2).sum())
            if score<best[w,p]:best[w,p]=score;points[w,p]=pos[i];active[w,p]=True
        known=self._known_mask()
        active &= known.cpu().numpy()
        points[~active]=0
        points-=self.env.scene.env_origins.cpu().numpy()[:,None]
        force=self.env.xi[:,34:52].reshape(self.env.N,6,3)*self.env.force_scale
        return self._mask_unknown(dict(active=torch.as_tensor(active,device=self.env.device),known=known,point=torch.as_tensor(points,device=self.env.device),surface_local=torch.full((self.env.N,6),-1,device=self.env.device,dtype=torch.long),force=force.clone(),load=force[...,2].clamp_min(0)))

    def target_faces(self, points, ids):
        """Resolve demonstrated witnesses onto the actual finite faces of matching worlds.

        Face IDs are local to each world's ordered native terrain geometry, so
        their meaning is independent of the number of parallel worlds.
        """
        from somaforge_core.native_surface_index import NativeSurfaceIndex
        worlds = ids.cpu().numpy()
        if not bool((self.env.task_ids[ids] == self.env.task_ids[ids[0]]).all()):
            raise ValueError('Target face mapping requires matching task terrains')
        origin = self.env.scene.env_origins.cpu().numpy()
        first = int(worlds[0]); faces = self.world_faces[first]
        reference = np.asarray([f['triangles_w'] for f in faces])-origin[first]
        for world in worlds:
            actual = np.asarray([f['triangles_w'] for f in self.world_faces[world]])-origin[world]
            if actual.shape != reference.shape or not np.allclose(actual, reference, atol=2e-3, rtol=0):
                raise ValueError('Parallel target terrains have different native faces')
        index = NativeSurfaceIndex(faces)
        xyz = points.detach().cpu().numpy().reshape(-1, 3)+origin[first]
        local = index.query(xyz, np.tile([0.,0.,1.], (len(xyz),1)))
        return torch.as_tensor(local,device=points.device).reshape(points.shape[:-1])

    def read_device(self,*,substep=None,same_solve=False):
        if self.direct:return self._read_direct(substep=substep,same_solve=same_solve)
        return self._read_classified()

    def _read_classified(self):
        """Use PHP's shared device selector; adapt its groups to this task API.

        No local face-selection or contact-activation implementation. The
        original CPU read remains available for collectors and audit checks.
        """
        import warp as wp
        import mujoco_warp as mjw
        from somaforge_core.live_contact_device import DeviceSupportContactSelector
        if not hasattr(self,'device_selector'):
            self.device_selector=DeviceSupportContactSelector(self.binding,self.model.body_label,self.env.device,
                                                              body_names_by_part=self.body_names)
        wp.stream_from_torch(torch.cuda.current_stream(self.env.device)).wait_stream(wp.get_stream(self.model.device))
        contact=self.solver.mjw_data.contact
        get=lambda key:wp.to_torch(getattr(contact,key))
        count=wp.to_torch(self.solver.mjw_data.nacon).reshape(-1)[0]
        # Same native tensor snapshot contract used by PHP MeasuredContacts.
        world=get('worldid').long();geom=get('geom').long();dist=get('dist')
        mapping=wp.to_torch(self.solver.mjc_geom_to_newton_shape)
        shape_body=wp.to_torch(self.model.shape_body)
        valid=torch.arange(len(dist),device=dist.device)<count
        mapped=(world>=0)&(world<mapping.shape[0])&((geom>=0)&(geom<mapping.shape[1])).all(-1)
        shapes=mapping[world.clamp(0,mapping.shape[0]-1)[:,None],geom.clamp(0,mapping.shape[1]-1)].long()
        shape_valid=((shapes>=0)&(shapes<len(shape_body))).all(-1)
        bodies=shape_body[shapes.clamp(0,len(shape_body)-1)]
        cone=int(self.solver.mjw_model.opt.cone);dim=get('dim')
        if cone not in (0,1):raise ValueError('Unknown native contact cone')
        rows=torch.where(dim==1,1,2*(dim-1)) if cone==0 else dim
        snapshot=dict(dist=dist,includemargin=get('includemargin'),type=get('type'),efc_address=get('efc_address'),
            worldid=world,position_w=get('pos'),frame_w=get('frame'),count=count,constraint_rows=rows,
            shape0=shapes[:,0],shape1=shapes[:,1],body0=bodies[:,0],body1=bodies[:,1],
            mapping_error=(valid&(~mapped|~shape_valid)).any())
        selected=self.device_selector(snapshot)
        selected['errors']['normal'] = self._normal_error(snapshot)
        self.device_selector.validate(selected)
        index=selected['eligible'].nonzero().flatten();groups=selected['groups'][index]
        total=self.env.N*6
        counts=torch.zeros(total,device=self.env.device).index_add_(0,groups,torch.ones(len(index),device=self.env.device))
        points=torch.zeros((total,3),device=self.env.device).index_add_(0,groups,snapshot['position_w'][index])
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.env.device))):
            wrench=wp.zeros(len(index),dtype=wp.spatial_vector,device=self.model.device)
            if len(index):mjw.contact_force(self.solver.mjw_model,self.solver.mjw_data,wp.from_torch(index.int()),True,wrench)
        values=wp.to_torch(wrench)[:,:3]
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite native contact force')
        sign=torch.where(selected['robot_side'][index]==1,1.,-1.)
        force=torch.zeros((total,3),device=self.env.device).index_add_(0,groups,values*sign[:,None])
        load=torch.zeros(total,device=self.env.device).index_add_(0,groups,(values*snapshot['frame_w'][index,0]).sum(-1).clamp_min(0))
        # Same deepest allocated-contact witness and tie ordering as read().
        depth = snapshot['dist'][index]-snapshot['includemargin'][index]
        best = torch.full((total,),float('inf'),device=self.env.device)
        best.scatter_reduce_(0,groups,depth,reduce='amin',include_self=True)
        winner = torch.full((total,),len(dist),device=self.env.device,dtype=torch.long)
        winner.scatter_reduce_(0,groups,torch.where(depth==best[groups],index,len(dist)),reduce='amin',include_self=True)
        surface = torch.full((total,),-1,device=self.env.device,dtype=torch.long)
        present = winner < len(dist)
        surface[present] = selected['surfaces'][winner[present]]-self.surface_offsets_device[torch.arange(total,device=self.env.device)[present]//6]
        return self._mask_unknown(dict(active=(counts>0).reshape(-1,6),point=(points/counts.clamp_min(1)[:,None]).reshape(-1,6,3)-self.env.scene.env_origins[:,None],
            surface_local=surface.reshape(-1,6),force=force.reshape(-1,6,3),load=load.reshape(-1,6)))

    def read(self):
        if self.direct:return self._read_direct_reference()
        from somaforge_core.newton_contacts import snapshot_solver_contacts
        from somaforge_core.newton_support import select_support_contacts, support_motion

        snap = snapshot_solver_contacts(self.solver, self.model, include_support=True)
        if self._normal_error(snap):
            raise ValueError('Invalid active native constraint normal')
        selected = select_support_contacts(snap, self.binding, self.model.body_label,
                                           face_indexes=self.indexes, body_names_by_part=self.body_names)
        n = self.env.N * 6
        support = support_motion(snap, eligible=selected['eligible'], robot_side=selected['robot_side'],
                                 groups=selected['groups'], group_count=n)
        rows = np.flatnonzero(selected['eligible'])
        group = selected['groups'][rows]
        points, count, force = np.zeros((n,3),np.float32), np.zeros(n,np.int64), np.zeros((n,3),np.float32)
        np.add.at(points, group, snap['position_w'][rows])
        np.add.at(count, group, 1)
        sign = np.where(selected['robot_side'][rows] == 1, 1., -1.)
        np.add.at(force, group, snap['force_on_body1_w'][rows] * sign[:,None])
        surface=np.full(n,-1,np.int64)
        for g in np.unique(group):
            candidates=rows[group==g]
            winner=candidates[np.argmin(snap['dist'][candidates]-snap['includemargin'][candidates])]
            surface[g]=selected['surfaces'][winner]
        points /= count.clip(1)[:,None]
        points = points.reshape(self.env.N,6,3) - self.env.scene.env_origins[:,None].cpu().numpy()
        self.last_snapshot = {**snap, 'eligible': selected['eligible'], 'groups': selected['groups'],
                              'robot_side': selected['robot_side']}
        from somaforge_core.contact_system.evidence import BodyBinding,ContactEvidence,Sampling
        from .contact_control import PARTS
        from .motion import CONTROL_DT
        if not hasattr(self,'evidence_binding'):
            self.evidence_binding=BodyBinding(self.model.body_label,self.binding['body_env'],PARTS,
                                              self.body_names,self.env.device)
        known = self._known_mask()
        self.last_evidence=ContactEvidence(snap,
            sampling=Sampling(CONTROL_DT,float(self.env.sim.backend_manager.get_solver_dt())),
            known=known,control_step=self.env.ep_step,
            provenance=dict(backend='supermimic_mjwarp',selection='legacy_primary_upward_faces',parts=PARTS,
                robot_asset=self.env.execution_robot_asset,
                contact_response=getattr(self.env,'contact_response','legacy')))
        result = dict(active=count.reshape(self.env.N,6)>0, point=points,
                      surface=surface.reshape(self.env.N,6),
                      surface_local=np.where(surface.reshape(self.env.N,6)>=0,surface.reshape(self.env.N,6)-np.asarray(self.surface_offsets)[:,None],-1),
                      force=force.reshape(self.env.N,6,3),
                      load=np.asarray(support['normal_force_n'],dtype=np.float64).reshape(self.env.N,6),
                      slip=np.asarray(support['rms_tangent_speed_m_s'],dtype=np.float64).reshape(self.env.N,6),
                      known=known.cpu().numpy())
        from .contact_body_summary import summarize
        bodies=summarize(snap,self._summary_world,self._summary_slot,np.asarray(list(self.indexes)),
                         self.env.N,len(self.body_contact_metadata['body_contact_names']),
                         known[:,0].cpu().numpy())
        bodies['body_contact_point']-=self.env.scene.env_origins.cpu().numpy()[:,None]
        result.update(bodies)
        return self._mask_unknown({key: torch.as_tensor(value, device=self.env.device) for key,value in result.items()})
