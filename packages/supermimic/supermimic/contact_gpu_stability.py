"""Graph-safe native substep loss witnesses; no host-sized contact compaction.

The shared primary-face kernel is unchanged. Loaded rows are linked by part;
matching uses the reference greedy global minimum and native row-order ties.
Storage is bounded by the solver's actual contact capacity, never a row cap.
"""
import torch
import warp as wp
from somaforge_core.live_contact_device import DeviceSupportContactSelector, _primary_by_shape
from .contact_stability import CONTRACT

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.func
def _score(v:wp.vec3):
    # Match PyTorch CUDA's three-component sum reduction (x + z) + y.
    # A different association can change native-row tie breaking.
    return (v[0]*v[0]+v[2]*v[2])+v[1]*v[1]


@wp.kernel
def _prepare(n:wp.array[int],dist:wp.array[float],margin:wp.array[float],kind:wp.array[int],
             dim:wp.array[int],address:wp.array2d[int],world:wp.array[int],geom:wp.array[wp.vec2i],
             mapping:wp.array2d[int],shape_body:wp.array[int],body_world:wp.array[wp.int64],
             body_part:wp.array[wp.int64],pos:wp.array[wp.vec3],frame:wp.array[wp.mat33],
             first:wp.array[wp.int64],cone:int,observed:wp.array[wp.bool],
             query:wp.array[wp.bool],external:wp.array[wp.int64],points:wp.array[wp.vec3d],
             normals:wp.array[wp.vec3d],groups:wp.array[int],side:wp.array[int],
             robot_shape:wp.array[int],terrain_shape:wp.array[int],errors:wp.array[int]):
    i=wp.tid()
    query[i]=False;groups[i]=-1
    if i==0 and (n[0]<0 or n[0]>dist.shape[0]):wp.atomic_or(errors,0,1)
    if i>=n[0]:return
    w=world[i];g=geom[i]
    if w<0 or w>=mapping.shape[0] or g[0]<0 or g[0]>=mapping.shape[1] or g[1]<0 or g[1]>=mapping.shape[1]:
        wp.atomic_or(errors,0,2);return
    s0=mapping[w,g[0]];s1=mapping[w,g[1]]
    if s0<0 or s0>=shape_body.shape[0] or s1<0 or s1>=shape_body.shape[0]:
        wp.atomic_or(errors,0,2);return
    needed=dim[i]
    if cone==0 and dim[i]>1:needed=2*(dim[i]-1)
    if needed<1 or needed>address.shape[1] or not wp.isfinite(dist[i]) or not wp.isfinite(margin[i]):
        wp.atomic_or(errors,0,4);return
    active=(kind[i]&1)!=0 and dist[i]<margin[i]
    normal=wp.vec3(frame[i][0,0],frame[i][0,1],frame[i][0,2])
    p=pos[i]
    for axis in range(3):
        for other in range(3):
            if not wp.isfinite(frame[i][axis,other]):wp.atomic_or(errors,0,8)
    if not wp.isfinite(wp.length(p)) or not wp.isfinite(wp.length(normal)):
        wp.atomic_or(errors,0,8);return
    if active and wp.abs(wp.length(normal)-1.0)>0.000011:wp.atomic_or(errors,0,8)
    if not active:return
    for j in range(needed):
        if address[i,j]<0:wp.atomic_or(errors,0,16);return
    b0=shape_body[s0];b1=shape_body[s1]
    w0=int(-1);w1=int(-1);p0=int(-1);p1=int(-1)
    if b0>=0 and b0<body_world.shape[0]:w0=int(body_world[b0]);p0=int(body_part[b0])
    if b1>=0 and b1<body_world.shape[0]:w1=int(body_world[b1]);p1=int(body_part[b1])
    if (w0>=0)==(w1>=0):return
    sd=int(0);part=p0;own=w0;rs=s0;ts=s1;sign=float(-1.0)
    if w1>=0:sd=1;part=p1;own=w1;rs=s1;ts=s0;sign=1.0
    if part<0 or own!=w:return
    if ts<0 or ts>=first.shape[0]:wp.atomic_or(errors,0,32);return
    if first[ts]<0:wp.atomic_or(errors,0,32);return
    groups[i]=w*6+part;side[i]=sd;robot_shape[i]=rs;terrain_shape[i]=ts
    external[i]=wp.int64(ts);points[i]=wp.vec3d(p);normals[i]=wp.vec3d(normal)*wp.float64(sign)
    query[i]=observed[w]


@wp.kernel
def _link(n:wp.array[int],groups:wp.array[int],query:wp.array[wp.bool],face:wp.array[wp.int64],upward:wp.array[wp.bool],
          side:wp.array[int],force:wp.array[wp.spatial_vector],pos:wp.array[wp.vec3],
          part_body:wp.array[wp.int64],xpos:wp.array2d[wp.vec3],xmat:wp.array2d[wp.mat33],
          heads:wp.array[int],next_row:wp.array[int],local:wp.array[wp.vec3],
          load:wp.array[float],errors:wp.array[int]):
    i=wp.tid();load[i]=0.0
    if i<n[0]:
        for axis in range(6):
            if not wp.isfinite(force[i][axis]):wp.atomic_or(errors,0,128)
    if not query[i]:return
    f=int(face[i])
    if f<0:wp.atomic_or(errors,0,64);return
    if not upward[f]:return
    g=groups[i];w=g//6;b=int(part_body[g%6])
    load[i]=force[i][2]*wp.where(side[i]==1,1.0,-1.0)
    if not wp.isfinite(load[i]):wp.atomic_or(errors,0,128);return
    local[i]=wp.transpose(xmat[w,b])*(pos[i]-xpos[w,b])
    next_row[i]=wp.atomic_exch(heads,g,i)


@wp.kernel
def _qualify(heads:wp.array[int],next_row:wp.array[int],local:wp.array[wp.vec3],pos:wp.array[wp.vec3],
             load:wp.array[float],robot_shape:wp.array[int],terrain_shape:wp.array[int],
             old_heads:wp.array[int],old_next:wp.array[int],old_local:wp.array[wp.vec3],old_pos:wp.array[wp.vec3],
             old_load:wp.array[float],old_robot:wp.array[int],old_terrain:wp.array[int],
             used:wp.array[int],old_used:wp.array[int],part_body:wp.array[wp.int64],
             xpos:wp.array2d[wp.vec3],xmat:wp.array2d[wp.mat33],targets:wp.array[wp.vec3],origins:wp.array[wp.vec3],
             ep_step:wp.array[wp.int64],old_step:wp.array[wp.int64],substep:wp.array[int],substeps:int,
             dt:float,min_load:float,relative_load:float,max_speed:float,match_distance:float,
             stable:wp.array[wp.bool],row_stable:wp.array[wp.bool],load_stable:wp.array[wp.bool],
             material_speed:wp.array[float],total_load:wp.array[float],qualified:wp.array[wp.bool],witness:wp.array[wp.vec3]):
    g=wp.tid();w=g//6;b=int(part_body[g%6]);step=ep_step[w]*wp.int64(substeps)+wp.int64(substep[0])
    consecutive=step==old_step[g]+wp.int64(1)
    if substep[0]==0:qualified[g]=False;witness[g]=wp.vec3(0.0)
    count=int(0);old_count=int(0);mass=float(0.0);previous_mass=float(0.0);speed=float(0.0)
    best_point=int(-1);best_score=float(wp.inf)
    i=heads[g]
    while i>=0:
        if load[i]>0.0:
            score=_score(pos[i]-targets[g])
            if score<best_score or (score==best_score and (best_point<0 or i<best_point)):
                best_score=score;best_point=i
            count+=1;mass+=load[i]
        i=next_row[i]
    j=old_heads[g]
    while j>=0:
        if old_load[j]>0.0:
            old_count+=1;previous_mass+=old_load[j]
            speed=wp.max(speed,wp.length(xpos[w,b]+xmat[w,b]*old_local[j]-old_pos[j])/dt)
        j=old_next[j]
    matched=int(0)
    # Stop only when no finite compatible pair remains. The reference's later
    # infinite-distance assignments cannot contribute to the bijection count.
    for k in range(wp.min(count,old_count)):
        best=float(wp.inf);bi=int(-1);bj=int(-1);i=heads[g]
        while i>=0:
            if load[i]>0.0 and used[i]==0:
                j=old_heads[g]
                while j>=0:
                    if old_load[j]>0.0 and old_used[j]==0 and robot_shape[i]==old_robot[j] and terrain_shape[i]==old_terrain[j]:
                        distance=wp.length(local[i]-old_local[j])
                        if distance<best or (distance==best and (bi<0 or i<bi or (i==bi and j<bj))):
                            best=distance;bi=i;bj=j
                    j=old_next[j]
            i=next_row[i]
        if bi<0:break
        used[bi]=1;old_used[bj]=1
        if best<=match_distance:matched+=1
    rows=consecutive and count>0 and count==old_count and matched==count
    loads=consecutive and mass>min_load and previous_mass>min_load and wp.abs(mass-previous_mass)<=relative_load*wp.max(mass,previous_mass)
    speed=wp.where(consecutive,speed,0.0)
    ok=rows and loads and speed<=max_speed
    stable[g]=ok;row_stable[g]=rows;load_stable[g]=loads;material_speed[g]=speed;total_load[g]=mass
    if ok:
        qualified[g]=True;witness[g]=pos[best_point]-origins[w]
    old_step[g]=step


@wp.kernel
def _filter_initial_support(heads:wp.array[int],observed:wp.array[wp.bool],ep_step:wp.array[wp.int64],
        substep:wp.array[int],exit_steps:int,initial_reference:wp.array[wp.bool],raw_stable:wp.array[wp.bool],raw_point:wp.array[wp.vec3],
        targets:wp.array[wp.vec3],origins:wp.array[wp.vec3],tolerance:wp.array[float],
        initial_support:wp.array[wp.bool],absence:wp.array[int],
        qualified:wp.array[wp.bool],witness:wp.array[wp.vec3],
        part_body:wp.array[wp.int64],xpos:wp.array2d[wp.vec3],xmat:wp.array2d[wp.mat33],
        witness_position:wp.array[wp.vec3],witness_rotation:wp.array[wp.mat33],witness_candidate:wp.array[wp.int32],
        guidance_qualified:wp.array[wp.bool],guidance_point:wp.array[wp.vec3],guidance_position:wp.array[wp.vec3],guidance_rotation:wp.array[wp.mat33],
        material_enabled:wp.array[wp.bool],material_target:wp.array[wp.vec3],material_goal:wp.array[wp.vec3],material_valid:wp.array[wp.bool],material_candidates:int,material_radius:float,
        next_row:wp.array[int],row_local:wp.array[wp.vec3],row_load:wp.array[float],row_pos:wp.array[wp.vec3],finished:wp.array[wp.bool],initial_absence:wp.array[wp.int32],motion_release:wp.array[wp.bool]):
    g=wp.tid();world=g//6
    if substep[0]==0:
        qualified[g]=False;witness[g]=wp.vec3(0.0)
        witness_position[g]=wp.vec3(0.0);witness_rotation[g]=wp.mat33(0.0);witness_candidate[g]=0
        guidance_qualified[g]=False;guidance_point[g]=wp.vec3(0.0);guidance_position[g]=wp.vec3(0.0);guidance_rotation[g]=wp.mat33(0.0)
        if ep_step[world]==0:
            initial_support[g]=initial_reference[g]
            absence[g]=initial_absence[g]
    if motion_release[g]:initial_support[g]=False
    if initial_support[g] and observed[world]:
        if heads[g]<0:
            absence[g]+=1
            if absence[g]>=exit_steps:initial_support[g]=False
        else:
            absence[g]=0
    if raw_stable[g] and not initial_support[g]:
        body=int(part_body[g%6]);guidance_qualified[g]=True;guidance_point[g]=raw_point[g]
        guidance_position[g]=xpos[world,body]-origins[world];guidance_rotation[g]=xmat[world,body]
    if raw_stable[g]:
        material_match=not material_enabled[g]
        chosen=int(-1);chosen_candidate=int(-1);distance=float(wp.inf);i=heads[g]
        while i>=0:
            if row_load[i]>0.0:
                for candidate in range(material_candidates):
                    ci=g*material_candidates+candidate
                    if material_valid[ci]:
                        d=wp.length(row_local[i]-material_target[ci])
                        if d<distance:distance=d;chosen=i;chosen_candidate=ci
            i=next_row[i]
        if material_enabled[g]:material_match=chosen>=0 and distance<=material_radius
        selected_point=raw_point[g];selected_goal=targets[g]
        if material_enabled[g] and material_match:
            selected_point=row_pos[chosen]-origins[world];selected_goal=material_goal[chosen_candidate]
        in_goal=wp.length(selected_point+origins[world]-selected_goal)<=tolerance[g]
        if material_match and not initial_support[g]:
            qualified[g]=True;witness[g]=selected_point
            body=int(part_body[g%6])
            witness_position[g]=xpos[world,body]-origins[world]
            witness_rotation[g]=xmat[world,body]
            if material_enabled[g]:witness_candidate[g]=chosen_candidate%material_candidates


@wp.kernel
def _advance(substep:wp.array[int],substeps:int):
    substep[0]=(substep[0]+1)%substeps


@wp.kernel
def _onset_snapshot(stop:wp.array[int],substep:wp.array[int],blocked:wp.array[wp.bool],
        counts:wp.array[int],heads:wp.array[int],next_row:wp.array[int],points:wp.array[wp.vec3],
        part_body:wp.array[wp.int64],xpos:wp.array2d[wp.vec3],xmat:wp.array2d[wp.mat33],
        origins:wp.array[wp.vec3],targets:wp.array[wp.vec3],qualified:wp.array[wp.bool],guidance:wp.array[wp.bool],
        event_position:wp.array[wp.vec3],event_rotation:wp.array[wp.mat33],event_active:wp.array[wp.bool],
        event_primary:wp.array[wp.bool],event_point:wp.array[wp.vec3]):
    g=wp.tid();world=g//6;body=int(part_body[g%6])
    # Raw side/edge contact keeps an entry support blocked as well.
    if blocked[g]:
        qualified[g]=False;guidance[g]=False
    if stop[g]!=substep[0]:return
    event_position[g]=xpos[world,body]-origins[world]
    event_rotation[g]=xmat[world,body]
    event_active[g]=counts[g]>0
    event_primary[g]=heads[g]>=0
    i=heads[g];best=float(wp.inf);chosen=int(-1)
    while i>=0:
        distance=_score(points[i]-targets[g])
        if distance<best or (distance==best and (chosen<0 or i<chosen)):
            best=distance;chosen=i
        i=next_row[i]
    event_point[g]=wp.vec3(0.0)
    if chosen>=0:event_point[g]=points[chosen]-origins[world]


class GPUContactLossMonitor:
    """Runs fixed-storage Warp kernels in the solver CUDA graph.

    Validation reports malformed solver evidence once at the control boundary;
    it never turns invalid evidence into a missing-contact guidance term.
    """
    def __init__(self,env,reader,material_candidates=1):
        self.env=env;self.reader=reader;self.manager=env.sim.backend_manager
        self.dt=float(self.manager.get_solver_dt());self.substeps=int(self.manager._num_substeps)
        self.solver=reader.solver;self.device=str(reader.model.device)
        self.selector=DeviceSupportContactSelector(reader.binding,reader.model.body_label,env.device,body_names_by_part=reader.body_names)
        self.capacity=len(self.solver.mjw_data.contact.dist);self.groups=env.N*6;self.material_candidates=material_candidates
        self.arrays={};self.previous={}
        for name,dtype in [('heads',wp.int32),('next',wp.int32),('local',wp.vec3),('pos',wp.vec3),('load',wp.float32),('robot',wp.int32),('terrain',wp.int32)]:
            size=self.groups if name=='heads' else self.capacity
            self.arrays[name]=wp.full(size,-1,dtype=dtype,device=self.device) if name=='heads' else wp.zeros(size,dtype=dtype,device=self.device)
            self.previous[name]=wp.clone(self.arrays[name])
        alloc=lambda n,dtype:wp.zeros(n,dtype=dtype,device=self.device)
        self.query=alloc(self.capacity,wp.bool);self.external=alloc(self.capacity,wp.int64)
        self.points=alloc(self.capacity,wp.vec3d);self.normals=alloc(self.capacity,wp.vec3d)
        self.row_groups=alloc(self.capacity,wp.int32);self.side=alloc(self.capacity,wp.int32);self.face=alloc(self.capacity,wp.int64)
        self.force=alloc(self.capacity,wp.spatial_vector);self.contact_ids=wp.array(range(self.capacity),dtype=wp.int32,device=self.device)
        self.used=alloc(self.capacity,wp.int32);self.old_used=alloc(self.capacity,wp.int32)
        self.errors=alloc(1,wp.int32);self.substep=alloc(1,wp.int32)
        self.old_step=wp.full(self.groups,-2,dtype=wp.int64,device=self.device)
        self.ep_step=alloc(env.N,wp.int64);self.targets=alloc(self.groups,wp.vec3)
        self.origins=wp.from_torch(env.scene.env_origins.contiguous(),dtype=wp.vec3)
        self.part_body=wp.array(reader.part_body_ids,dtype=wp.int64,device=self.device)
        self.observed=wp.from_torch(reader._known_mask(solved=True)[:,0].contiguous())
        self.features={k:alloc(self.groups,dtype) for k,dtype in [('contact_stable',wp.bool),('contact_row_stable',wp.bool),('contact_load_stable',wp.bool),('contact_material_speed',wp.float32),('contact_load',wp.float32)]}
        self.qualified=alloc(self.groups,wp.bool);self.witness=alloc(self.groups,wp.vec3)
        self.guidance_qualified=alloc(self.groups,wp.bool);self.guidance_point=alloc(self.groups,wp.vec3)
        self.guidance_position=alloc(self.groups,wp.vec3);self.guidance_rotation=alloc(self.groups,wp.mat33)
        from .contact_temporal import TEMPORAL_CONTRACT
        import math
        self.exit_steps=math.ceil(TEMPORAL_CONTRACT['exit_frames']*TEMPORAL_CONTRACT['dt']/self.dt)
        from .contact_onset import NativeContactOnset
        self.onset=NativeContactOnset(env.N,self.device,self.exit_steps)
        from .contact_collision_gpu import CollisionMonitor
        self.collision=CollisionMonitor(env.N,self.capacity,material_candidates,self.device,self.dt)
        self.event_position=alloc(self.groups,wp.vec3)
        self.event_rotation=alloc(self.groups,wp.mat33)
        self.event_active=alloc(self.groups,wp.bool)
        self.event_primary=alloc(self.groups,wp.bool)
        self.event_point=alloc(self.groups,wp.vec3)
        self.initial_support=alloc(self.groups,wp.bool);self.absence=alloc(self.groups,wp.int32);self.initial_absence=alloc(self.groups,wp.int32)
        self.filtered_qualified=alloc(self.groups,wp.bool);self.filtered_witness=alloc(self.groups,wp.vec3)
        self.witness_position=alloc(self.groups,wp.vec3);self.witness_rotation=alloc(self.groups,wp.mat33);self.witness_candidate=alloc(self.groups,wp.int32)
        self.tolerance=alloc(self.groups,wp.float32);self.initial_reference=alloc(self.groups,wp.bool)
        self.material_enabled=alloc(self.groups,wp.bool);self.material_target=alloc(self.groups*material_candidates,wp.vec3)
        self.material_goal=alloc(self.groups*material_candidates,wp.vec3);self.material_valid=alloc(self.groups*material_candidates,wp.bool)
        self.cone=int(self.solver.mjw_model.opt.cone)
        if self.cone not in (0,1):raise ValueError('Unknown native contact cone')

    def __enter__(self):
        self.original=self.manager._step_solver
        self.manager._step_solver=self._step;self.manager._invalidate_graph()
        return self

    def reset(self,ids):
        """Clear per-episode substep evidence after a new reset/stage handoff."""
        ids=torch.as_tensor(ids,dtype=torch.long,device=self.env.device)
        wp.to_torch(self.old_step).reshape(self.env.N,6)[ids]=-2
        wp.to_torch(self.initial_support).reshape(self.env.N,6)[ids]=False
        wp.to_torch(self.absence).reshape(self.env.N,6)[ids]=0
        wp.to_torch(self.filtered_qualified).reshape(self.env.N,6)[ids]=False
        wp.to_torch(self.filtered_witness).reshape(self.env.N,6,3)[ids]=0
        self.substep.zero_();self.errors.zero_()

    def begin_stage(self,ids,initial_active,required,initial_absence=None):
        self.reset(ids)
        self.onset.begin_stage(ids,initial_active,required,initial_absence)
        wp.to_torch(self.initial_absence).reshape(self.env.N,6)[ids]=0 if initial_absence is None else initial_absence
        wp.to_torch(self.absence).reshape(self.env.N,6)[ids]=0 if initial_absence is None else initial_absence
        self.collision.reset(ids)
        wp.to_torch(self.initial_support).reshape(self.env.N,6)[ids]=initial_active
        wp.to_torch(self.initial_reference).reshape(self.env.N,6)[ids]=initial_active

    def begin_control(self,ids,tolerance,initial_reference,material_local=None,material_goal=None,material_valid=None):
        self.onset.begin_control()
        wp.to_torch(self.ep_step).copy_(self.env.ep_step)
        wp.to_torch(self.targets).copy_(self.reader.targets.reshape(-1,3))
        wp.to_torch(self.tolerance).reshape(self.env.N,6)[ids]=tolerance.expand(-1,6)
        wp.to_torch(self.initial_reference).reshape(self.env.N,6)[ids]=initial_reference
        enabled=wp.to_torch(self.material_enabled).reshape(self.env.N,6)
        enabled[ids]=False
        if material_local is not None:
            enabled[ids,4:]=True
            if material_local.ndim==3:
                material_local=material_local.unsqueeze(2)
                material_goal=self.reader.targets[ids].unsqueeze(2)
                material_valid=torch.ones(material_local.shape[:-1],device=material_local.device,dtype=torch.bool)
            wp.to_torch(self.material_target).reshape(self.env.N,6,self.material_candidates,3)[ids]=material_local
            wp.to_torch(self.material_goal).reshape(self.env.N,6,self.material_candidates,3)[ids]=material_goal
            wp.to_torch(self.material_valid).reshape(self.env.N,6,self.material_candidates)[ids]=material_valid
        wp.get_stream(self.device).wait_stream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
        self.errors.zero_()

    def _step(self,*args,**kwargs):
        result=self.original(*args,**kwargs)
        self.sample()
        return result

    def sample(self):
        import mujoco_warp as mjw
        d=self.solver.mjw_data;c=d.contact;s=self.selector;a=self.arrays;o=self.previous
        a['heads'].fill_(-1);self.used.zero_();self.old_used.zero_()
        wp.launch(_prepare,self.capacity,inputs=[d.nacon,c.dist,c.includemargin,c.type,c.dim,c.efc_address,c.worldid,c.geom,
            self.solver.mjc_geom_to_newton_shape,self.reader.model.shape_body,wp.from_torch(s.body_world),wp.from_torch(s.body_part),c.pos,c.frame,
            wp.from_torch(s.first),self.cone,self.observed,self.query,self.external,self.points,self.normals,self.row_groups,self.side,a['robot'],a['terrain'],self.errors],device=self.device)
        wp.launch(_primary_by_shape,self.capacity,inputs=[self.points,self.normals,self.query,self.external,wp.from_torch(s.first),wp.from_torch(s.stop),s.vertices,s.owners,s.face_normals,wp.from_torch(s.surfaces),self.face],device=self.device)
        self.collision.classify(self.row_groups,self.query,c.pos,self.face,wp.from_torch(s.upward))
        self.onset.set_positions(self.part_body,d.xpos)
        self.onset.sample(self.row_groups,self.query,self.observed,self.substep,self.collision.counts)
        mjw.contact_force(self.solver.mjw_model,d,self.contact_ids,True,self.force)
        wp.launch(_link,self.capacity,inputs=[d.nacon,self.row_groups,self.query,self.face,wp.from_torch(s.upward),self.side,self.force,c.pos,self.part_body,d.xpos,d.xmat,a['heads'],a['next'],a['local'],a['load'],self.errors],device=self.device)
        wp.copy(a['pos'],c.pos)
        f=self.features
        wp.launch(_qualify,self.groups,inputs=[a['heads'],a['next'],a['local'],a['pos'],a['load'],a['robot'],a['terrain'],o['heads'],o['next'],o['local'],o['pos'],o['load'],o['robot'],o['terrain'],self.used,self.old_used,self.part_body,d.xpos,d.xmat,self.targets,self.origins,self.ep_step,self.old_step,self.substep,self.substeps,self.dt,CONTRACT['min_load_n'],CONTRACT['max_relative_load_change'],CONTRACT['max_material_speed_m_s'],CONTRACT['row_match_distance_m'],f['contact_stable'],f['contact_row_stable'],f['contact_load_stable'],f['contact_material_speed'],f['contact_load'],self.qualified,self.witness],device=self.device)
        wp.launch(_filter_initial_support,self.groups,inputs=[a['heads'],self.observed,self.ep_step,self.substep,self.exit_steps,self.initial_reference,
            f['contact_stable'],self.witness,self.targets,self.origins,self.tolerance,self.initial_support,self.absence,
            self.filtered_qualified,self.filtered_witness,self.part_body,d.xpos,d.xmat,
            self.witness_position,self.witness_rotation,self.witness_candidate,self.guidance_qualified,self.guidance_point,self.guidance_position,self.guidance_rotation,self.material_enabled,self.material_target,self.material_goal,self.material_valid,self.material_candidates,CONTRACT['row_match_distance_m'],
            a['next'],a['local'],a['load'],a['pos'],self.onset.done,self.initial_absence,self.onset.motion_release],device=self.device)
        wp.launch(_onset_snapshot,self.groups,inputs=[self.onset.stop_substep,self.substep,
            self.onset.blocked,self.onset.counts,a['heads'],a['next'],c.pos,self.part_body,
            d.xpos,d.xmat,self.origins,self.targets,self.filtered_qualified,self.guidance_qualified,
            self.event_position,self.event_rotation,self.event_active,self.event_primary,self.event_point],device=self.device)
        self.collision.sample(self.row_groups,self.query,c.pos,self.normals,c.dist,self.onset.required,self.onset.blocked,self.part_body,d.xpos,d.xmat,self.force)
        self.onset.confirm(self.filtered_qualified,self.observed)
        for key in a:wp.copy(o[key],a[key])
        wp.launch(_advance,1,inputs=[self.substep,self.substeps],device=self.device)

    def summary(self):
        wp.stream_from_torch(torch.cuda.current_stream(self.device)).wait_stream(wp.get_stream(self.device))
        error=int(wp.to_torch(self.errors)[0])
        if error:raise ValueError(f'Invalid native substep contact evidence (GPU flags {error})')
        result={k:wp.to_torch(v).reshape(self.env.N,6) for k,v in self.features.items()}
        result.update(contact_stable=wp.to_torch(self.filtered_qualified).reshape(self.env.N,6),contact_loss_point=wp.to_torch(self.filtered_witness).reshape(self.env.N,6,3))
        result.update(contact_raw_stable=wp.to_torch(self.guidance_qualified).reshape(self.env.N,6),contact_raw_point=wp.to_torch(self.guidance_point).reshape(self.env.N,6,3),contact_raw_position=wp.to_torch(self.guidance_position).reshape(self.env.N,6,3),contact_raw_rotation=wp.to_torch(self.guidance_rotation).reshape(self.env.N,6,3,3))
        result.update(contact_loss_position=wp.to_torch(self.witness_position).reshape(self.env.N,6,3),
                      contact_loss_rotation=wp.to_torch(self.witness_rotation).reshape(self.env.N,6,3,3),contact_loss_candidate=wp.to_torch(self.witness_candidate).reshape(self.env.N,6))
        result.update(contact_new_event=wp.to_torch(self.onset.event).reshape(self.env.N,6).clone(),
            contact_event_parts=wp.to_torch(self.onset.event_parts).reshape(self.env.N,6).clone(),
            contact_confirming=(wp.to_torch(self.onset.pending)&~wp.to_torch(self.onset.done)).reshape(self.env.N,6).clone(),
            contact_confirmation_lost=wp.to_torch(self.onset.lost).reshape(self.env.N,6).clone(),
            contact_confirmation_done=wp.to_torch(self.onset.done).reshape(self.env.N,6).clone(),
            contact_confirmation_steps=wp.to_torch(self.onset.confirmation_steps).reshape(self.env.N,6).clone(),
            contact_stop_substep=wp.to_torch(self.onset.stop_substep).reshape(self.env.N,6).clone(),
            contact_entry_blocked=wp.to_torch(self.onset.blocked).reshape(self.env.N,6).clone(),
            contact_native_active=(wp.to_torch(self.onset.counts).reshape(self.env.N,6)>0),
            contact_event_position=wp.to_torch(self.event_position).reshape(self.env.N,6,3).clone(),
            contact_event_rotation=wp.to_torch(self.event_rotation).reshape(self.env.N,6,3,3).clone(),
            contact_event_active=wp.to_torch(self.event_active).reshape(self.env.N,6).clone(),
            contact_event_primary=wp.to_torch(self.event_primary).reshape(self.env.N,6).clone(),
            contact_event_point=wp.to_torch(self.event_point).reshape(self.env.N,6,3).clone())
        result.update(self.collision.summary())
        return result

    def __exit__(self,*args):
        self.manager._step_solver=self.original;self.manager._invalidate_graph()
