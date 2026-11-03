"""Opt-in graph-safe canonical contact order and scoped MJWarp determinism.

No contact activation/geometry/material changes. Buffers are allocated once.
This implementation targets independent worlds with a bounded contact count;
overflow is an error, never silent contact truncation.
"""
import warp as wp


@wp.kernel(enable_backward=False)
def _bucket(n:wp.array[int], world:wp.array[int], counts:wp.array[int], buckets:wp.array2d[int], overflow:wp.array[int]):
    i=wp.tid()
    if i<n[0]:
        w=world[i];slot=wp.atomic_add(counts,w,1)
        if slot<buckets.shape[1]:buckets[w,slot]=i
        else:wp.atomic_max(overflow,0,1)


@wp.func
def _less(a:int,b:int,geom:wp.array[wp.vec2i],pos:wp.array[wp.vec3],dist:wp.array[float],frame:wp.array[wp.mat33]):
    for k in range(2):
        if geom[a][k]!=geom[b][k]:return geom[a][k]<geom[b][k]
    for k in range(3):
        if pos[a][k]!=pos[b][k]:return pos[a][k]<pos[b][k]
    if dist[a]!=dist[b]:return dist[a]<dist[b]
    for j in range(3):
        for k in range(3):
            if frame[a][j,k]!=frame[b][j,k]:return frame[a][j,k]<frame[b][j,k]
    return False


@wp.kernel(enable_backward=False)
def _sort(counts:wp.array[int], ends:wp.array[int], buckets:wp.array2d[int],geom:wp.array[wp.vec2i],pos:wp.array[wp.vec3],dist:wp.array[float],frame:wp.array[wp.mat33],order:wp.array[int],inverse:wp.array[int]):
    w=wp.tid();n=counts[w]
    if n>buckets.shape[1]:return
    for i in range(1,n):
        key=buckets[w,i];j=i-1
        while j>=0:
            if not _less(key,buckets[w,j],geom,pos,dist,frame):break
            buckets[w,j+1]=buckets[w,j];j-=1
        buckets[w,j+1]=key
    begin=ends[w]-n
    for i in range(n):
        old=buckets[w,i];order[begin+i]=old;inverse[old]=begin+i


@wp.kernel(enable_backward=False)
def _remap(n:wp.array[int],mapping:wp.array[int],inverse:wp.array[int],overflow:wp.array[int]):
    i=wp.tid();old=mapping[i]
    if overflow[0]==0 and old>=0 and old<n[0]:mapping[i]=inverse[old]


def _gather_kernels(dtype,ndim):
    if ndim==1:
        @wp.kernel(module='unique',enable_backward=False)
        def gather(n:wp.array[int],order:wp.array[int],src:wp.array(dtype=dtype),dst:wp.array(dtype=dtype),overflow:wp.array[int]):
            i=wp.tid()
            if i<n[0] and overflow[0]==0:dst[i]=src[order[i]]
        @wp.kernel(module='unique',enable_backward=False)
        def restore(n:wp.array[int],src:wp.array(dtype=dtype),dst:wp.array(dtype=dtype),overflow:wp.array[int]):
            i=wp.tid()
            if i<n[0] and overflow[0]==0:dst[i]=src[i]
    elif ndim==2:
        @wp.kernel(module='unique',enable_backward=False)
        def gather(n:wp.array[int],order:wp.array[int],src:wp.array2d(dtype=dtype),dst:wp.array2d(dtype=dtype),overflow:wp.array[int]):
            i,j=wp.tid()
            if i<n[0] and overflow[0]==0:dst[i,j]=src[order[i],j]
        @wp.kernel(module='unique',enable_backward=False)
        def restore(n:wp.array[int],src:wp.array2d(dtype=dtype),dst:wp.array2d(dtype=dtype),overflow:wp.array[int]):
            i,j=wp.tid()
            if i<n[0] and overflow[0]==0:dst[i,j]=src[i,j]
    else:raise ValueError(f'Unsupported contact field rank {ndim}')
    return gather,restore


class CanonicalContacts:
    def __init__(self,solver,*,per_world_capacity=256):
        self.solver=solver;d=solver.mjw_data;dev=solver.device;self.capacity=d.naconmax
        self.counts=wp.zeros(d.nworld,dtype=int,device=dev);self.ends=wp.zeros_like(self.counts)
        self.buckets=wp.empty((d.nworld,per_world_capacity),dtype=int,device=dev)
        self.order=wp.empty(d.naconmax,dtype=int,device=dev);self.inverse=wp.empty_like(self.order)
        self.overflow=wp.zeros(1,dtype=int,device=dev)
        self.fields=[];cache={}
        for name,value in vars(d.contact).items():
            if isinstance(value,wp.array) and value.ndim and value.shape[0]==d.naconmax:
                key=(value.dtype,value.ndim)
                if key not in cache:cache[key]=_gather_kernels(*key)
                self.fields.append((value,wp.empty_like(value),*cache[key]))
        self.original=solver._convert_contacts_to_mjwarp
        solver._convert_contacts_to_mjwarp=self.convert

    def convert(self,*args,**kwargs):
        # Cached converter IDs would refer to the pre-permutation layout.
        self.solver._invalidate_contact_fast_path()
        self.original(*args,**kwargs)
        d=self.solver.mjw_data;c=d.contact;dev=self.solver.device
        self.counts.zero_()
        wp.launch(_bucket,dim=self.capacity,inputs=[d.nacon,c.worldid,self.counts,self.buckets,self.overflow],device=dev)
        wp.utils.array_scan(self.counts,self.ends,inclusive=True)
        wp.launch(_sort,dim=d.nworld,inputs=[self.counts,self.ends,self.buckets,c.geom,c.pos,c.dist,c.frame,self.order,self.inverse],device=dev)
        for value,scratch,gather,restore in self.fields:
            wp.launch(gather,dim=value.shape,inputs=[d.nacon,self.order,value,scratch,self.overflow],device=dev)
            wp.launch(restore,dim=value.shape,inputs=[d.nacon,scratch,value,self.overflow],device=dev)
        wp.launch(_remap,dim=self.solver._contact_tid_to_cid.shape,inputs=[d.nacon,self.solver._contact_tid_to_cid,self.inverse,self.overflow],device=dev)

    def check(self):
        if int(self.overflow.numpy()[0]):raise RuntimeError('Canonical native contact bucket capacity exceeded')


def install(solver,*,scope='solver',per_world_capacity=256):
    """Scope selects the deterministic module set; simulation geometry is unchanged."""
    from newton._src.solvers.mujoco.solver_mujoco import _mujoco_warp_deterministic_max_records
    from newton._src.sensors import sensor_contact
    import mujoco_warp._src.solver as mj_solver
    import mujoco_warp._src.constraint as constraint
    sorter=CanonicalContacts(solver,per_world_capacity=per_world_capacity)
    solver._deterministic=wp.DeterministicMode.RUN_TO_RUN
    solver._deterministic_max_records=_mujoco_warp_deterministic_max_records(solver.mj_model,solver.mjw_data)
    if scope in ('full','bounded'):
        solver._set_mujoco_warp_module_options()
        if scope=='bounded':
            if solver.mjw_model.is_sparse:raise ValueError('Bounded mode requires the dense Jacobian path')
            _bounded_dense_kernels(solver)
    elif scope=='solver':
        for module in (mj_solver,constraint):
            solver._set_module_options({'deterministic':wp.DeterministicMode.RUN_TO_RUN,'deterministic_max_records':0},module=module)
    else:raise ValueError(scope)
    solver._prepare_generated_kernels()
    wp.set_module_options({'deterministic':wp.DeterministicMode.RUN_TO_RUN},module=sensor_contact)
    return sorter


def _bounded_dense_kernels(solver):
    """Avoid applying the generic 1000-record bound to one-atomic kernels.

    Warp infers exact static bounds for these kernels. Dynamic kernels keep
    upstream conservative bounds; no physical contact/constraint cap is reduced.
    """
    import mujoco_warp._src.smooth as smooth
    import mujoco_warp._src.constraint as constraint
    # Dense generated kernels have no variable-length atomic Hessian reductions.
    solver._deterministic_max_records=0
    for name in ('_subtree_com_acc','_crb_accumulate','_qLD_acc','_cfrc_backward',
                 '_cfrc_ext_contact','_linear_momentum','_angular_momentum'):
        kernel=getattr(smooth,name)
        original=getattr(kernel,'_canonical_original',kernel)
        replacement=wp.kernel(module='unique',enable_backward=False,
            module_options={'deterministic':wp.DeterministicMode.RUN_TO_RUN,'deterministic_max_records':0})(original.func)
        replacement._canonical_original=original
        setattr(smooth,name,replacement)

    # Linesearch already uses fixed tile reductions. Its only global atomic is
    # bitwise OR on overflow flags, which is order-independent. Rewriting that
    # tiled kernel as a deterministic scatter causes spurious buffer overflow.
    import mujoco_warp._src.solver as mj_solver
    factory=getattr(mj_solver._linesearch_iterative_kernel,'_canonical_original',mj_solver._linesearch_iterative_kernel)
    def linesearch(*args,**kwargs):
        kernel=factory(*args,**kwargs)
        wp.set_module_options({'deterministic':wp.DeterministicMode.NOT_GUARANTEED},module=kernel.module)
        return kernel
    linesearch._canonical_original=factory
    mj_solver._linesearch_iterative_kernel=linesearch
