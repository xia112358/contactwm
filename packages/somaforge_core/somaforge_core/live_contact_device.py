"""GPU attribution at actual solver force points; no collision or integration.

Float32 BVHs are conservative broad phases only. Final finite-triangle distance,
normal ordering and the existing numerical tie rule use float64. Activation and
allocation always come from the actual Newton/MJWarp constraint tensors.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

wp.set_module_options({'enable_backward':False, 'fast_math':False, 'fuse_fp':False})


@wp.func
def _distance(p:wp.vec3d, a:wp.vec3d, b:wp.vec3d, c:wp.vec3d):
    u = b-a; v = c-a; w = p-a
    uu = wp.dot(u,u); uv = wp.dot(u,v); vv = wp.dot(v,v)
    wu = wp.dot(w,u); wv = wp.dot(w,v)
    den = uu*vv-uv*uv
    s = (wu*vv-wv*uv)/den; r = (wv*uu-wu*uv)/den
    d = wp.float64(1.0e308)
    if s >= wp.float64(0.0) and r >= wp.float64(0.0) and s+r <= wp.float64(1.0):
        delta = a+s*u+r*v-p
        d = wp.dot(delta,delta)
    for i in range(3):
        start = a; end = b
        if i == 1:
            start = b; end = c
        elif i == 2:
            start = c; end = a
        edge = end-start
        t = wp.clamp(wp.dot(p-start,edge)/wp.dot(edge,edge),wp.float64(0.0),wp.float64(1.0))
        delta = start+t*edge-p
        d = wp.min(d,wp.dot(delta,delta))
    return d


@wp.kernel
def _bound(mesh:wp.uint64, points:wp.array(dtype=wp.vec3d), task:wp.array(dtype=wp.bool),
           external:wp.array(dtype=wp.int64), first:wp.array(dtype=wp.int64),
           triangle_shape:wp.array(dtype=wp.int64), vertices:wp.array(dtype=wp.vec3d),
           radius:wp.array(dtype=wp.float64)):
    i = wp.tid()
    if task[i]:
        p = points[i]
        triangle = int(first[external[i]])
        approximate = wp.mesh_query_point_no_sign(mesh,wp.vec3(p),wp.float32(1.0e30))
        if approximate.result:
            if triangle_shape[approximate.face] == external[i]:
                triangle = approximate.face
        d = _distance(p,vertices[3*triangle],vertices[3*triangle+1],vertices[3*triangle+2])
        radius[i] = wp.sqrt(d+wp.float64(1.0e-12)+wp.float64(1.0e-9)*wp.abs(d))


@wp.kernel
def _primary(bvh:wp.uint64, points:wp.array(dtype=wp.vec3d), normals:wp.array(dtype=wp.vec3d),
             task:wp.array(dtype=wp.bool), external:wp.array(dtype=wp.int64),
             lower:wp.array(dtype=wp.vec3), upper:wp.array(dtype=wp.vec3),
             vertices:wp.array(dtype=wp.vec3d), triangle_shape:wp.array(dtype=wp.int64),
             owner:wp.array(dtype=wp.int64), face_normal:wp.array(dtype=wp.vec3d),
             surface_ids:wp.array(dtype=wp.int64), output:wp.array(dtype=wp.int64)):
    i = wp.tid()
    output[i] = wp.int64(-1)
    if task[i]:
        p = points[i]; minimum = wp.float64(1.0e308)
        query = wp.bvh_query_aabb(bvh,lower[i],upper[i])
        t = int(0)
        while wp.bvh_query_next(query,t):
            if triangle_shape[t] == external[i]:
                d = _distance(p,vertices[3*t],vertices[3*t+1],vertices[3*t+2])
                minimum = wp.min(minimum,d)
        query = wp.bvh_query_aabb(bvh,lower[i],upper[i])
        best = int(-1); score = wp.float64(-1.0e308)
        while wp.bvh_query_next(query,t):
            if triangle_shape[t] == external[i]:
                d = _distance(p,vertices[3*t],vertices[3*t+1],vertices[3*t+2])
                if wp.abs(d-minimum) <= wp.float64(1.0e-12)+wp.float64(1.0e-9)*wp.abs(minimum):
                    face = int(owner[t]); alignment = wp.dot(face_normal[face],normals[i])
                    take = best < 0 or alignment > score
                    if best >= 0 and alignment == score:
                        take = surface_ids[face] < surface_ids[best]
                        if surface_ids[face] == surface_ids[best]:
                            take = face < best
                    if take:
                        best = face; score = alignment
        output[i] = wp.int64(best)


class DeviceSupportContactSelector:
    """Small status transfers only; contact rows and classifications stay on GPU."""
    def __init__(self, binding, body_labels, device, *, body_names_by_part=None):
        from .newton_support import SupportContactSelector
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Live device contact selection requires CUDA')
        cpu = SupportContactSelector(binding,body_labels,body_names_by_part=body_names_by_part)
        self.body_world = torch.as_tensor(cpu.body_world,device=self.device)
        self.body_part = torch.as_tensor(cpu.body_part,device=self.device)
        triangles, owners, triangle_shapes, normals, surfaces, upward = [], [], [], [], [], []
        first = np.full(max(cpu.face_ids)+1,-1,np.int64)
        for shape, faces in binding['shape_surface'].items():
            shape = int(shape); first[shape] = len(triangles)
            for local, face in enumerate(faces):
                owner = len(normals)
                normals.append(face['normal_w']); surfaces.append(face['surface'])
                upward.append(cpu.face_upward[shape][local])
                triangles.extend(face['triangles_w'])
                owners.extend([owner]*len(face['triangles_w']))
                triangle_shapes.extend([shape]*len(face['triangles_w']))
        triangles = np.asarray(triangles,np.float64)
        from .newton_contacts import triangle_distance_squared
        triangle_distance_squared(np.zeros(3),triangles)  # Reject invalid/degenerate binding once.
        self.vertices = wp.array(triangles.reshape(-1,3),dtype=wp.vec3d,device=str(self.device))
        self.triangle_shapes = wp.array(triangle_shapes,dtype=wp.int64,device=str(self.device))
        self.owners = wp.array(owners,dtype=wp.int64,device=str(self.device))
        self.face_normals = wp.array(normals,dtype=wp.vec3d,device=str(self.device))
        self.surfaces = torch.tensor(surfaces,device=self.device,dtype=torch.long)
        self.upward = torch.tensor(upward,device=self.device,dtype=torch.bool)
        self.first = torch.tensor(first,device=self.device)
        self.mesh_points = wp.array(triangles.astype(np.float32).reshape(-1,3),dtype=wp.vec3,device=str(self.device))
        self.mesh_indices = wp.array(np.arange(3*len(triangles),dtype=np.int32),dtype=wp.int32,device=str(self.device))
        self.mesh = wp.Mesh(points=self.mesh_points,indices=self.mesh_indices)
        self.lower = wp.array(np.nextafter(triangles.min(1).astype(np.float32),np.float32(-np.inf)),
                              dtype=wp.vec3,device=str(self.device))
        self.upper = wp.array(np.nextafter(triangles.max(1).astype(np.float32),np.float32(np.inf)),
                              dtype=wp.vec3,device=str(self.device))
        self.bvh = wp.Bvh(self.lower,self.upper)
        wp.stream_from_torch(torch.cuda.current_stream(self.device)).wait_stream(
            wp.get_stream(str(self.device)))

    def __call__(self, snapshot):
        dist, margin, kind = (snapshot[k] for k in ('dist','includemargin','type'))
        n = len(dist); row = torch.arange(n,device=self.device)
        count = torch.as_tensor(snapshot['count'],device=self.device)
        valid = row < count
        address, needed = snapshot['efc_address'], snapshot['constraint_rows']
        required = torch.arange(address.shape[1],device=self.device)[None] < needed[:,None]
        active = valid & ((kind & 1) != 0) & (dist < margin)
        allocated = active & ((address >= 0) | ~required).all(-1)
        bodies = torch.stack((snapshot['body0'],snapshot['body1']),-1).long()
        body_valid = (bodies >= 0) & (bodies < len(self.body_world))
        safe = bodies.clamp(0,len(self.body_world)-1)
        worlds = torch.where(body_valid,self.body_world[safe],-1)
        parts = torch.where(body_valid,self.body_part[safe],-1)
        robot = worlds >= 0; side = robot[:,1].long()
        own_world, part = worlds[row,side], parts[row,side]
        task = active & (robot.sum(-1) == 1) & (part >= 0) & (own_world == snapshot['worldid'])
        external = torch.where(side == 0,snapshot['shape1'],snapshot['shape0']).long()
        shape_valid = (external >= 0) & (external < len(self.first))
        safe_shape = external.clamp(0,len(self.first)-1)
        bound = shape_valid & (self.first[safe_shape] >= 0)
        point = snapshot['position_w'].double().contiguous()
        normal = (snapshot['frame_w'][:,0].double()*torch.where(side == 1,1.,-1.)[:,None]).contiguous()
        finite = torch.isfinite(point).all(-1) & torch.isfinite(normal).all(-1)
        query = (task & bound & finite & allocated).contiguous()
        radius = torch.zeros(n,device=self.device,dtype=torch.float64)
        face = torch.full((n,),-1,device=self.device,dtype=torch.long)
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            if n:
                wp.launch(_bound,n,inputs=[self.mesh.id,wp.from_torch(point,dtype=wp.vec3d),
                    wp.from_torch(query),wp.from_torch(safe_shape),wp.from_torch(self.first),
                    self.triangle_shapes,self.vertices,wp.from_torch(radius)],device=str(self.device))
                low = (point-radius[:,None]).float()
                high = (point+radius[:,None]).float()
                low = torch.nextafter(low,torch.full_like(low,-torch.inf)).contiguous()
                high = torch.nextafter(high,torch.full_like(high,torch.inf)).contiguous()
                wp.launch(_primary,n,inputs=[self.bvh.id,wp.from_torch(point,dtype=wp.vec3d),
                    wp.from_torch(normal,dtype=wp.vec3d),wp.from_torch(query),wp.from_torch(safe_shape),
                    wp.from_torch(low,dtype=wp.vec3),wp.from_torch(high,dtype=wp.vec3),self.vertices,
                    self.triangle_shapes,self.owners,self.face_normals,wp.from_torch(self.surfaces),
                    wp.from_torch(face)],device=str(self.device))
        safe_face = face.clamp_min(0)
        eligible = task & allocated & bound & finite & (face >= 0) & self.upward[safe_face]
        surface = torch.where(task,self.surfaces[safe_face],-1)
        groups = torch.where(task,own_world*6+part,-1)
        errors = dict(capacity=(count < 0) | (count > n),
            fields=(valid & (~torch.isfinite(dist) | ~torch.isfinite(margin) |
                            (needed < 1) | (needed > address.shape[1]))).any(),
            unallocated=(active & ~allocated).any(),
            external=(task & ~bound).any(), geometry=(task & ~finite).any(),
            face=(query & (face < 0)).any())
        if 'active' in snapshot:
            errors['activation'] = ((active != snapshot['active']) | (allocated != snapshot['constraint_allocated'])).any()
        if 'mapping_error' in snapshot:
            errors['mapping'] = snapshot['mapping_error']
        return dict(eligible=eligible,robot_side=torch.where(task,side,-1),groups=groups,
                    surfaces=surface,excluded=task & ~eligible,active=active,
                    constraint_allocated=allocated,errors=errors)

    @staticmethod
    def validate(selected):
        keys = list(selected['errors'])
        # This is the only device-to-host status transfer in classification.
        invalid = torch.stack([selected['errors'][key] for key in keys]).detach().cpu().tolist()
        for key, failed in zip(keys,invalid):
            if failed:
                raise ValueError('Invalid live Newton contact evidence: '+key)
