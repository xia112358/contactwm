"""Read-only replay of upstream mesh BVH candidate and front-face predicates."""
import numpy as np
import warp as wp
from newton._src.geometry.collision_core import _compute_mesh_vs_convex_query_aabb, _mesh_triangle_is_front_facing_local

@wp.kernel
def inspect_midphase(shape:int, transforms:wp.array[wp.transform], types:wp.array[int], data:wp.array[wp.vec4], pointers:wp.array[wp.uint64], gaps:wp.array[float], bounds:wp.array[wp.vec3], hits:wp.array[int], fronts:wp.array[int], signed:wp.array[float]):
    threshold=gaps[0]+gaps[shape]+data[0][3]+data[shape][3]
    lo,hi,center=_compute_mesh_vs_convex_query_aabb(0,shape,transforms[0],transforms[shape],types,data,pointers,threshold)
    bounds[0]=lo;bounds[1]=hi;bounds[2]=center
    query=wp.mesh_query_aabb(pointers[0],lo,hi)
    tri=int(0)
    mesh=wp.mesh_get(pointers[0])
    while wp.mesh_query_aabb_next(query,tri):
        hits[tri]=1
        fronts[tri]=int(_mesh_triangle_is_front_facing_local(pointers[0],center,tri))
        a=mesh.points[mesh.indices[3*tri]];b=mesh.points[mesh.indices[3*tri+1]];c=mesh.points[mesh.indices[3*tri+2]]
        normal=wp.normalize(wp.cross(b-a,c-a))
        signed[tri]=wp.dot(normal,center-a)

def inspect(query,shape):
    n=len(query.model.shape_source[0].indices)//3
    bounds=wp.zeros(3,dtype=wp.vec3,device=query.model.device)
    hits=wp.zeros(n,dtype=int,device=query.model.device);fronts=wp.zeros_like(hits)
    signed=wp.zeros(n,dtype=float,device=query.model.device)
    wp.launch(inspect_midphase,1,inputs=[shape,query.pipeline.geom_transform,query.model.shape_type,query.pipeline.geom_data,query.model.shape_source_ptr,query.model.shape_gap,bounds,hits,fronts,signed],device=query.model.device)
    h=hits.numpy().astype(bool);f=fronts.numpy().astype(bool);d=signed.numpy()
    return dict(shape=shape,bounds=bounds.numpy().tolist(),bvh_triangles=np.flatnonzero(h).tolist(),
        front_facing_triangles=np.flatnonzero(h&f).tolist(),culled_triangles=np.flatnonzero(h&~f).tolist(),
        center_signed_distance_m={str(i):float(d[i]) for i in np.flatnonzero(h)},
        contract='Passive replay of pinned upstream predicates. Does not change actual collision pipeline.')
