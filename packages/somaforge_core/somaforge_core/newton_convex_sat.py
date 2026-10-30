"""Rare convex-mesh overlap fallback using actual hull faces and edge axes.

Only returns a penetration result for overlapping polyhedra. A positive SAT
gap is NOT a Euclidean separation distance and is not emitted as contact data.
"""
import warp as wp
from newton._src.geometry.support_function import GenericShapeData,unpack_mesh_ptr

@wp.func
def vertex(g:GenericShapeData,index:int,rotation:wp.quat,position:wp.vec3):
    mesh=wp.mesh_get(unpack_mesh_ptr(g.auxiliary))
    return wp.quat_rotate(rotation,wp.cw_mul(mesh.points[index],g.scale))+position

@wp.func
def interval(g:GenericShapeData,n:wp.vec3d,rotation:wp.quat,position:wp.vec3):
    mesh=wp.mesh_get(unpack_mesh_ptr(g.auxiliary))
    lo=wp.float64(1.0e30);hi=wp.float64(-1.0e30)
    low=wp.vec3(0.0);high=wp.vec3(0.0)
    for i in range(mesh.points.shape[0]):
        p=vertex(g,i,rotation,position);v=wp.dot(wp.vec3d(p),n)
        if v<lo:lo=v;low=p
        if v>hi:hi=v;high=p
    return lo,hi,low,high

@wp.func
def axis_result(a:GenericShapeData,b:GenericShapeData,axis:wp.vec3d,qb:wp.quat,pb:wp.vec3):
    n=wp.normalize(axis)
    al,ah,aplo,aphi=interval(a,n,wp.quat_identity(),wp.vec3(0.0))
    bl,bh,bplo,bphi=interval(b,n,qb,pb)
    gap=bl-ah;pa=aphi;p=bplo
    if al-bh>gap:
        gap=al-bh;n=-n;pa=aplo;p=bphi
    return gap,wp.vec3(n),pa,p

@wp.func
def solve_overlap(a:GenericShapeData,b:GenericShapeData,qb:wp.quat,pb:wp.vec3):
    best=wp.float64(-1.0e30);normal=wp.vec3(0.0);point_a=wp.vec3(0.0);point_b=wp.vec3(0.0)
    if a.shape_type!=10 or b.shape_type!=10:
        return False,point_a,point_b,normal,float(best)
    ma=wp.mesh_get(unpack_mesh_ptr(a.auxiliary));mb=wp.mesh_get(unpack_mesh_ptr(b.auxiliary))
    for side in range(2):
        g=a;q=wp.quat_identity();p=wp.vec3(0.0);mesh=ma
        if side==1:g=b;q=qb;p=pb;mesh=mb
        for f in range(mesh.indices.shape[0]//3):
            v0=wp.vec3d(vertex(g,mesh.indices[3*f],q,p))
            v1=wp.vec3d(vertex(g,mesh.indices[3*f+1],q,p))
            v2=wp.vec3d(vertex(g,mesh.indices[3*f+2],q,p))
            axis=wp.cross(v1-v0,v2-v0)
            if wp.length_sq(axis)>wp.float64(1.0e-24):
                gap,n,pa,pb2=axis_result(a,b,axis,qb,pb)
                if gap>best:best=gap;normal=n;point_a=pa;point_b=pb2
                if gap>wp.float64(0.0):return False,point_a,point_b,normal,float(best)
    # All triangle edges include hull edges; extra coplanar diagonals do not
    # invalidate SAT. Cost is confined to the exceptional fallback path.
    for e in range(ma.indices.shape[0]):
        f=e//3;j=e%3
        ea=wp.vec3d(vertex(a,ma.indices[3*f+(j+1)%3],wp.quat_identity(),wp.vec3(0.0)))-wp.vec3d(vertex(a,ma.indices[e],wp.quat_identity(),wp.vec3(0.0)))
        for k in range(mb.indices.shape[0]):
            h=k//3;l=k%3
            eb=wp.vec3d(vertex(b,mb.indices[3*h+(l+1)%3],qb,pb))-wp.vec3d(vertex(b,mb.indices[k],qb,pb))
            axis=wp.cross(ea,eb)
            if wp.length_sq(axis)>wp.float64(1.0e-24):
                gap,n,pa,pb2=axis_result(a,b,axis,qb,pb)
                if gap>best:best=gap;normal=n;point_a=pa;point_b=pb2
                if gap>wp.float64(0.0):return False,point_a,point_b,normal,float(best)
    valid=best>wp.float64(-1.0e29) and best<=wp.float64(0.0)
    return valid,point_a,point_b,normal,float(best)
