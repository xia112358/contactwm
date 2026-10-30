import itertools
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull
from newton._src.geometry.support_function import GenericShapeData,pack_mesh_ptr
from somaforge_core.newton_convex_sat import solve_overlap

@wp.kernel
def cases(mesh:wp.uint64,result:wp.array2d[float]):
    i=wp.tid();a=GenericShapeData();a.shape_type=10;a.scale=wp.vec3(1.0);a.auxiliary=pack_mesh_ptr(mesh)
    b=a;offset=wp.vec3(1.5,0.0,0.0)
    if i==1:offset=wp.vec3(3.0,0.0,0.0)
    if i==2:b.scale=wp.vec3(0.2);offset=wp.vec3(0.0)
    ok,pa,pb,n,d=solve_overlap(a,b,wp.quat_identity(),offset)
    result[i,0]=float(ok);result[i,1]=d;result[i,2]=wp.length(n)

def test_overlap_separation_and_containment():
    points=np.array(list(itertools.product([-1.,1.],repeat=3)),dtype=np.float32)
    hull=ConvexHull(points)
    mesh=wp.Mesh(wp.array(points,dtype=wp.vec3,device='cpu'),wp.array(hull.simplices.astype(np.int32).reshape(-1),dtype=int,device='cpu'))
    result=wp.zeros((3,3),dtype=float,device='cpu')
    wp.launch(cases,3,inputs=[mesh.id,result],device='cpu')
    values=result.numpy()
    np.testing.assert_allclose(values[0],[1,-.5,1],atol=1e-6)
    assert values[1,0]==0 and values[1,1]>0
    np.testing.assert_allclose(values[2],[1,-1.2,1],atol=1e-6)
