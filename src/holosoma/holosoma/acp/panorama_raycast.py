"""Optional compiled CPU DDA reader. Warp is used only for array computation."""
import numpy as np
import warp as wp


@wp.kernel
def _trace(keys: wp.array(dtype=wp.int64), stamps: wp.array(dtype=wp.float64),
           directions: wp.array(dtype=wp.vec3d), origin: wp.vec3d,
           resolution: wp.float64, min_range: wp.float64, max_range: wp.float64,
           timestamp: wp.float64, max_age: wp.float64, steps: int,
           depth: wp.array(dtype=wp.float32), matches: wp.array(dtype=wp.int32)):
    i=wp.tid()
    d=directions[i]
    p=origin+d*min_range
    cell=wp.vec3i(int(wp.floor(p[0]/resolution)),int(wp.floor(p[1]/resolution)),int(wp.floor(p[2]/resolution)))
    step=wp.vec3i(0,0,0)
    crossing=wp.vec3d(wp.float64(1.e30))
    delta=wp.vec3d(wp.float64(1.e30))
    for axis in range(3):
        if wp.abs(d[axis])>wp.float64(1.e-12):
            step[axis]=1
            if d[axis]<wp.float64(0.):step[axis]=-1
            boundary=wp.float64(cell[axis])*resolution
            if step[axis]>0:boundary=boundary+resolution
            crossing[axis]=(boundary-origin[axis])/d[axis]
            delta[axis]=resolution/wp.abs(d[axis])
    travel=min_range
    matches[i]=-1
    for _ in range(steps):
        x=wp.int64(cell[0])+wp.int64(1048576)
        y=wp.int64(cell[1])+wp.int64(1048576)
        z=wp.int64(cell[2])+wp.int64(1048576)
        key=(x<<wp.int64(42))|(y<<wp.int64(21))|z
        low=int(0)
        high=int(keys.shape[0])
        while low<high:
            middle=(low+high)//2
            if keys[middle]<key:low=middle+1
            else:high=middle
        if low<keys.shape[0]:
            if keys[low]==key and timestamp-stamps[low]<=max_age:
                depth[i]=wp.float32(travel)
                matches[i]=low
                break
        next_t=wp.min(crossing[0],wp.min(crossing[1],crossing[2]))
        if next_t>max_range:break
        for axis in range(3):
            if wp.abs(crossing[axis]-next_t)<=wp.float64(1.e-10):
                cell[axis]=cell[axis]+step[axis]
                crossing[axis]=crossing[axis]+delta[axis]
        travel=next_t


def trace(keys,stamps,directions,origin,config,timestamp):
    wp.init()
    n=len(directions)
    depth=wp.zeros(n,dtype=wp.float32,device='cpu')
    matches=wp.empty(n,dtype=wp.int32,device='cpu')
    steps=int(np.ceil((config.max_range-config.min_range)*np.sqrt(3)/config.resolution))+4
    wp.launch(_trace,dim=n,inputs=[wp.array(keys,dtype=wp.int64,device='cpu'),
        wp.array(stamps,dtype=wp.float64,device='cpu'),wp.array(directions,dtype=wp.vec3d,device='cpu'),
        wp.vec3d(*origin),config.resolution,config.min_range,config.max_range,timestamp,config.max_age,steps],
        outputs=[depth,matches],device='cpu')
    return depth.numpy(),matches.numpy()
