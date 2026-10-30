"""Batched causal surface-voxel history and spherical DDA on the current Torch stream.

Only deskewed observed points are inserted. Unknown rays remain unknown. No
height raster or target contacts enter the map. Environments have isolated tables.
"""
import math
import torch
import warp as wp
from holosoma.acp.panorama import PanoramaConfig

@wp.func
def key_of(c:wp.vec3i):
    return ((wp.int64(c[0])+wp.int64(1048576))<<wp.int64(42))|((wp.int64(c[1])+wp.int64(1048576))<<wp.int64(21))|(wp.int64(c[2])+wp.int64(1048576))

@wp.func
def bucket(c:wp.vec3i,capacity:int):
    h=wp.uint32(c[0])*wp.uint32(73856093)^wp.uint32(c[1])*wp.uint32(19349663)^wp.uint32(c[2])*wp.uint32(83492791)
    return int(h&wp.uint32(capacity-1))

@wp.kernel
def insert(points:wp.array2d(dtype=wp.vec3),valid:wp.array2d(dtype=wp.bool),ids:wp.array(dtype=int),
           keys:wp.array2d(dtype=wp.int64),stamps:wp.array2d(dtype=float),counts:wp.array2d(dtype=float),
           timestamp:float,resolution:float,capacity:int,overflow:wp.array(dtype=int)):
    row,i=wp.tid();env=ids[row]
    if valid[row,i]:
        p=points[row,i];c=wp.vec3i(int(wp.floor(p[0]/resolution)),int(wp.floor(p[1]/resolution)),int(wp.floor(p[2]/resolution)))
        key=key_of(c);h=bucket(c,capacity);found=int(0)
        for j in range(128):
            slot=(h+j)&(capacity-1);old=keys[env,slot]
            if old==wp.int64(-1) or old==wp.int64(-2):old=wp.atomic_cas(keys,env,slot,old,key)
            if old==key or old==wp.int64(-1) or old==wp.int64(-2):
                previous=wp.atomic_max(stamps,env,slot,timestamp)
                if previous<timestamp:wp.atomic_add(counts,env,slot,1.0)
                found=1;break
        if found==0:wp.atomic_add(overflow,env,1)

@wp.kernel
def prune(center:wp.array(dtype=wp.vec3),ids:wp.array(dtype=int),keys:wp.array2d(dtype=wp.int64),
          stamps:wp.array2d(dtype=float),counts:wp.array2d(dtype=float),timestamp:float,max_age:float,radius:float,resolution:float):
    row,slot=wp.tid();env=ids[row];key=keys[env,slot]
    if key>=wp.int64(0):
        mask=wp.int64(2097151)
        p=wp.vec3(float((key>>wp.int64(42))&mask)-1048576.0+.5,float((key>>wp.int64(21))&mask)-1048576.0+.5,float(key&mask)-1048576.0+.5)*resolution
        if timestamp-stamps[env,slot]>max_age or wp.length(p-center[row])>radius:
            keys[env,slot]=wp.int64(-2);stamps[env,slot]=-1.e20;counts[env,slot]=0.0

@wp.kernel
def read_sphere(origins:wp.array(dtype=wp.vec3),angles:wp.array(dtype=float),directions:wp.array(dtype=wp.vec3),
                keys:wp.array2d(dtype=wp.int64),stamps:wp.array2d(dtype=float),counts:wp.array2d(dtype=float),
                timestamp:float,max_age:float,resolution:float,min_range:float,max_range:float,capacity:int,max_steps:int,
                depth:wp.array2d(dtype=float),age:wp.array2d(dtype=float),confidence:wp.array2d(dtype=float)):
    env,ray=wp.tid();a=angles[env];local=directions[ray]
    d=wp.vec3(wp.cos(a)*local[0]-wp.sin(a)*local[1],wp.sin(a)*local[0]+wp.cos(a)*local[1],local[2]);origin=origins[env]
    p=origin+d*min_range;c=wp.vec3i(int(wp.floor(p[0]/resolution)),int(wp.floor(p[1]/resolution)),int(wp.floor(p[2]/resolution)))
    step=wp.vec3i(0);crossing=wp.vec3(1.e20);delta=wp.vec3(1.e20);travel=min_range
    depth[env,ray]=0.0;age[env,ray]=1.e20;confidence[env,ray]=0.0
    for axis in range(3):
        if wp.abs(d[axis])>1.e-8:
            step[axis]=1
            if d[axis]<0.0:step[axis]=-1
            edge=float(c[axis])*resolution
            if step[axis]>0:edge+=resolution
            crossing[axis]=(edge-origin[axis])/d[axis];delta[axis]=resolution/wp.abs(d[axis])
    for k in range(max_steps):
        key=key_of(c);h=bucket(c,capacity);hit=int(-1)
        for j in range(128):
            slot=(h+j)&(capacity-1);value=keys[env,slot]
            if value==wp.int64(-1):break
            if value==key:
                if timestamp-stamps[env,slot]<=max_age:hit=slot
                break
        if hit>=0:
            elapsed=timestamp-stamps[env,hit]
            depth[env,ray]=travel;age[env,ray]=elapsed
            confidence[env,ray]=(1.0-wp.exp(-wp.min(counts[env,hit],32.0)))*wp.exp(-elapsed/max_age)
            break
        next_t=wp.min(crossing[0],wp.min(crossing[1],crossing[2]))
        if next_t>max_range:break
        for axis in range(3):
            if wp.abs(crossing[axis]-next_t)<=1.e-6:
                c[axis]+=step[axis];crossing[axis]+=delta[axis]
        travel=next_t

class BatchedSurfacePanorama:
    def __init__(self,num_envs,device,config=None,capacity=262144):
        wp.init()
        self.config=config or PanoramaConfig();self.device=torch.device(device);self.num_envs=num_envs;self.capacity=capacity
        if capacity<128 or capacity&(capacity-1):raise ValueError('Power-of-two map capacity required')
        self.keys=torch.full((num_envs,capacity),-1,dtype=torch.int64,device=device)
        self.stamps=torch.full((num_envs,capacity),-1.e20,device=device)
        self.counts=torch.zeros((num_envs,capacity),device=device)
        self.clock=torch.full((num_envs,),-torch.inf,device=device)
        c=self.config
        az=-torch.pi+(torch.arange(c.width,device=device)+.5)*2*torch.pi/c.width
        el=torch.pi/2-(torch.arange(c.height,device=device)+.5)*torch.pi/c.height
        el,az=torch.meshgrid(el,az,indexing='ij')
        self.directions=torch.stack((el.cos()*az.cos(),el.cos()*az.sin(),el.sin()),-1).reshape(-1,3).contiguous()
    def reset(self,ids):
        self.keys[ids]=-1;self.stamps[ids]=-1.e20;self.counts[ids]=0;self.clock[ids]=-torch.inf
    def _launch(self,kernel,dim,inputs):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            wp.launch(kernel,dim=dim,inputs=inputs,device=str(self.device))
    @torch.no_grad()
    def integrate_world(self,points,valid,centers,timestamp,ids=None):
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        ids=ids.to(torch.int32).contiguous();c=self.config
        if points.shape!=(len(ids),valid.shape[1],3) or centers.shape!=(len(ids),3):raise ValueError('Batched scan shapes differ')
        if (self.clock[ids]>timestamp).any():raise ValueError('Timestamp went backwards without reset')
        valid=valid.bool()&torch.isfinite(points).all(-1)&((points-centers[:,None]).norm(dim=-1)<=c.retain_radius)
        valid&=(self.clock[ids]<timestamp)[:,None]
        if (points[valid].abs()>((1<<20)-2)*c.resolution).any():raise ValueError('Voxel coordinate key range exceeded')
        p=wp.from_torch(points.contiguous(),dtype=wp.vec3);v=wp.from_torch(valid.contiguous())
        center=wp.from_torch(centers.contiguous(),dtype=wp.vec3);indices=wp.from_torch(ids)
        keys=wp.from_torch(self.keys);stamps=wp.from_torch(self.stamps);counts=wp.from_torch(self.counts)
        self._launch(prune,(len(ids),self.capacity),[center,indices,keys,stamps,counts,float(timestamp),c.max_age,c.retain_radius,c.resolution])
        overflow=torch.zeros(self.num_envs,dtype=torch.int32,device=self.device)
        self._launch(insert,valid.shape,[p,v,indices,keys,stamps,counts,float(timestamp),c.resolution,self.capacity,wp.from_torch(overflow)])
        if overflow.any():raise RuntimeError('Surface history hash capacity/probe limit exceeded; observations were not silently dropped')
        if ((self.keys[ids]>=0).sum(-1)>c.max_voxels).any():raise RuntimeError('Configured observed-voxel capacity exceeded')
        self.clock[ids]=timestamp
    @torch.no_grad()
    def render(self,origins,yaw,timestamp):
        c=self.config
        if origins.shape!=(self.num_envs,3) or yaw.shape!=(self.num_envs,):raise ValueError('Invalid panorama pose shapes')
        if (self.clock>timestamp).any():raise ValueError('Rendering before latest integrated scan')
        shape=(self.num_envs,c.height*c.width)
        depth=torch.empty(shape,device=self.device);age=torch.empty_like(depth);confidence=torch.empty_like(depth)
        self._launch(read_sphere,shape,[wp.from_torch(origins.contiguous(),dtype=wp.vec3),wp.from_torch(yaw.contiguous()),
            wp.from_torch(self.directions,dtype=wp.vec3),wp.from_torch(self.keys),wp.from_torch(self.stamps),wp.from_torch(self.counts),
            float(timestamp),c.max_age,c.resolution,c.min_range,c.max_range,self.capacity,
            math.ceil((c.max_range-c.min_range)*math.sqrt(3)/c.resolution)+4,
            wp.from_torch(depth),wp.from_torch(age),wp.from_torch(confidence)])
        shape=(self.num_envs,c.height,c.width)
        return dict(depth=depth.reshape(shape),valid=(depth>0).reshape(shape),age=age.reshape(shape),confidence=confidence.reshape(shape))
