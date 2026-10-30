"""One rolling spherical surface image, reprojected and fused at each scan.

Stores one exact observed world point + timestamp per panorama pixel. No voxel
history or complete geometry is retained. Occluded second-layer surfaces are
intentionally discarded, unlike the original multi-surface voxel map.
"""
import torch
import warp as wp

H,W=64,128
EMPTY=9223372036854775807

@wp.kernel
def project_candidates(old_points:wp.array2d(dtype=wp.vec3),old_stamps:wp.array2d(dtype=float),
                       scan_points:wp.array2d(dtype=wp.vec3),scan_valid:wp.array2d(dtype=wp.bool),
                       origins:wp.array(dtype=wp.vec3),yaw:wp.array(dtype=float),timestamp:float,
                       keys:wp.array2d(dtype=wp.int64)):
    e,i=wp.tid();nr=scan_points.shape[1]
    point=wp.vec3(0.0);stamp=float(-1.e20);valid=int(0)
    # Fresh scan wins equal-distance ties; selected coordinates remain exact.
    if i<nr:
        point=scan_points[e,i];stamp=timestamp
        if scan_valid[e,i]:valid=1
    else:
        point=old_points[e,i-nr];stamp=old_stamps[e,i-nr]
        if timestamp-stamp<=15.0:valid=1
    if valid==1:
        v=point-origins[e];distance=wp.length(v)
        if distance>=.12 and distance<=5.0:
            az=wp.atan2(v[1],v[0])-yaw[e]
            az=az-2.0*wp.pi*wp.floor((az+wp.pi)/(2.0*wp.pi))
            el=wp.asin(wp.clamp(v[2]/distance,-1.0,1.0))
            x=wp.clamp(int(wp.floor((az+wp.pi)*128.0/(2.0*wp.pi))),0,127)
            y=wp.clamp(int(wp.floor((wp.pi/2.0-el)*64.0/wp.pi)),0,63)
            units=wp.int64(wp.floor(distance*1000000.0+.5))
            key=(units<<wp.int64(32))|wp.int64(i)
            wp.atomic_min(keys,e,y*128+x,key)

@wp.kernel
def resolve_candidates(old_points:wp.array2d(dtype=wp.vec3),old_stamps:wp.array2d(dtype=float),
                       scan_points:wp.array2d(dtype=wp.vec3),origins:wp.array(dtype=wp.vec3),timestamp:float,
                       keys:wp.array2d(dtype=wp.int64),new_points:wp.array2d(dtype=wp.vec3),
                       new_stamps:wp.array2d(dtype=float),depth:wp.array2d(dtype=float)):
    e,r=wp.tid();key=keys[e,r];nr=scan_points.shape[1]
    new_points[e,r]=wp.vec3(0.0);new_stamps[e,r]=-1.e20;depth[e,r]=0.0
    if key!=wp.int64(9223372036854775807):
        index=int(key&wp.int64(4294967295));point=wp.vec3(0.0);stamp=float(0.0)
        if index<nr:point=scan_points[e,index];stamp=timestamp
        else:point=old_points[e,index-nr];stamp=old_stamps[e,index-nr]
        new_points[e,r]=point;new_stamps[e,r]=stamp
        distance=wp.length(point-origins[e])
        if distance<=2.5:depth[e,r]=distance

class RollingPanorama:
    def __init__(self,n,device):
        self.device=torch.device(device);self.n=n
        self.points=torch.zeros(n,H*W,3,device=device)
        self.stamps=torch.full((n,H*W),-1.e20,device=device)
        self.next_points=torch.empty_like(self.points);self.next_stamps=torch.empty_like(self.stamps)
        self.keys=torch.empty(n,H*W,dtype=torch.int64,device=device)
        self.clock=torch.full((n,),-torch.inf,device=device)
    @property
    def gpu_bytes(self):
        return sum(x.numel()*x.element_size() for x in [self.points,self.stamps,self.next_points,self.next_stamps,self.keys,self.clock])
    def reset(self,ids):
        self.stamps[ids]=-1.e20;self.clock[ids]=-torch.inf
    @torch.no_grad()
    def update(self,points,valid,origins,yaw,timestamp):
        if (self.clock>timestamp).any():raise ValueError('Rolling history timestamp went backwards')
        self.keys.fill_(EMPTY)
        depth=torch.empty(self.n,H*W,device=self.device)
        common=[wp.from_torch(self.points,dtype=wp.vec3),wp.from_torch(self.stamps),wp.from_torch(points.contiguous(),dtype=wp.vec3)]
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            wp.launch(project_candidates,dim=(self.n,points.shape[1]+H*W),inputs=[*common,wp.from_torch(valid.contiguous()),
                      wp.from_torch(origins.contiguous(),dtype=wp.vec3),wp.from_torch(yaw.contiguous()),float(timestamp),wp.from_torch(self.keys)],device=str(self.device))
            wp.launch(resolve_candidates,dim=depth.shape,inputs=[*common,wp.from_torch(origins.contiguous(),dtype=wp.vec3),float(timestamp),
                      wp.from_torch(self.keys),wp.from_torch(self.next_points,dtype=wp.vec3),wp.from_torch(self.next_stamps),wp.from_torch(depth)],device=str(self.device))
        self.points,self.next_points=self.next_points,self.points
        self.stamps,self.next_stamps=self.next_stamps,self.stamps
        self.clock.fill_(timestamp)
        depth=depth.reshape(self.n,H,W)
        return dict(depth=depth,valid=depth>0,age=(timestamp-self.stamps).reshape(self.n,H,W))
