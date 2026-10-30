"""Near-field panoramic depth from a bounded, historical 3-D surface map.

Single-robot CPU reference implementation. Inputs are deskewed metric points
and scan-time poses, never a height map. Occupied voxels represent observed
surfaces, not free space or contact truth. DDA reads the nearest occupied voxel
along each spherical ray; missed rays remain unknown. Static-scene assumption:
old surfaces expire but this implementation does not perform free-space carving.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PanoramaConfig:
    resolution: float = .03
    retain_radius: float = 5.0
    max_age: float = 15.0
    max_voxels: int = 200_000
    height: int = 64
    width: int = 128
    min_range: float = .12
    max_range: float = 2.5

    def __post_init__(self):
        values=(self.resolution,self.retain_radius,self.max_age,self.min_range,self.max_range)
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('Panorama lengths and ages must be positive and finite')
        if self.min_range>=self.max_range or self.retain_radius<self.max_range:
            raise ValueError('Require min_range < max_range <= retain_radius')
        if self.width<4 or self.height<2 or self.max_voxels<1:
            raise ValueError('Invalid panorama dimensions or map capacity')


def rotation_xyzw(q):
    q=np.asarray(q,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('Invalid xyzw pose quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _keys(cells):
    # Signed global voxel coordinates, collision-free within this explicit range.
    c=np.asarray(cells,dtype=np.int64)+(1<<20)
    if np.any(c<0) or np.any(c>=(1<<21)):
        raise ValueError('World position exceeds voxel key coordinate range')
    return (c[:,0]<<42)|(c[:,1]<<21)|c[:,2]


class SurfacePanorama:
    def __init__(self, config=None, *, backend="numpy"):
        if backend not in ("numpy", "warp"):
            raise ValueError("Panorama backend must be numpy or warp")
        self.backend=backend
        self.config=config or PanoramaConfig()
        c=self.config
        azimuth=-np.pi+(np.arange(c.width)+.5)*2*np.pi/c.width
        elevation=np.pi/2-(np.arange(c.height)+.5)*np.pi/c.height
        az,el=np.meshgrid(azimuth,elevation)
        self.directions=np.stack((np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)),-1)
        self.reset()

    def reset(self):
        self.keys=np.empty(0,np.int64)
        self.cells=np.empty((0,3),np.int64)
        self.stamp=np.empty(0,np.float64)
        self.count=np.empty(0,np.float32)
        self.clock=-np.inf

    def _pose(self,position,quaternion,timestamp):
        position=np.asarray(position,dtype=np.float64)
        if position.shape!=(3,) or not np.isfinite(position).all() or not math.isfinite(timestamp):
            raise ValueError('Invalid pose position or timestamp')
        if timestamp<self.clock:
            raise ValueError('Timestamp moved backwards; reset explicitly')
        return position,rotation_xyzw(quaternion)

    def _prune(self,center,timestamp):
        c=self.config
        keep=(timestamp-self.stamp<=c.max_age)&(np.linalg.norm((self.cells+.5)*c.resolution-center,axis=1)<=c.retain_radius)
        self.keys,self.cells,self.stamp,self.count=(x[keep] for x in (self.keys,self.cells,self.stamp,self.count))
        if len(self.keys)>c.max_voxels:
            ids=np.sort(np.argsort(self.stamp,kind='stable')[-c.max_voxels:])
            self.keys,self.cells,self.stamp,self.count=(x[ids] for x in (self.keys,self.cells,self.stamp,self.count))

    def integrate(self,points_sensor,valid,position_w,quaternion_xyzw,timestamp):
        position,rotation=self._pose(position_w,quaternion_xyzw,timestamp)
        points=np.asarray(points_sensor,dtype=np.float64)
        mask=np.asarray(valid,dtype=bool)
        if points.ndim!=2 or points.shape[1]!=3 or mask.shape!=(len(points),):
            raise ValueError('Expected points [N,3] and validity [N]')
        if timestamp==self.clock:
            return  # Re-reading one scan must not increase its evidence count.
        c=self.config
        points=points[mask&np.isfinite(points).all(1)]
        points=points[np.linalg.norm(points,axis=1)<=c.retain_radius]
        cells=np.floor((points@rotation.T+position)/c.resolution).astype(np.int64)
        keys,idx=np.unique(_keys(cells),return_index=True)
        cells=cells[idx]
        joined,inverse=np.unique(np.concatenate((self.keys,keys)),return_inverse=True)
        old,new=inverse[:len(self.keys)],inverse[len(self.keys):]
        merged_cells=np.empty((len(joined),3),np.int64)
        merged_cells[old]=self.cells;merged_cells[new]=cells
        stamps=np.full(len(joined),-np.inf);stamps[old]=self.stamp;stamps[new]=timestamp
        counts=np.zeros(len(joined),np.float32);counts[old]=self.count
        counts[new]=np.minimum(counts[new]+1,32)
        self.keys,self.cells,self.stamp,self.count=joined,merged_cells,stamps,counts
        self.clock=timestamp
        self._prune(position,timestamp)

    def render(self,position_w,quaternion_xyzw,timestamp):
        """Full-sphere nearest-surface depth, not vertical height sampling.

        The caller chooses a gravity-aligned robot-yaw reading pose. Depth is
        distance to the occupied voxel entry, accurate to the voxel resolution.
        No return means unknown; it does not certify free space out to max_range.
        Confidence is a heuristic freshness/evidence score, not a probability.
        """
        origin,rotation=self._pose(position_w,quaternion_xyzw,timestamp)
        c=self.config
        direction=self.directions.reshape(-1,3)@rotation.T
        n=len(direction)
        depth=np.zeros(n,np.float32);valid=np.zeros(n,bool)
        age=np.full(n,np.inf,np.float32);confidence=np.zeros(n,np.float32)
        active=np.arange(n)
        start=origin+direction*c.min_range
        cells=np.floor(start/c.resolution).astype(np.int64)
        step=np.sign(direction).astype(np.int64)
        safe=np.abs(direction)>1e-12
        delta=np.full_like(direction,np.inf)
        np.divide(c.resolution,np.abs(direction),out=delta,where=safe)
        boundary=(cells+(step>0))*c.resolution
        next_t=np.full_like(direction,np.inf)
        np.divide(boundary-origin,direction,out=next_t,where=safe)
        travel=np.full(n,c.min_range)
        limit=int(math.ceil((c.max_range-c.min_range)*math.sqrt(3)/c.resolution))+4
        if self.backend=="warp":
            from holosoma.acp.panorama_raycast import trace
            # Validate the whole near-field coordinate domain before packing keys.
            _keys(np.floor((origin[None]+np.array([[-c.max_range]*3,[c.max_range]*3]))/c.resolution).astype(np.int64))
            depth,matches=trace(self.keys,self.stamp,direction,origin,c,timestamp)
            valid=matches>=0
            age[valid]=timestamp-self.stamp[matches[valid]]
            confidence[valid]=(1-np.exp(-self.count[matches[valid]]))*np.exp(-age[valid]/c.max_age)
            active=np.empty(0,dtype=np.int64)
        for _ in range(limit):
            if not len(active) or not len(self.keys):break
            keys=_keys(cells[active]);ids=np.searchsorted(self.keys,keys)
            ids_safe=np.minimum(ids,len(self.keys)-1)
            hit=(ids<len(self.keys))&(self.keys[ids_safe]==keys)&(timestamp-self.stamp[ids_safe]<=c.max_age)
            rays=active[hit];matches=ids_safe[hit]
            depth[rays]=travel[rays];valid[rays]=True
            age[rays]=timestamp-self.stamp[matches]
            confidence[rays]=(1-np.exp(-self.count[matches]))*np.exp(-age[rays]/c.max_age)
            active=active[~hit]
            if not len(active):break
            crossing=next_t[active].min(1)
            active=active[crossing<=c.max_range]
            if not len(active):break
            crossing=next_t[active].min(1)
            axes=np.isclose(next_t[active],crossing[:,None],rtol=0,atol=1e-10)
            cells[active]+=step[active]*axes
            next_t[active]=np.where(axes,next_t[active]+delta[active],next_t[active])
            travel[active]=crossing
        shape=(c.height,c.width)
        points=origin+direction*depth[:,None]
        points[~valid]=np.nan
        return dict(depth=depth.reshape(shape),valid=valid.reshape(shape),age=age.reshape(shape),
                    confidence=confidence.reshape(shape),points_w=points.reshape(*shape,3))

    def surface_points(self,timestamp):
        if not math.isfinite(timestamp) or timestamp<self.clock:raise ValueError('Invalid query timestamp')
        keep=timestamp-self.stamp<=self.config.max_age
        return ((self.cells[keep]+.5)*self.config.resolution).astype(np.float32)
