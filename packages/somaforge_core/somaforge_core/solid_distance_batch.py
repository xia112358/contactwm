"""Batched calls to the same Coal solver, with complete witness validation."""
import numpy as np
from somaforge_core.coal_batch_loader import load_coal_batch


def run_candidates(self,p,r,low,high,candidates):
    sample,pair=np.nonzero(candidates)
    i,j=self.allowed[pair].T
    ids=np.column_stack((sample,i,j)).astype(np.int64)
    raw=load_coal_batch().run([v['geometry'] for v in self.records],np.ascontiguousarray(p),np.ascontiguousarray(r),ids,self.request)
    gap=raw[:,0];normal=raw[:,1:4];p0=raw[:,4:7];p1=raw[:,7:10]
    tolerance=max(self.request.gjk_tolerance,self.request.epa_tolerance)
    world_low=np.minimum(low[sample,i],low[sample,j]);world_high=np.maximum(high[sample,i],high[sample,j])
    exit_bound=np.minimum(high[sample,i]-low[sample,j],high[sample,j]-low[sample,i]).min(-1)
    bound=np.linalg.norm(world_high-world_low,axis=-1)
    bad_distance=~np.isfinite(gap)|(gap>bound+tolerance)|((gap<0)&(-gap>exit_bound+tolerance))
    negative=(gap<0)&~bad_distance
    bad_witness=np.zeros(len(gap),dtype=bool)
    indices=np.flatnonzero(negative)
    if len(indices):
        n,x,y=normal[indices],p0[indices],p1[indices]
        lo,hi=world_low[indices]-tolerance,world_high[indices]+tolerance
        valid=(np.isfinite(n).all(-1)&np.isfinite(x).all(-1)&np.isfinite(y).all(-1)
            &(x>=lo).all(-1)&(x<=hi).all(-1)&(y>=lo).all(-1)&(y<=hi).all(-1)
            &(np.abs(n)<=1.+1.e-8).all(-1))
        usable=np.flatnonzero(valid)
        valid[usable] &= (np.isclose(np.linalg.norm(n[usable],axis=-1),1.,atol=1.e-8,rtol=0)
            &np.isclose(np.sum(n[usable]*(y[usable]-x[usable]),axis=-1),gap[indices[usable]],atol=1.e-8,rtol=1.e-8))
        bad_witness[indices]=~valid
    bad=np.flatnonzero(bad_distance|bad_witness)
    if len(bad):
        k=bad[0];first,second=self.records[i[k]],self.records[j[k]]
        detail=(dict(reason='invalid_distance_or_geometric_exit_bound',gap=float(gap[k]),
                     exit_bound=float(exit_bound[k]),distance_bound=float(bound[k]),tolerance=tolerance)
                if bad_distance[k] else dict(reason='invalid_witness_coordinates_or_normal',gap=float(gap[k]),
                     normal=normal[k].tolist(),point0=p0[k].tolist(),point1=p1[k].tolist()))
        raise self._invalid_result(int(sample[k]),first,second,p[sample[k],[i[k],j[k]]],r[sample[k],[i[k],j[k]]],**detail)
    selected=negative
    a,b=i[selected],j[selected]
    integer=lambda key:np.array([v[key] for v in self.records],dtype=np.int64)
    shape,component,link=(integer(k) for k in ('shape','component','link'))
    return dict(sample=sample[selected],shape0=shape[a],shape1=shape[b],component0=component[a],component1=component[b],
        body_link0=link[a],body_link1=link[b],full_kind=(np.minimum(link[a],link[b])>=0).astype(np.int64),
        dist=gap[selected],normal_w=normal[selected],point0_w=p0[selected],point1_w=p1[selected])

