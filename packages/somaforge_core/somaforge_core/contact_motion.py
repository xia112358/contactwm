"""Same-material motion over actual contact regions, not contact truth.

Each displacement follows one body-local point between adjacent poses. The
pivot minimum measures existence only; the full region distribution and actual
loaded-point motion must not be hidden by that minimum.
"""
import torch

CONTACT_MOTION_SCHEMA = 'native_contact_pivot_and_region_motion_v3'
REGION_QUANTILE = .25


def calibrate_pivot_budgets(steps, phases):
    """Robust source step budgets, pooled by part over declared support spans.

    Large source excursions are quality failures, not a lower bound on the
    allowed motion. Use median + 3 MAD; never multiply a contaminated event's
    P95 by its duration or preserve its full observed path as tolerance.
    These remain geometric budgets, not measured force-slip thresholds.
    """
    pools={}
    for a,b,p in phases:
        pools.setdefault(p,set()).update(range(a,b))
    limits={}
    for p,ids in pools.items():
        values=steps[sorted(ids),p]
        values=values[torch.isfinite(values)]
        if not len(values) or bool((values<0).any()):
            raise ValueError('Source calibration has missing/invalid support steps')
        median=torch.quantile(values,.5);mad=torch.quantile((values-median).abs(),.5)
        limits[p]=(median,mad,median+3*mad,len(values))
    rows=[]
    for a,b,p in phases:
        values=steps[a:b,p]
        if not len(values) or not bool(torch.isfinite(values).all()) or bool((values<0).any()):
            raise ValueError('Source calibration has missing/invalid support steps')
        path=values.sum();p95=torch.quantile(values,.95)
        median,mad,limit,count=limits[p]
        rows.append(dict(start=a,end=b,part=p,source_path_m=float(path),step_p95_m=float(p95),
                         budget_m=float(limit*(b-a)),step_median_m=float(median),step_mad_m=float(mad),
                         step_limit_m=float(limit),calibration_steps=count,
                         budget_schema='part_pooled_median_3mad_v1',source_over_budget=bool(path>limit*(b-a))))
    return rows


def same_material_tangent_motion(positions, rotations, frames, bodies, local, normals):
    """Same local point in two poses, projected onto its actual surface plane."""
    delta=positions[frames+1,bodies]-positions[frames,bodies]+torch.einsum(
        'nij,nj->ni',rotations[frames+1,bodies]-rotations[frames,bodies],local)
    return delta-(delta*normals).sum(-1,keepdim=True)*normals


def loaded_motion_statistics(speed, load, groups, size):
    """Measured load-weighted motion, independent of geometric pivot minimum.

    Inputs are already native eligible contacts. Positive load localizes support
    but never creates activation. Empty/unloaded groups remain unknown.
    """
    import numpy as np
    speed,load=np.asarray(speed,float),np.asarray(load,float)
    groups=np.asarray(groups,int)
    if speed.shape!=load.shape or speed.shape!=groups.shape or speed.ndim!=1:
        raise ValueError('Invalid loaded motion groups')
    if not np.isfinite(speed).all() or not np.isfinite(load).all() or (speed<0).any() or (load<0).any():
        raise ValueError('Invalid loaded motion values')
    if ((groups<0)|(groups>=size)).any():raise ValueError('Loaded group outside timeline')
    total=np.bincount(groups,weights=load,minlength=size)
    square=np.bincount(groups,weights=load*speed**2,minlength=size)
    count=np.bincount(groups[load>0],minlength=size)
    rms=np.full(size,np.nan);maximum=np.full(size,np.nan);minimum=np.full(size,np.nan)
    known=total>0;rms[known]=np.sqrt(square[known]/total[known])
    low=np.full(size,np.inf);high=np.full(size,-np.inf)
    positive=load>0
    np.minimum.at(low,groups[positive],speed[positive])
    np.maximum.at(high,groups[positive],speed[positive])
    minimum[known]=low[known];maximum[known]=high[known]
    return dict(normal_force_n=total,rms_tangent_speed_m_s=rms,
                minimum_loaded_speed_m_s=minimum,maximum_loaded_speed_m_s=maximum,
                loaded_constraint_count=count,load_bearing=known,loaded_motion_known=known)


def contact_region_statistics(distance, groups, size, *, width=None, validate=True):
    """Reduce verified contact samples; empty groups stay unknown (NaN)."""
    if distance.ndim != 1 or groups.shape != distance.shape or size < 1:
        raise ValueError('Invalid contact motion groups')
    if validate and (not bool(torch.isfinite(distance).all()) or bool((distance < 0).any())):
        raise ValueError('Invalid same-material motion')
    if validate and bool(((groups < 0) | (groups >= size)).any()):
        raise ValueError('Contact motion group outside timeline')
    counts = torch.bincount(groups, minlength=size)
    width = max(1, int(counts.max())) if width is None else width
    dense = distance.new_full((size, width), float('nan'))
    order = groups.argsort()
    sorted_groups = groups[order]
    offsets = counts.cumsum(0)-counts
    slots = torch.arange(len(groups), device=groups.device)-offsets[sorted_groups]
    dense[sorted_groups, slots] = distance[order]
    quantiles = torch.nanquantile(dense, distance.new_tensor([0., REGION_QUANTILE, .5, .75, 1.]), dim=1)
    return dict(minimum=quantiles[0], pivot=quantiles[0],
                lower_quartile=quantiles[1],median=quantiles[2],
                upper_quartile=quantiles[3], maximum=quantiles[4], count=counts)
