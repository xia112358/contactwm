"""Event extraction and independent evaluation of actual solver observations."""
import numpy as np


def contact_transitions(active, known, minimum_hold=5, minimum_history=5):
    """Onsets with known preceding absence and a known sustained active interval.

    Both dynamic and fixed-q arrays can produce candidates. This function alone
    makes no assertion that reference candidates were dynamically executed.
    """
    active=np.asarray(active,bool);known=np.asarray(known,bool)
    if active.ndim!=2 or active.shape[1]!=6 or known.shape!=active.shape:
        raise ValueError('Expected time x six-part arrays')
    if minimum_hold<1 or minimum_history<1:raise ValueError('Positive history/hold required')
    result=[]
    for t in range(minimum_history,len(active)-minimum_hold+1):
        established=(~active[t-1])&active[t]&known[t-1]&known[t]
        established &= (active[t:t+minimum_hold]&known[t:t+minimum_hold]).all(0)
        if not established.any():continue
        # Supports were known active through the approach and hold interval.
        supports=(active[t-minimum_history:t+minimum_hold]&known[t-minimum_history:t+minimum_hold]).all(0)
        if not supports.any():continue
        result.append(dict(arrival=t,establish=established,maintain=supports,
                           starts=list(range(t-minimum_history,t))))
    return result


def native_success(active,known,force_w,point_w,point_velocity_w,failed,goal,*,arrival,hold,
                   position_tolerance=.05,slip_tolerance=.05,position_required=None):
    """Actual simulator success; model outputs are not accepted as evidence.

    Inputs must be native allocated-constraint evidence under the task's shared
    face-selection rule, not a proximity detector. This function checks task
    position/load/slip in addition to activation; it does not redefine activation.
    """
    active=np.asarray(active,bool);known=np.asarray(known,bool)
    if arrival<0 or hold<1 or arrival+hold>len(active):raise ValueError('Incomplete verification window')
    if active.shape!=known.shape or active.shape[1:]!=(6,):raise ValueError('Six contacts required')
    for value in (force_w,point_w,point_velocity_w):
        if np.asarray(value).shape!=(*active.shape,3):raise ValueError('Invalid native vector shape')
    establish=np.asarray(goal['establish'],bool);maintain=np.asarray(goal['maintain'],bool)
    required=np.broadcast_to(maintain,active.shape).copy()
    required[arrival:arrival+hold] |= establish
    required[arrival+hold:]=False
    placement_required=required if position_required is None else required&np.asarray(position_required,bool)[None]
    if placement_required.shape!=required.shape:raise ValueError('Invalid placement requirement mask')
    if position_required is not None and not np.asarray(position_required,bool)[establish].all():
        raise ValueError('New target contacts must always require placement')
    normals=np.asarray(goal['normal_w']);force=np.sum(force_w*normals,axis=-1)
    target=np.where(maintain[:,None],goal['support_position_w'],goal['position_w'])
    distance=np.linalg.norm(point_w-target,axis=-1)
    normal_velocity=np.sum(point_velocity_w*normals,axis=-1)
    slip=np.linalg.norm(point_velocity_w-normal_velocity[...,None]*normals,axis=-1)
    finite=np.isfinite(distance)&np.isfinite(slip)&np.isfinite(force)
    checks=dict(known=bool(known[required].all()),activation=bool(active[required].all()),
                force=bool((force>=goal['min_normal_force'])[required].all()),
                position=bool((distance<=position_tolerance)[placement_required].all()),
                slip=bool((slip<=slip_tolerance)[required].all()),finite=bool(finite[required].all()),
                no_failure=not bool(np.asarray(failed)[:arrival+hold].any()),
                nonempty=bool(establish.any() and maintain.any()))
    return dict(success=all(checks.values()),checks=checks,
                max_position_error=float(distance[placement_required].max()) if placement_required.any() else None,
                max_slip=float(slip[required].max()) if required.any() else None)
