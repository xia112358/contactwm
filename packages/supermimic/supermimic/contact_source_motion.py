"""Offline posture phases and auditable edge exclusions; never rewrite source data."""
import torch
from .rotations import quat_to_mat
from .contact_temporal import _runs,TEMPORAL_CONTRACT

SOURCE_MOTION_CONTRACT=dict(version='source_motion_edges_v2',dt=.02,quiet_frames=15,
    quiet_root_m=.01,quiet_root_angle_rad=.04,quiet_joint_rad=.06,quiet_part_m=.015,
    posture_min_frames=15,rise_height_m=.12,posture_root_m=.03,
    initial_drop_speed_m_s=.35,initial_settle_search_frames=25,
    fall_upright_cos=.5,fall_drop_m=.12,fall_speed_m_s=.5,
    exclusions='edge-only static waits, measured initialization settling, known physical/numeric failures or unknown native evidence; partial normal transfers retained',
    posture='use associated support placements for stance; measured contact/support gaps remain separate',
    middle='flag interior anomalies; never join frames across an excluded interval')


def source_motion_analysis(data,temporal):
    T=len(data['s1']);root=data['s1'][:,:3];q=data['s1'][:,13:42]
    R=quat_to_mat(data['s1'][:,3:7]);p=data['native_part_position'];dt=SOURCE_MOTION_CONTRACT['dt']
    vz=torch.zeros(T,device=root.device);vz[1:]=(root[1:,2]-root[:-1,2])/dt
    lower=temporal['placement_support'][:,[0,1,4,5]].any(-1)
    double=(temporal['placement_support'][:,0]|temporal['placement_support'][:,4])&(temporal['placement_support'][:,1]|temporal['placement_support'][:,5])
    known=data['known'].all(-1);quiet=torch.zeros(T,dtype=torch.bool,device=root.device)
    width=SOURCE_MOTION_CONTRACT['quiet_frames']
    for end in range(width,T+1):
        start=end-width
        if not bool(known[start:end].all() and lower[start:end].all()):continue
        rel=R[start:end]@R[start].T
        angle=torch.acos(((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)).max()
        if ((root[start:end]-root[start]).norm(dim=-1).max()<=SOURCE_MOTION_CONTRACT['quiet_root_m']
            and angle<=SOURCE_MOTION_CONTRACT['quiet_root_angle_rad']
            and (q[start:end]-q[start]).abs().max()<=SOURCE_MOTION_CONTRACT['quiet_joint_rad']
            and (p[start:end]-p[start]).norm(dim=-1).max()<=SOURCE_MOTION_CONTRACT['quiet_part_m']):quiet[start:end]=True
    exclusions=[];lo=0;hi=T
    while lo<hi and not bool(known[lo]):lo+=1
    if lo:exclusions.append(dict(start=0,stop=lo,reason='unknown_native_prefix'))
    while hi>lo and not bool(known[hi-1]):hi-=1
    if hi<T:exclusions.append(dict(start=hi,stop=T,reason='unknown_native_suffix'))
    # Initialization drop needs rigid, almost vertical falling evidence, not
    # merely an airborne limb or an unfinished first step.
    n=min(lo+3,hi)
    raw_lower=(data['active'][:,[0,1,4,5]]&(data['contact_load'][:,[0,1,4,5]]>30)).any(-1)
    if n-lo>=3 and not bool(raw_lower[lo:n].any()) and bool((vz[lo+1:n]<-SOURCE_MOTION_CONTRACT['initial_drop_speed_m_s']).all()) and float((q[lo:n]-q[lo]).abs().max())<.08 and float((root[lo:n,:2]-root[lo,:2]).norm(dim=-1).max())<.02:
        for t in range(n,min(hi-4,lo+SOURCE_MOTION_CONTRACT['initial_settle_search_frames'])):
            if bool(lower[t:t+5].all() and (vz[t:t+5].abs()<=.25).all() and (R[t:t+5,2,2]>.7).all()):
                exclusions.append(dict(start=lo,stop=t,reason='initialization_drop',evidence=dict(initial_vertical_speed_m_s=vz[lo+1:n].tolist(),settled_start=t)));lo=t;break
    # A normal descending step remains upright; a truncated flight alone is
    # not sufficient evidence of a terminal fall.
    fall=torch.zeros(T,dtype=torch.bool,device=root.device)
    for t in range(max(1,lo),hi):
        before=max(lo,t-10)
        fall[t]=R[t,2,2]<SOURCE_MOTION_CONTRACT['fall_upright_cos'] and root[before,2]-root[t,2]>SOURCE_MOTION_CONTRACT['fall_drop_m'] and vz[t]<-SOURCE_MOTION_CONTRACT['fall_speed_m_s']
    interior=[]
    for a,b in _runs(fall):
        if b-a<3:continue
        if bool((R[b:hi,2,2]<.7).all()):
            onset=a
            while onset>max(lo,a-10) and float(vz[onset-1])<-.3:onset-=1
            exclusions.append(dict(start=onset,stop=hi,reason='terminal_fall',evidence=dict(confirmed=a,upright_cos=float(R[a,2,2]),vertical_speed_m_s=float(vz[a]))));hi=onset;break
        interior.append(dict(start=a,stop=b,reason='interior_fall_evidence'))
    start=lo
    while lo<hi and bool(quiet[lo]):lo+=1
    if lo>start:exclusions.append(dict(start=start,stop=lo,reason='static_prefix'))
    end=hi
    while hi>lo and bool(quiet[hi-1]):hi-=1
    if hi<end:exclusions.append(dict(start=hi,stop=end,reason='static_suffix'))
    posture=[]
    for a,b in _runs(double&known):
        a=max(a+1,lo);b=min(b,hi)
        if b-a<SOURCE_MOTION_CONTRACT['posture_min_frames']:continue
        dz=float(root[b-1,2]-root[a,2]);travel=float((root[a:b]-root[a]).norm(dim=-1).max())
        if dz>=SOURCE_MOTION_CONTRACT['rise_height_m']:kind='rise'
        elif dz<=-SOURCE_MOTION_CONTRACT['rise_height_m']:kind='lowering'
        elif travel>=SOURCE_MOTION_CONTRACT['posture_root_m']:kind='support_posture_motion'
        elif bool(quiet[a:b].all()):kind='stationary_hold'
        else:continue
        posture.append(dict(start=a,stop=b,kind=kind,root_height_change_m=dz,root_excursion_m=travel))
    return dict(start=lo,stop=hi,exclusions=sorted(exclusions,key=lambda x:x['start']),posture_phases=posture,interior_flags=interior,quiet=quiet)
