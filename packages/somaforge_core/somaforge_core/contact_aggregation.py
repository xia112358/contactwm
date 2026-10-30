"""Offline multi-demonstration contact consensus, NOT realized contact labels.

Inputs must already be verified Newton observations. Every demonstration has
one vote per (part, robot shape, actual mapped face), including absent votes.
The exported consensus is a denoised interaction hypothesis. It must never be
passed off as solver activation for a newly generated pose.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

SCHEMA = 'somaforge_offline_contact_consensus_v1'


@dataclass(frozen=True)
class AggregationConfig:
    switch_cost: float = 2.0
    alignment_band_frames: int = 10

    def __post_init__(self):
        if not np.isfinite(self.switch_cost) or self.switch_cost < 0:
            raise ValueError('Invalid switching cost')
        if type(self.alignment_band_frames) is not int or self.alignment_band_frames < 0:
            raise ValueError('Invalid alignment band')

    def contract(self, *, fps, source_contract):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid fps')
        value = dict(schema=SCHEMA, config=asdict(self), fps=float(fps),
            semantics='offline_interaction_hypothesis_not_realized_contact',
            vote_unit='one_demonstration_per_shape_face_per_aligned_frame',
            temporal_model='binary_total_variation_equal_vote_disagreement',
            alignment='bounded_monotone_pose_only_dtw_no_contact_labels',
            source_contact_contract=source_contract)
        value['fingerprint'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return value


def binary_consensus(votes, switch_cost):
    """Exact time DP: average disagreement + switch_cost per transition.

    votes [E,T,G] are binary observations, not contact-point counts.
    With zero cost this is strict majority (ties prefer off). No topology
    from one face is merged with another. Smooth states and raw support both
    remain visible; smoothing never changes the underlying observations.
    """
    votes = np.asarray(votes)
    if votes.ndim != 3 or min(votes.shape) == 0 or not np.isin(votes,[0,1]).all():
        raise ValueError('Expected nonempty binary [demonstration,time,group]')
    if not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError('Invalid switching cost')
    support = votes.mean(0)
    if switch_cost == 0:
        return support > .5, support
    n,g=support.shape
    costs=np.stack((support[0],1-support[0]),axis=-1)
    previous=np.zeros((n,g,2),np.int8)
    for t in range(1,n):
        for state in (0,1):
            options=costs+np.array([switch_cost if state else 0,0 if state else switch_cost])
            previous[t,:,state]=options.argmin(-1)
        costs=np.stack((np.minimum(costs[:,0],costs[:,1]+switch_cost)+support[t],
                        np.minimum(costs[:,1],costs[:,0]+switch_cost)+1-support[t]),axis=-1)
    state=costs.argmin(-1); result=np.zeros((n,g),bool); columns=np.arange(g)
    for t in range(n-1,-1,-1):
        result[t]=state
        state=previous[t,columns,state]
    return result,support


def pose_features(q):
    q=np.asarray(q,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=36 or not np.isfinite(q).all():
        raise ValueError('Expected finite canonical [T,36]')
    quat=q[:,3:7]/np.maximum(np.linalg.norm(q[:,3:7],axis=-1,keepdims=True),1e-12)
    # Hemisphere-invariant rotation matrix features.
    w,x,y,z=quat.T
    rotation=np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                       2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                       2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1)
    return np.concatenate((q[:,:3]/.1,rotation,q[:,7:]),axis=-1)


def align_pose(reference, source, band=10):
    """Bounded DTW, endpoint-preserving, offline only; no contact supervision.

    Returns one original source-frame index per reference frame. No q or
    contacts are interpolated. The complete many-to-one path is also returned.
    """
    a,b=pose_features(reference),pose_features(source)
    if len(a)!=len(b) or band<0:
        raise ValueError('Alignment requires equally sampled complete demonstrations')
    n=len(a); cost=np.full((n+1,n+1),np.inf); cost[0,0]=0
    back=np.full((n,n),-1,np.int8)
    for i in range(n):
        lo=max(0,i-band); hi=min(n,i+band+1)
        dist=((b[lo:hi]-a[i])**2).mean(-1)
        for j,d in zip(range(lo,hi),dist):
            options=(cost[i,j],cost[i,j+1]+.01,cost[i+1,j]+.01)
            k=int(np.argmin(options)); cost[i+1,j+1]=options[k]+d+.001*abs(i-j)
            back[i,j]=k
    i=j=n-1; path=[]
    while i>=0 and j>=0:
        path.append((i,j)); k=back[i,j]
        if k==0: i-=1; j-=1
        elif k==1: i-=1
        elif k==2: j-=1
        else: raise ValueError('No valid alignment path')
    if i!=-1 or j!=-1:
        raise ValueError('Incomplete alignment')
    path=np.array(path[::-1],int); mapping=np.zeros(n,int)
    for t in range(n):
        candidates=path[path[:,0]==t,1]
        mapping[t]=candidates[len(candidates)//2]
    mapping[0]=0; mapping[-1]=n-1
    if (np.diff(mapping)<0).any() or (np.abs(mapping-np.arange(n))>band).any():
        raise AssertionError('Invalid time mapping')
    return mapping,path


def group_observations(pair_sequences):
    """Retain original multi-point clouds; deduplicate ONLY binary votes.

    Face identity is the shared Newton mapper's normal-ranked primary face.
    Candidate alternatives are retained on the untouched input pair records.
    """
    if not pair_sequences or not pair_sequences[0]:
        raise ValueError('No observations')
    if any(len(s)!=len(pair_sequences[0]) for s in pair_sequences):
        raise ValueError('Unequal time dimensions')
    keys=sorted({(p['part'],p['robot_shape'],p['surface']) for seq in pair_sequences for row in seq for p in row})
    if not keys:
        raise ValueError('No contact groups')
    index={k:i for i,k in enumerate(keys)}
    votes=np.zeros((len(pair_sequences),len(pair_sequences[0]),len(keys)),bool)
    for e,seq in enumerate(pair_sequences):
        for t,row in enumerate(seq):
            for p in row:
                if not p['allocated'] or p['surface']<0:
                    raise ValueError('Unallocated or unmapped contact')
                votes[e,t,index[(p['part'],p['robot_shape'],p['surface'])]]=True
    return keys,votes


def equal_demo_patch_cloud(clouds):
    """Unmodified point sets, with each contributing demo carrying unit mass.

    An observed-cloud medoid is a representative, not a synthetic averaged
    contact point. Preserve all clouds so multimodality is not erased.
    """
    clouds=[np.asarray(c,float) for c in clouds]
    if not clouds or any(c.ndim!=2 or c.shape[1]!=3 or len(c)==0 or not np.isfinite(c).all() for c in clouds):
        raise ValueError('Expected nonempty finite point clouds')
    centers=np.array([np.median(c,axis=0) for c in clouds])
    distance=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)
    medoid=int(distance.sum(1).argmin())
    return dict(representative_cloud=clouds[medoid],representative_index=medoid,
        points=np.concatenate(clouds),
        point_weights=np.concatenate([np.full(len(c),1/(len(clouds)*len(c))) for c in clouds]),
        demo_centers=centers,center_dispersion_m=float(np.median(distance[medoid])))


def first_surface_acquisitions(mask, surfaces, *, window, occupancy):
    mask = np.asarray(mask, dtype=bool)
    surfaces = np.asarray(surfaces)
    if mask.ndim != 2 or surfaces.shape != mask.shape:
        raise ValueError('Expected matching [frame, endpoint] arrays')
    if window < 1 or not 0 < occupancy <= 1:
        raise ValueError('Invalid observation window')
    events = []
    needed = int(np.ceil(window * occupancy))
    for part in range(mask.shape[1]):
        for surface in np.unique(surfaces[mask[:, part], part]):
            active = mask[:, part] & (surfaces[:, part] == surface)
            for frame in np.flatnonzero(active):
                if frame + window > len(mask):
                    break
                if active[frame:frame + window].sum() < needed:
                    continue
                if frame != 0:
                    events.append(dict(frame=int(frame), part_index=part,
                                       surface=int(surface), kind='first_surface_acquisition'))
                break
    return sorted(events, key=lambda row: (row['frame'], row['part_index']))


def release_supervision(active, certified_release=None):
    """Require external action evidence; an all-off vote alone is insufficient."""
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2:
        raise ValueError('Expected [source, frame] observations')
    if certified_release is None:
        return np.zeros(active.shape[1], dtype=bool)
    certified_release = np.asarray(certified_release)
    if certified_release.dtype != np.bool_ or certified_release.shape != active.shape:
        raise ValueError('Release evidence must be a boolean [source, frame] array')
    return ((~active & certified_release).sum(axis=0) > len(active) / 2)


def match_pose_anchors(reference, source, output_anchors, source_centers, *, band=10):
    """Original ordered pose-match DP, with independently observed source centers.

    Centers are sampling-clock evidence, not reference-frame deadlines. Returns
    anchors only; continuous callers must separately constrain the whole clock.
    """
    a, b = pose_features(reference), pose_features(source)
    edges, centers = np.asarray(output_anchors, float), np.asarray(source_centers, float)
    if (len(a) != len(b) or edges.shape != centers.shape or edges.ndim != 1 or len(edges) < 2
            or not np.isfinite(edges).all() or not np.isfinite(centers).all()
            or edges[0] != 0 or edges[-1] != len(a)-1
            or centers[0] != 0 or centers[-1] != len(a)-1
            or np.any(np.diff(edges) <= 0) or type(band) is not int or band < 0):
        raise ValueError('Invalid observed pose anchor clocks')
    candidates = [np.array([int(t)]) if i in (0, len(edges)-1)
                  else np.arange(max(1, int(np.ceil(t-band))), min(len(a)-2, int(np.floor(t+band)))+1)
                  for i, t in enumerate(centers)]
    if any(not len(c) for c in candidates):
        raise ValueError('No event-order-preserving alignment')
    costs = np.array([0.]); backs = []
    for i in range(1, len(edges)):
        previous, current = candidates[i-1], candidates[i]
        options = np.where(previous[:, None] < current[None, :], costs[:, None], np.inf)
        back = options.argmin(axis=0)
        target = np.asarray([np.interp(edges[i], np.arange(len(a)), channel) for channel in a.T])
        local = ((b[current]-target)**2).mean(-1)+1e-9*(current-centers[i])**2
        costs = options[back, np.arange(len(current))]+local
        backs.append(back)
    if not np.isfinite(costs).any():
        raise ValueError('No event-order-preserving alignment')
    j = int(costs.argmin()); selected = [int(candidates[-1][j])]
    for i in range(len(edges)-1, 0, -1):
        j = int(backs[i-1][j]); selected.append(int(candidates[i-1][j]))
    return np.asarray(selected[::-1])


def align_event_anchors(reference, source, boundaries, *, band=10):
    """Match an explicit shared event sequence using pose evidence, not dropouts.

    Event identity and order are input, including repeated same-surface actions.
    Dynamic programming chooses strictly ordered source times within the band.
    One whole-body map transports observations; it never creates contact truth.
    """
    from scipy.interpolate import PchipInterpolator
    a, b = pose_features(reference), pose_features(source)
    edges = np.asarray(boundaries)
    if (len(a) != len(b) or edges.ndim != 1 or len(edges) < 2
            or not np.issubdtype(edges.dtype, np.integer)
            or edges[0] != 0 or edges[-1] != len(a)-1
            or np.any(np.diff(edges) <= 0) or type(band) is not int or band < 0):
        raise ValueError('Expected complete, strictly ordered event boundaries')
    selected = match_pose_anchors(reference, source, edges, edges, band=band)
    continuous = PchipInterpolator(edges, selected)(np.arange(len(a)))
    # Bound the shared clock as well as its anchors; this preserves monotonicity.
    clock = np.arange(len(a))
    continuous = np.clip(continuous, np.maximum(0, clock-band), np.minimum(len(a)-1, clock+band))
    mapping = np.rint(continuous).astype(int)
    if np.any(np.diff(mapping) < 0):
        raise AssertionError('Non-monotone event clock')
    return mapping, selected


def phase_support_penalties(positions, rotations, tasks, *, scale, normal_mode='upper'):
    """Soft event support prior; output contact still requires Newton requery.

    tasks: (start, inclusive_end, body, local_material_cloud, normal, height).
    A step follows one material sample, while successive steps may choose
    different samples. Rotation of the body about its contact is permitted.
    """
    import torch
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Expected positive residual scale')
    if normal_mode not in ('upper', 'match'):
        raise ValueError('Unknown normal objective')
    normal_loss = positions.new_zeros(())
    tangent_loss = positions.new_zeros(())
    for start, end, body, cloud, normal, target in tasks:
        if not 0 <= start < end < len(positions):
            raise ValueError('Invalid support phase')
        points = positions[start:end+1, body, None, :] + torch.einsum(
            'tij,kj->tki', rotations[start:end+1, body], cloud)
        minimum, chosen = (points @ normal).min(-1)
        error = (minimum-target)/scale
        if normal_mode == 'upper':
            error = error.relu()
        normal_loss = normal_loss + error.square().mean()
        steps = torch.arange(end-start, device=positions.device)
        delta = points[steps+1, chosen[:-1]] - points[steps, chosen[:-1]]
        tangent = delta - (delta*normal).sum(-1, keepdim=True)*normal
        tangent_loss = tangent_loss + (tangent/scale).square().mean()
    return normal_loss/max(1, len(tasks)), tangent_loss/max(1, len(tasks))
