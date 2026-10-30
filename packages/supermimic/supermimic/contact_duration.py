"""GPU per-segment duration bands, with a fixed initial-collection prior."""
import torch
import torch.nn.functional as F


class DurationBandHistory:
    """Use initial priors during bootstrap, then the last reliable batch band."""
    def __init__(self, initial_centers, bootstrap_batches=5):
        if bootstrap_batches < 0:
            raise ValueError('Bootstrap batch count must be nonnegative')
        self.initial = [int(x) for x in initial_centers]
        self.latest = self.initial.copy()
        self.reliable_seen = [False] * len(self.initial)
        self.bootstrap_batches = bootstrap_batches
        self.batches = 0

    def select(self, data, halfwidth=2, min_success=8, min_mass=.6):
        bootstrap = self.batches < self.bootstrap_batches
        fallback = self.initial if bootstrap else self.latest
        ids, stops, report = select_duration_batch(data, fallback, halfwidth, min_success, min_mass)
        report.update(history_batch=self.batches+1, bootstrap_batches=self.bootstrap_batches,
                      history_phase='bootstrap' if bootstrap else 'last_reliable')
        for row in report['segments']:
            g = row['segment']-1
            row['initial_center'] = self.initial[g]
            row['last_reliable_center'] = self.latest[g] if self.reliable_seen[g] else None
            if row['fallback']:
                row['source'] = ('initial_prior' if bootstrap else
                                 'last_reliable' if self.reliable_seen[g] else 'initial_no_reliable_history')
            else:
                self.latest[g] = row['center']
                self.reliable_seen[g] = True
        self.batches += 1
        return ids, stops, report

    def state_dict(self):
        return dict(initial=self.initial.copy(), latest=self.latest.copy(),
                    reliable_seen=self.reliable_seen.copy(), bootstrap_batches=self.bootstrap_batches,
                    batches=self.batches)

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(state['initial'], state['bootstrap_batches'])
        obj.latest = list(state['latest'])
        obj.reliable_seen = list(state['reliable_seen'])
        obj.batches = int(state['batches'])
        assert len(obj.latest) == len(obj.initial) == len(obj.reliable_seen)
        return obj


@torch.no_grad()
def select_duration_batch(data, prior_centers, halfwidth=2, min_success=8, min_mass=.6):
    if halfwidth < 0 or min_success < 1 or not 0 < min_mass <= 1:
        raise ValueError('Invalid duration filter settings')
    assert data['done'].any(0).all(), 'Complete attempts required'
    frames = data['done'].long().argmax(0) + 1
    slots = torch.arange(len(frames), device=frames.device)
    success = ~data['term'][frames - 1, slots]
    segment = data['goal_index'][0]
    prior = torch.as_tensor(prior_centers, device=frames.device, dtype=torch.long)
    G, H = len(prior), data['done'].shape[0] + 1
    assert bool((prior > 0).all()) and bool((segment < G).all())
    # One histogram and sliding-window reduction, no per-segment GPU syncs.
    hist = torch.bincount((segment*H + frames)[success], minlength=G*H).reshape(G,H)
    bands = F.pad(hist, (halfwidth,halfwidth)).unfold(1,2*halfwidth+1,1).sum(-1)
    axis = torch.arange(H, device=frames.device)
    peaks = bands == bands.max(-1,keepdim=True).values
    # Ties between equally dense windows prefer the initial prior.
    distance = (axis[None]-prior[:,None]).abs().masked_fill(~peaks,torch.iinfo(torch.long).max)
    window_center = distance.argmin(-1)
    inside = (axis[None]-window_center[:,None]).abs() <= halfwidth
    local_hist = hist*inside
    local_n = local_hist.sum(-1)
    # Actual observed lower median within the densest window, not the midpoint
    # between separate peaks. Recheck mass after recentering.
    candidate = (local_hist.cumsum(-1) >= ((local_n+1)//2).clamp_min(1)[:,None]).long().argmax(-1)
    nsuccess = hist.sum(-1)
    mass = (hist*((axis[None]-candidate[:,None]).abs()<=halfwidth)).sum(-1)/nsuccess.clamp_min(1)
    reliable = (nsuccess>=min_success)&(mass>=min_mass)
    centers = torch.where(reliable,candidate,prior)
    lower=(centers[segment]-halfwidth).clamp_min(1)
    upper=centers[segment]+halfwidth
    keep=frames>=lower
    cut=keep & (frames>upper)
    stop_steps=torch.minimum(frames,upper)
    truncated=torch.bincount(segment[cut],minlength=G)
    totals = torch.bincount(segment,minlength=G)
    retained = torch.bincount(segment[keep],minlength=G)
    kept_failed = torch.bincount(segment[keep & ~success],minlength=G)
    packed = torch.stack((totals,nsuccess,centers,retained,kept_failed,reliable,prior,mass,truncated),-1).cpu().tolist()
    reports=[]
    for g,(n,ns,c,k,kf,ok,p,m,nt) in enumerate(packed):
        if not n:continue
        reports.append(dict(segment=g+1,attempts=int(n),successes=int(ns),center=int(c),
            lower=max(1,int(c)-halfwidth),upper=int(c)+halfwidth,prior_center=int(p),
            retained=int(k),discarded=int(n-k),retained_failures=int(kf),truncated=int(nt),unchanged=int(k-nt),
            fallback=not bool(ok),source='batch' if ok else 'initial_prior',
            candidate_mass=m))
    n_kept=sum(r['retained'] for r in reports)
    return keep.nonzero().flatten(), stop_steps, dict(mode='dynamic_band_truncate_long_discard_short',
        truncated=sum(r['truncated'] for r in reports),unchanged=sum(r['unchanged'] for r in reports),
        halfwidth=halfwidth,min_success=min_success,min_mass=min_mass,total=len(frames),
        retained=n_kept,discarded=len(frames)-n_kept,retention_rate=n_kept/len(frames),segments=reports)


class FixedDurationBand:
    """Frozen stage-one bands: early samples excluded, late capped; never recenter."""
    def __init__(self, centers):
        self.centers=list(centers)

    def select(self, data, halfwidth=2, min_success=8, min_mass=.6):
        done=data['done'];frames=done.long().argmax(0)+1;ids=torch.arange(len(frames),device=frames.device)
        return self.select_durations(frames,~data['term'][frames-1,ids],data['goal_index'][0],halfwidth,min_success,min_mass)

    @torch.no_grad()
    def select_durations(self, frames, success, segment, halfwidth=2, min_success=8, min_mass=.6):
        """Apply identical frozen bands directly to already-known stage lengths."""
        centers=torch.as_tensor(self.centers,device=frames.device)[segment]
        lower=(centers-halfwidth).clamp_min(1);upper=centers+halfwidth
        keep=frames>=lower;truncated=keep&(frames>upper);stops=torch.minimum(frames,upper)
        values=torch.stack((torch.ones_like(keep),success,keep,keep&~success,truncated),-1).long()
        totals=torch.zeros(len(self.centers),5,device=frames.device,dtype=torch.long)
        totals.scatter_add_(0,segment[:,None].expand_as(values),values)
        rows=[]
        for g,(n,passed,retained,failed,cut) in enumerate(totals.cpu().tolist()):
            if not n:continue
            c=self.centers[g]
            rows.append(dict(segment=g+1,attempts=n,successes=passed,center=c,lower=max(1,c-halfwidth),upper=c+halfwidth,prior_center=c,retained=retained,discarded=n-retained,retained_failures=failed,truncated=cut,fallback=True,source='fixed_inherited'))
        count=sum(r['retained'] for r in rows);total=len(frames)
        return keep.nonzero().flatten(),stops,dict(mode='fixed_inherited_band',halfwidth=halfwidth,total=total,retained=count,discarded=total-count,truncated=sum(r['truncated'] for r in rows),retention_rate=count/total,segments=rows,scope='frozen first-stage centers; shorter samples excluded; actual native deadline is upper bound')

    def state_dict(self):
        return dict(fixed_centers=self.centers)
