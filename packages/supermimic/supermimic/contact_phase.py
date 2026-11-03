"""Reference landing phase boundaries, independent of target proximity.

Only the demonstrated target flight is armed. Before its reference release
boundary, old supports and contact flicker cannot become the landing event.
After that boundary, native known-inactive and known-active confirmations
plus support qualification select the student landing. Force/motion qualify
support only; native activation is unchanged.
"""
import torch

VERSION='reference_stable_support_v4'


def reference_release_starts(clock, frames):
    from .contact_library import segment_ranges
    starts=segment_ranges(clock)[0]
    result=torch.full((len(starts),6),-1,dtype=torch.long,device=clock.data['active'].device)
    for segment,(begin,end) in enumerate(zip(starts.tolist(),clock.event_ends.tolist())):
        run=torch.zeros(6,dtype=torch.long,device=result.device)
        for t,(active,known) in enumerate(zip(clock.data['active'][begin:end],clock.data['known'][begin:end])):
            run=torch.where(known&~active,run+1,0)
            result[segment]=torch.where(run==frames,t-frames+1,result[segment])
    new=clock.target_masks[clock.event_ends-1]&~clock.keep_masks[clock.event_ends-1]
    if (new&(result<0)).any():
        raise ValueError('A new landing target must have a demonstrated confirmed release phase')
    return result


class LandingPhaseTracker:
    def __init__(self,release_starts,required,frames):
        self.release_starts=release_starts
        self.required=required
        self.frames=frames
        self.step=torch.zeros(release_starts.shape[0],dtype=torch.long,device=release_starts.device)
        self.release_run=torch.zeros_like(release_starts)
        self.released=torch.zeros_like(required)
        self.active_run=torch.zeros_like(release_starts)
        self.confirmed=torch.zeros_like(required)

    def reset(self,slots):
        self.step[slots]=0;self.release_run[slots]=0;self.released[slots]=False
        self.active_run[slots]=0;self.confirmed[slots]=False

    def update(self,active,known,stable):
        armed=self.step[:,None]>=self.release_starts
        self.release_run=torch.where(armed&known&~active,self.release_run+1,0)
        self.released|=self.release_run>=self.frames
        self.active_run=torch.where(self.released&known&active,self.active_run+1,0)
        hit=(self.active_run>=self.frames)&stable&self.required&~self.confirmed
        self.confirmed|=hit;self.step+=1
        return hit


def landing_phase_events(active,known,release_starts,required,frames,live,stable):
    tracker=LandingPhaseTracker(release_starts,required,frames)
    return torch.stack([tracker.update(a,k&l[:,None],s) for a,k,l,s in zip(active,known,live,stable)])
