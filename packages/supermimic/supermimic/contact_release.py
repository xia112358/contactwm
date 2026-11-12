"""Native absence plus measured part motion; shared reference-release semantics."""
import numpy as np
from .contact_stability import CONTRACT


def native_release_history(active, known, position, exit_steps):
    """Return state after every actual solve; caller separates source traces."""
    active=np.asarray(active);known=np.asarray(known);position=np.asarray(position)
    phase=np.zeros(active.shape[1],dtype=bool);absent=np.zeros(active.shape[1],dtype=np.int32)
    anchor=position[0].copy();valid=np.zeros_like(phase);onset=np.full(len(phase),-1,dtype=np.int64)
    phases=[];absences=[];onsets=[]
    for t in range(len(active)):
        a=active[t]&known[t]
        newly=a&~phase;onset=np.where(newly,t,onset)
        absent=np.where(known[t]&~a,absent+1,0)
        moved=valid&(np.linalg.norm(position[t]-anchor,axis=-1)>CONTRACT['row_match_distance_m'])
        released=known[t]&~a&((absent>=exit_steps)|moved)
        phase=(phase|a)&~released
        anchor=np.where((a|~valid)[:,None],position[t],anchor);valid=known[t]
        phases.append(phase.copy());absences.append(absent.copy());onsets.append(onset.copy())
    return np.stack(phases),np.stack(absences),np.stack(onsets)
