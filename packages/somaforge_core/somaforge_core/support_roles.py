"""Preserve declared contact roles without inferring physical support."""
import numpy as np


def complete_event_support_phases(phases, events, mask, surfaces):
    """Keep explicit task phases; contact continuity cannot add static roles.

    The historical name is retained for the diagnostic optimizer's import.
    Its output is declared contact intent, not an actual support assessment.
    """
    result=[dict(row) for row in phases]
    mask=np.asarray(mask,bool);surfaces=np.asarray(surfaces)
    if mask.shape!=surfaces.shape or mask.ndim!=2 or mask.shape[1]!=6:
        raise ValueError('Expected actual six-part surface evidence')
    for row in result:
        row['role_source']='declared_contact_keep_intent'
        row['actual_support_status']='unknown_without_solver_loads'
    return sorted(result,key=lambda row:(row['start'],row['end'],row['part']))
