import numpy as np
from somaforge_core.support_roles import complete_event_support_phases


def test_continuous_contact_does_not_create_static_support():
    mask=np.zeros((8,6),bool);mask[:,0]=True;mask[:3,1]=True
    surface=np.where(mask,0,-1)
    events=[dict(segment_id='first',start_frame=0,end_frame=7,persistent_parts=[])]
    rows=complete_event_support_phases([],events,mask,surface)
    assert rows==[]
    assert events[0]['persistent_parts']==[]


def test_surface_transfer_or_hole_does_not_create_static_role():
    mask=np.ones((8,6),bool);surface=np.zeros((8,6),int)
    mask[3,0]=False;surface[4:,1]=1
    rows=complete_event_support_phases([], [dict(segment_id='e',start_frame=0,end_frame=7)],mask,surface)
    assert rows==[]


def test_explicit_phase_is_not_extended_by_contact_continuity():
    mask=np.zeros((8,6),bool);mask[:,0]=True;surface=np.where(mask,0,-1)
    phases=[dict(event_id='e',start=3,end=7,part=0,surface=0)]
    rows=complete_event_support_phases(phases,[dict(segment_id='e',start_frame=0,end_frame=7)],mask,surface)
    assert [(r['start'],r['end']) for r in rows]==[(3,7)]
    assert rows[0]['actual_support_status']=='unknown_without_solver_loads'
