import sys
from pathlib import Path
import numpy as np
from generator.research.archived_experiments.ordered_contact_teacher import OrderedContactTeacher


def make():
    return OrderedContactTeacher([dict(sample=i,current_frame=i*10,target_frame=(i+1)*10,
        active=[True,True,False,False,False,False],surface=[0,0,-1,-1,-1,-1],roles=[1,2,0,0,0,0]) for i in range(2)])


def test_failed_target_stays_and_valid_advances_once():
    t=make();q=np.zeros(36);q2=q.copy();q2[0]=.01
    args=(q,q2,[True,True,False,False,False,False],[0,0,-1,-1,-1,-1])
    assert not t.advance(*args,terminal=True)
    assert t.align(q,None,None)[0]==0
    assert t.advance(*args,terminal=False)
    assert t.align(q,None,None)[0]==1
    assert t.align(q,None,None)[1]['duration_frames']==10


def test_identical_pose_cannot_consume_events():
    t=make();q=np.zeros(36)
    assert not t.advance(q,q,[True,True,False,False,False,False],[0,0,-1,-1,-1,-1],terminal=False)


def test_observed_topology_not_intent_controls_progress():
    t=make();q=np.zeros(36);q2=q.copy();q2[0]=.01
    assert not t.advance(q,q2,[True,False,False,False,False,False],
                         [0,0,-1,-1,-1,-1],terminal=False)
    assert not t.advance(q,q2,[True,True,False,False,False,False],
                         [0,1,-1,-1,-1,-1],terminal=False)
    assert t.advance(q,q2,[True,True,False,False,False,False],
                     [0,0,-1,-1,-1,-1],terminal=False)


def test_nonfinite_endpoint_rejected():
    t=make();q=np.zeros(36);q2=q.copy();q2[0]=np.nan
    assert not t.advance(q,q2,[True,True,False,False,False,False],
                         [0,0,-1,-1,-1,-1],terminal=False)
