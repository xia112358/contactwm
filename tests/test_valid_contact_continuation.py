import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from generator.research.archived_experiments import adjacent_events as m


def test_only_adjacent_climbing_transitions():
    samples=[SimpleNamespace(motion_id=0,current_frame=a,target_frame=b) for a,b in [(0,10),(10,20),(20,30),(50,60),(60,70)]]
    roles=[[2,2,1,1,0,0],[0,0,2,2,1,0],[1,1,0,0,0,0],[0,0,1,0,0,0],[1,1,0,0,0,0]]
    surfaces=[[0,0,1,1,-1,-1],[-1,-1,1,1,1,-1],[1,1,-1,-1,-1,-1],[-1,-1,1,-1,-1,-1],[0,0,-1,-1,-1,-1]]
    assert m.adjacent_pairs(samples,roles,surfaces,range(5))==[(0,1),(1,2)]


def test_no_cross_motion_pair():
    samples=[SimpleNamespace(motion_id=i,current_frame=i*10,target_frame=(i+1)*10) for i in range(2)]
    assert m.adjacent_pairs(samples,[[1]*6]*2,[[1]*6]*2,[0,1])==[]
