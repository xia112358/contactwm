import numpy as np
import pytest

import align_recorded_action_clock as clock


def evidence(length=30):
    mask=np.zeros((length,6),bool);mask[:,0]=True
    faces=np.full(mask.shape,-1,int);faces[:,0]=0
    return dict(contact_part_mask=mask,contact_surface=faces,fps=50.)


def test_fresh_source_clock_aligns_stable_event_without_editing_or_dropping(monkeypatch):
    data=evidence();data['contact_part_mask'][5,2]=True;data['contact_surface'][5,2]=1
    data['contact_part_mask'][11:,2]=True;data['contact_surface'][11:,2]=1
    original=data['contact_part_mask'].copy()
    monkeypatch.setattr(clock,'observations',lambda *args:data)
    intervals=[dict(segment_id='reach',start_frame=0,end_frame=5),
               dict(segment_id='next',start_frame=5,end_frame=29)]
    aligned,report=clock.align('motion','labels',intervals)
    assert aligned[0]['end_frame']==aligned[1]['start_frame']==13
    assert [r['segment_id'] for r in aligned]==['reach','next']
    assert intervals[0]['end_frame']==5
    np.testing.assert_array_equal(data['contact_part_mask'],original)
    assert report['source_self_audit']['passed']
    assert report['omitted_action_count']==0


def test_matching_source_clock_is_not_shifted(monkeypatch):
    data=evidence();monkeypatch.setattr(clock,'observations',lambda *args:data)
    intervals=[dict(segment_id='one',start_frame=0,end_frame=29)]
    aligned,report=clock.align('motion','labels',intervals)
    assert aligned==intervals and report['shifts']==[]


def test_missing_terminal_evidence_cannot_be_pruned_or_weakened(monkeypatch):
    data=evidence(10);data['contact_part_mask'][9,2]=True;data['contact_surface'][9,2]=1
    monkeypatch.setattr(clock,'observations',lambda *args:data)
    with pytest.raises(ValueError,match='terminal boundary'):
        clock.align('motion','labels',[dict(segment_id='one',start_frame=0,end_frame=9)])
