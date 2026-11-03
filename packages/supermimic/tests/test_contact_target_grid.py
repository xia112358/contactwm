import types
import torch
from supermimic.contact_target_grid import expand_target_grid


def test_grid_changes_commands_only_and_preserves_source_history():
    n=8
    masks=torch.zeros(n,6,dtype=torch.bool);masks[5,2]=True;masks[5,0]=True
    keep=torch.zeros_like(masks);keep[5,0]=True
    clock=types.SimpleNamespace(length=n,event_ends=torch.tensor([6]),
        segment_starts=torch.tensor([2]),segment_stops=torch.tensor([6]),
        segment_motion=torch.tensor([4]),case_ids=torch.tensor([28]),
        segment_local_bounds=[(2,6)],bounds=None,ends=torch.full((n,),6),
        frame_source_floor=torch.zeros(n,dtype=torch.long),target_masks=masks,
        keep_masks=keep,support_masks=keep.clone(),data={'point':torch.ones(n,6,3),
        's':torch.randn(n,71),'xi':torch.randn(n,34),'active':masks.clone()})
    original={k:v.clone() for k,v in clock.data.items()}
    grid=expand_target_grid(clock,.04,0)
    for k in original:assert torch.equal(clock.data[k],original[k])
    assert len(grid.event_ends)==9
    for i in range(9):
        assert torch.equal(grid.data['s'][i*n:(i+1)*n],original['s'])
        assert torch.equal(grid.data['xi'][i*n:(i+1)*n],original['xi'])
        assert grid.segment_starts[i]==2+i*n
        assert grid.frame_source_floor[grid.segment_starts[i]]==i*n
        target=grid.data['point'][grid.event_ends[i]-1]
        assert torch.equal(target[0],original['point'][5,0])
        assert torch.allclose(target[2]-original['point'][5,2],torch.tensor(grid.target_grid['offsets_w'][i]))
    assert torch.equal(grid.data['point'][4*n:5*n],original['point'])
    assert grid.target_grid['new_parts']==[2]


def test_native_metric_does_not_replace_first_contact_with_later_closer_point():
    from supermimic.contact_dmo import native_attempt_metrics
    state=torch.zeros(2,1,71);state[:,:,3]=1
    mask=torch.zeros(1,6,dtype=torch.bool);mask[:,2]=True
    clock=types.SimpleNamespace(event_ends=torch.tensor([1]),target_masks=mask,
        keep_masks=torch.zeros_like(mask),data={'point':torch.zeros(1,6,3),'s1':state[0]})
    points=torch.zeros(2,1,6,3);points[0,0,2,0]=.2;points[1,0,2,0]=.01
    first=torch.zeros(2,1,6,dtype=torch.bool);first[0,0,2]=True
    data=dict(done=torch.tensor([[False],[True]]),term=torch.zeros(2,1,dtype=torch.bool),
        goal_index=torch.zeros(2,1,dtype=torch.long),s1=state,point=points,
        active=mask[None].expand(2,1,6),known=torch.ones(2,1,6,dtype=torch.bool),contact_first=first)
    report=native_attempt_metrics(data,clock,None,first_contact=True)
    assert abs(report['first_contact_error_cm']-20)<1e-4
    assert report['contact_metric']=='first_native_onset'
    first.zero_()
    missing=native_attempt_metrics(data,clock,None,first_contact=True)
    assert missing['contact_error_samples']==0
    assert missing['first_contact_error_cm'] is None


def test_single_segment_gets_all_student_worlds():
    from supermimic.contact_library import case_world_counts,case_world_assignment
    clock=types.SimpleNamespace(segment_motion=torch.tensor([0,0,1,2,2,2]))
    counts=case_world_counts(clock,3,4098,segment_index=3)
    assert counts==[1,1,4096]
    task_ids=torch.repeat_interleave(torch.arange(3),torch.tensor(counts))
    ids,assigned,background=case_world_assignment(clock,task_ids,3)
    assert len(ids)==4096 and len(background)==2
    assert bool((assigned==3).all())
    assert case_world_counts(clock,3,12)==[4,2,6]
