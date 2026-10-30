import numpy as np
import pytest
from generator.demonstration_teacher import DemonstrationTeacher


def test_pose_check_accepts_quaternion_sign_but_not_position_error():
    from generator.demonstration_teacher import assert_same_pose
    q=np.zeros(36);q[3]=1
    same=q.copy();same[3]=-1
    assert_same_pose(q,same)
    same[0]=.01
    with pytest.raises(AssertionError):assert_same_pose(q,same)


def make_teacher(**kwargs):
    q=np.zeros((11,36));q[:,3]=1;q[:,0]=np.arange(11)*.01
    a=np.zeros((11,6),bool);a[:,0]=True;a[5:,2]=True
    s=np.where(a,0,-1)
    events=[dict(sample=i,current_frame=0 if i==0 else 5,target_frame=f,active=a[f],surface=s[f]) for i,f in enumerate((5,10))]
    return DemonstrationTeacher(q,a,s,events,**kwargs),q,a,s


def test_alignment_uses_state_not_number_of_queries():
    teacher,q,a,s=make_teacher()
    first,_=teacher.align(q[0],a[0],s[0])
    again,_=teacher.align(q[0],a[0],s[0])
    later,_=teacher.align(q[6],a[6],s[6])
    assert first==again==0 and later==1


def test_missing_contact_prevents_skipping_boundary():
    teacher,q,a,s=make_teacher()
    sample,audit=teacher.align(q[6],a[0],s[0])
    assert sample==0 and audit['completed_events']==0


def test_far_state_does_not_receive_arbitrary_nearest_label():
    teacher,q,a,s=make_teacher();far=q[0].copy();far[0]=10
    sample,audit=teacher.align(far,a[0],s[0])
    assert sample is None and audit['reason']=='outside_demonstration_alignment_coverage'
    assert teacher.progress==0


def test_validation_cannot_enter_teacher_bank():
    with pytest.raises(ValueError):make_teacher(split='validation')


def test_retreat_does_not_change_transition_or_its_duration():
    teacher,q,a,s=make_teacher()
    _,near=teacher.align(q[4],a[4],s[4])
    sample,back=teacher.align(q[1],a[1],s[1])
    assert sample==0 and near['duration_frames']==back['duration_frames']==5
    assert teacher.completed_frame==0


def test_off_path_contact_block_does_not_create_one_frame_duration():
    teacher,q,a,s=make_teacher();student=q[6].copy();student[1]=.13
    sample,audit=teacher.align(student,a[0],s[0])
    assert sample==0 and audit['completed_events']==0
    assert 'matched_frame' not in audit and 'timing' not in audit
    assert audit['duration_frames']==5
    assert teacher.completed_frame==0


def test_repeated_failed_queries_do_not_reduce_duration():
    teacher,q,a,s=make_teacher();student=q[6].copy();student[1]=.13
    values=[teacher.align(student,a[0],s[0])[1]['duration_frames'] for _ in range(4)]
    assert values==[values[0]]*4


def test_near_endpoint_still_supervises_whole_transition():
    teacher,q,a,s=make_teacher()
    sample,audit=teacher.align(q[4],a[4],s[4])
    assert sample==0 and audit['duration_frames']==5


def test_completed_boundary_not_undone_when_state_retreats():
    teacher,q,a,s=make_teacher();teacher.align(q[6],a[6],s[6])
    sample,audit=teacher.align(q[3],a[3],s[3])
    assert sample==1 and teacher.completed_frame==5
    assert audit['duration_frames']>=5


def test_static_reference_does_not_use_velocity_to_invent_timing():
    teacher,q,a,s=make_teacher();teacher.q[:,:3]=0
    student=teacher.q[0].copy();student[1]=.1
    sample,audit=teacher.align(student,a[0],s[0])
    assert sample==0 and audit['duration_frames']==5


def test_interior_frames_cannot_affect_target_or_duration():
    first,q,a,s=make_teacher();second,_,_,_=make_teacher()
    second.q=second.q.copy();second.q[[1,2,3,4,6,7,8,9],:3]=100
    for f in (0,4,6):
        x=first.align(q[f],a[f],s[f]);y=second.align(q[f],a[f],s[f])
        assert x==y


def test_missing_boundary_is_rejected():
    teacher,q,a,s=make_teacher();events=[dict(e) for e in teacher.events]
    events[0].pop('current_frame')
    with pytest.raises(ValueError,match='explicit start/end'):
        DemonstrationTeacher(q,a,s,events)


def test_repeating_same_state_does_not_advance_another_event():
    teacher,q,a,s=make_teacher()
    sample,_=teacher.align(q[9],a[9],s[9])
    assert sample==1
    for _ in range(4):
        sample,audit=teacher.align(q[9],a[9],s[9])
        assert sample==1 and audit['completed_events']==1
