"""Static demonstration alignment for online imitation; not a recovery oracle.

Progress and source identity are teacher-only. Alignment tolerances reject
unsupported supervision and never define or modify contact activation.
"""
from dataclasses import dataclass
import numpy as np

SCHEMA = 'demonstration207_transition_teacher_v3'


def assert_same_pose(actual, expected, atol=1e-5):
    """Quaternion q and -q represent the same pose; do not mutate either input."""
    aligned=np.array(actual,copy=True)
    if np.dot(aligned[3:7],np.asarray(expected)[3:7])<0:aligned[3:7]*=-1
    np.testing.assert_allclose(aligned,expected,atol=atol,rtol=0)


@dataclass(frozen=True)
class AlignmentLimits:
    root_m: float = .30
    joint_rmse_rad: float = .50
    orientation_rad: float = .75


class DemonstrationTeacher:
    def __init__(self, q, active, surface, events, *, start_frame=0,
                 limits=AlignmentLimits(), split='train'):
        if split != 'train': raise ValueError('Validation/test trajectories cannot be teachers')
        self.q = np.asarray(q)
        self.active = np.asarray(active, bool)
        self.surface = np.asarray(surface)
        if self.q.ndim != 2 or self.q.shape[1] != 36 or not np.isfinite(self.q).all():
            raise ValueError('Invalid demonstrated q')
        if self.active.shape != (len(self.q),6) or self.surface.shape != self.active.shape:
            raise ValueError('Demonstration contact timeline mismatch')
        self.events = sorted(events, key=lambda e:e['target_frame'])
        if not self.events or self.events[-1]['target_frame'] >= len(self.q):
            raise ValueError('Invalid demonstration event boundaries')
        for event in self.events:
            if 'current_frame' not in event or not 0<=event['current_frame']<event['target_frame']:
                raise ValueError('Each transition requires explicit start/end boundaries')
        self.progress = int(start_frame)
        self.completed_frame = int(start_frame)
        self.event_index = next((i for i,e in enumerate(self.events) if e['target_frame']>start_frame),len(self.events))
        self.limits = limits
        self._last_advanced_state = None

    def _errors(self, poses, q):
        root=np.linalg.norm(poses[:,:3]-q[:3],axis=1)
        joint=np.sqrt(np.mean((poses[:,7:]-q[7:])**2,axis=1))
        quat=poses[:,3:7]/np.linalg.norm(poses[:,3:7],axis=1,keepdims=True)
        angle=2*np.arccos(np.clip(np.abs(quat@(q[3:7]/np.linalg.norm(q[3:7]))),0,1))
        distance=np.sqrt((root/self.limits.root_m)**2+(joint/self.limits.joint_rmse_rad)**2+(angle/self.limits.orientation_rad)**2)
        return root,joint,angle,distance

    def align(self, q, active, surface):
        """Select an unfinished transition using only its boundary states.

        Endpoint-nearer plus matching Newton topology is a teacher alignment
        heuristic, not physical completion certification. Interior frames,
        inferred velocity and remaining-frame countdowns are never used.
        """
        q=np.asarray(q);active=np.asarray(active,bool);surface=np.asarray(surface)
        if q.shape!=(36,) or not np.isfinite(q).all() or np.linalg.norm(q[3:7])<1e-8:
            return None,dict(reason='invalid_student_state')
        if self.event_index>=len(self.events):return None,dict(reason='demonstration_finished')
        upcoming=self.events[self.event_index];end=upcoming['target_frame']
        start=upcoming['current_frame']
        root,joint,angle,distance=self._errors(self.q[[start,end]],q)
        wanted=np.asarray(upcoming['active'],bool);ws=np.asarray(upcoming['surface'])
        topology=bool(np.array_equal(active,wanted) and np.array_equal(surface[wanted],ws[wanted]))
        covered=(root<=self.limits.root_m)&(joint<=self.limits.joint_rmse_rad)&(angle<=self.limits.orientation_rad)
        if not covered.any():
            return None,dict(reason='outside_demonstration_alignment_coverage',transition_sample=upcoming['sample'],completed_events=self.event_index)
        state=np.concatenate((q,active,surface))
        repeated=self._last_advanced_state is not None and np.array_equal(state,self._last_advanced_state)
        advanced=None
        if topology and covered[1] and distance[1]<distance[0] and not repeated:
            self.completed_frame=end;self.progress=end;self.event_index+=1
            self._last_advanced_state=state.copy();advanced=upcoming['sample']
            if self.event_index>=len(self.events):
                return None,dict(reason='demonstration_finished',completed_events=self.event_index,advanced_transition=advanced)
            upcoming=self.events[self.event_index];start=upcoming['current_frame'];end=upcoming['target_frame']
            root,joint,angle,distance=self._errors(self.q[[start,end]],q)
            covered=(root<=self.limits.root_m)&(joint<=self.limits.joint_rmse_rad)&(angle<=self.limits.orientation_rad)
            if not covered.any():
                return None,dict(reason='outside_demonstration_alignment_coverage',completed_events=self.event_index,advanced_transition=advanced)
        audit=dict(completed_events=self.event_index,advanced_transition=advanced,
                   transition_sample=upcoming['sample'],transition_start_frame=start,
                   endpoint_root_error_m=float(root[1]),endpoint_joint_error_rad=float(joint[1]),
                   duration_semantics='full_demonstrated_transition_duration')
        return upcoming['sample'],dict(audit,reason='aligned',target_frame=upcoming['target_frame'],
                                      sample=upcoming['sample'],duration_frames=end-start)
