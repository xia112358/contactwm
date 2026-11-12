"""Online handoff events using the existing native-substep support state machine."""
import torch
from .contact_substep_support import SubstepContactMonitor


class SupportEventContactMonitor:
    def __init__(self, env, reader):
        self.env=env
        self.monitor=SubstepContactMonitor(env,reader,self._sample)
        self.dt=self.monitor.dt;self.substeps=self.monitor.substeps
        self.exit_steps=self.monitor.support.exit_steps
        self.blocked=torch.zeros((env.N,6),dtype=torch.bool,device=env.device)
        self.delivered=torch.zeros_like(self.blocked)
        self.new_event=torch.zeros_like(self.blocked)

    def begin_stage(self, ids, initial_contact, *, continuing=False):
        self.blocked[ids]=initial_contact
        if continuing:self.blocked[ids] |= self.monitor.support.support[ids]
        self.delivered[ids]=False
        self.new_event[ids]=False

    def begin_control(self, *args):
        self.new_event.zero_()

    def _sample(self,tick,substep,observation,features,support):
        # Reuse the same measured exit/short-gap rules as source segmentation.
        released=support['exited']|(self.monitor.support.off>=self.exit_steps)
        self.blocked &= ~released
        ready=support['event_ready']&observation['known']&observation['active']
        event=ready&~self.blocked&~self.delivered
        self.new_event |= event
        self.delivered |= event
        p,R=self.monitor.reader.last_part_poses
        self.last=dict(features,contact_loss_point=observation['point'].clone(),
            contact_loss_position=p.clone(),contact_loss_rotation=R.clone(),
            contact_loss_candidate=torch.zeros_like(self.blocked,dtype=torch.long),
            contact_raw_stable=features['contact_stable'],contact_raw_point=observation['point'].clone(),
            contact_raw_position=p.clone(),contact_raw_rotation=R.clone())

    def summary(self):
        return dict(self.last,**self.monitor.summary(),support_new_event=self.new_event.clone())

    def __enter__(self):
        self.monitor.__enter__();return self

    def __exit__(self,*args):
        return self.monitor.__exit__(*args)
