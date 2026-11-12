"""First native terrain contact after entry: airborne and supported entries differ.

The entry mask comes from a native solve at the actual reset/handoff state.
An airborne part is armed immediately. An initially contacting part is armed
only after the shared release debounce; row/face/material changes do not arm it.
Region membership selects arrival candidates after native activation. Raw contact
counts still determine whether initial support has released. Stability is
confirmed separately, without an elapsed-time deadline.
"""
import torch
import warp as wp
from .contact_stability import CONTRACT


@wp.kernel
def raw_contact_counts(groups: wp.array[int], query: wp.array[wp.bool], counts: wp.array[int]):
    row = wp.tid()
    if query[row]:
        wp.atomic_add(counts, groups[row], 1)


@wp.kernel
def part_positions(bodies:wp.array[wp.int64],p:wp.array2d[wp.vec3],current:wp.array[wp.vec3]):
    g=wp.tid();current[g]=p[g//6,int(bodies[g%6])]

@wp.kernel
def release_motion(counts:wp.array[int],known:wp.array[wp.bool],current:wp.array[wp.vec3],
                   anchor:wp.array[wp.vec3],valid:wp.array[wp.bool],distance:float,released:wp.array[wp.bool]):
    g=wp.tid();released[g]=False
    if not known[g//6]:
        valid[g]=False
        return
    if counts[g]>0 or not valid[g]:
        anchor[g]=current[g];valid[g]=True
    else:
        released[g]=wp.length(current[g]-anchor[g])>distance

@wp.kernel
def first_contact(counts: wp.array[int], arrivals: wp.array[int], known: wp.array[wp.bool],
                  required: wp.array[wp.bool], blocked: wp.array[wp.bool],
                  absent: wp.array[int], delivered: wp.array[wp.bool],
                  exit_steps: int, substep: wp.array[int],
                  event: wp.array[wp.bool], stop_substep: wp.array[int],
                  pending: wp.array[wp.bool], event_parts: wp.array[wp.bool],
                  done: wp.array[wp.bool], lost: wp.array[wp.bool], moved:wp.array[wp.bool]):
    g = wp.tid(); world = g // 6
    if done[g]:return
    # Unknown evidence cannot establish either release or arrival.
    if not known[world]:
        absent[g] = 0
        return
    active = counts[g] > 0
    if blocked[g] or pending[g]:
        present=active if blocked[g] else arrivals[g]>0
        if present:
            absent[g] = 0
        else:
            absent[g] += 1
            if absent[g] >= exit_steps or (blocked[g] and moved[g]):
                if blocked[g]:blocked[g] = False
                elif pending[g]:
                    lost[g] = True
                    pending[g] = False
                    delivered[g] = False
    if arrivals[g]>0 and not blocked[g] and not delivered[g] and required[g]:
        lost[g] = False
        delivered[g] = True
        event[g] = True
        event_parts[g] = True
        stop_substep[g] = substep[0]
        pending[g] = True


@wp.kernel
def confirm_contact(pending: wp.array[wp.bool], stable: wp.array[wp.bool],
                    known: wp.array[wp.bool], lost: wp.array[wp.bool],
                    steps: wp.array[int], done: wp.array[wp.bool]):
    g = wp.tid()
    if not pending[g] or done[g]:return
    steps[g] += 1
    # There is no elapsed-time confirmation deadline. Release debounce only
    # establishes an actual loss of contact, never a failure for settling slowly.
    done[g] = known[g // 6] and stable[g]


def part_joint_mask(joint_names):
    """Feet and knees share a leg; hands own their arm; waist stays controlled."""
    prefixes = (('left_hip_', 'left_knee_', 'left_ankle_'),
                ('right_hip_', 'right_knee_', 'right_ankle_'),
                ('left_shoulder_', 'left_elbow_', 'left_wrist_'),
                ('right_shoulder_', 'right_elbow_', 'right_wrist_'))
    limb = [[name.startswith(names) for name in joint_names] for names in prefixes]
    return torch.tensor([limb[0], limb[1], limb[2], limb[3], limb[0], limb[1]])


def held_joints(pending, joint_mask):
    return (pending[..., :, None] & joint_mask).any(-2)


def contact_decision(required, confirmed, reached, lost):
    """One completed part cannot terminate a still-running multi-target task."""
    failed_parts = required & confirmed & (lost | ~reached)
    all_confirmed = (~required | confirmed).all(-1)
    return failed_parts, all_confirmed


class NativeContactOnset:
    def __init__(self, worlds, device, exit_steps):
        self.worlds = worlds
        self.device = device
        self.exit_steps = exit_steps
        allocate = lambda size, dtype: wp.zeros(size, dtype=dtype, device=device)
        self.current=allocate(worlds*6,wp.vec3);self.anchor=allocate(worlds*6,wp.vec3)
        self.anchor_valid=allocate(worlds*6,wp.bool);self.motion_release=allocate(worlds*6,wp.bool)
        self.counts = allocate(worlds * 6, wp.int32)
        self.required = allocate(worlds * 6, wp.bool)
        self.blocked = allocate(worlds * 6, wp.bool)
        self.absent = allocate(worlds * 6, wp.int32)
        self.delivered = allocate(worlds * 6, wp.bool)
        self.event = allocate(worlds * 6, wp.bool)
        self.event_parts = allocate(worlds * 6, wp.bool)
        self.pending = allocate(worlds * 6, wp.bool)
        self.confirmation_steps = allocate(worlds * 6, wp.int32)
        self.done = allocate(worlds * 6, wp.bool)
        self.lost = allocate(worlds * 6, wp.bool)
        self.stop_substep = wp.full(worlds * 6, -1, dtype=wp.int32, device=device)

    def begin_stage(self, ids, initial_active, required, initial_absence=None):
        wp.to_torch(self.anchor_valid).reshape(self.worlds,6)[ids]=False
        wp.to_torch(self.motion_release).reshape(self.worlds,6)[ids]=False
        wp.to_torch(self.blocked).reshape(self.worlds, 6)[ids] = initial_active
        wp.to_torch(self.required).reshape(self.worlds, 6)[ids] = required
        wp.to_torch(self.absent).reshape(self.worlds, 6)[ids] = 0 if initial_absence is None else initial_absence
        wp.to_torch(self.delivered).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.event).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.event_parts).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.pending).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.confirmation_steps).reshape(self.worlds, 6)[ids] = 0
        wp.to_torch(self.done).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.lost).reshape(self.worlds, 6)[ids] = False
        wp.to_torch(self.stop_substep).reshape(self.worlds, 6)[ids] = -1

    def begin_control(self):
        self.event.zero_()
        self.stop_substep.fill_(-1)

    def set_positions(self,bodies,p):
        wp.launch(part_positions,self.worlds*6,inputs=[bodies,p,self.current],device=self.device)

    def sample(self, groups, query, known, substep, arrivals=None):
        self.counts.zero_()
        wp.launch(raw_contact_counts, len(groups), inputs=[groups, query, self.counts], device=self.device)
        wp.launch(release_motion,self.worlds*6,inputs=[self.counts,known,self.current,self.anchor,self.anchor_valid,CONTRACT['row_match_distance_m'],self.motion_release],device=self.device)
        wp.launch(first_contact, self.worlds * 6, inputs=[self.counts, self.counts if arrivals is None else arrivals, known, self.required,
            self.blocked, self.absent, self.delivered, self.exit_steps, substep,
            self.event, self.stop_substep, self.pending, self.event_parts, self.done, self.lost,self.motion_release], device=self.device)

    def confirm(self, stable, known):
        wp.launch(confirm_contact, self.worlds * 6, inputs=[self.pending, stable, known,
            self.lost, self.confirmation_steps, self.done], device=self.device)
