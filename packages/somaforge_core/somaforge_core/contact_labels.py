"""One versioned task-label policy for offline and streaming observations.

Newton observations are immutable facts. Filtering produces separate task
labels (including held positions), never a claim of solver activation.
Durations are measured in samples; fps is part of every contract.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

SCHEMA = 'somaforge_task_contact_labels_v1'


@dataclass(frozen=True)
class ContactLabelPolicy:
    on_frames: int = 1
    off_frames: int = 1
    surface_frames: int = 1

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()):
            raise ValueError('Contact debounce counts must be positive integers')


DEFAULT_POLICY = ContactLabelPolicy()


def label_contract(fps, policy=DEFAULT_POLICY):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Contact labels require a positive sample rate')
    payload = dict(schema=SCHEMA, fps=float(fps), policy=asdict(policy),
                   timing='causal_confirmation_frame', initialization='first_observation',
                   aggregation='none_requery_aggregated_q', surface='part_representative_face')
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return dict(**payload, fingerprint=hashlib.sha256(encoded.encode()).hexdigest())


def validate_contract(value, fps, policy=DEFAULT_POLICY):
    if value != label_contract(fps, policy):
        raise ValueError('Task contact contract mismatch; rebuild labels/caches with the shared policy')


class ContactLabelStream:
    """One stream; leading shape may contain independent environments/parts.

    Keep this instance across chunks. Reset only at a real episode boundary.
    A new surface is confirmed independently, never silently relabeled as top.
    """
    def __init__(self, *, fps, policy=DEFAULT_POLICY):
        self.contract = label_contract(fps, policy)
        self.policy = policy
        self.surface = None

    def step(self, active, surface, position):
        active, surface, position = np.asarray(active), np.asarray(surface), np.asarray(position)
        if (not np.isin(active, [0, 1]).all() or surface.shape != active.shape
                or position.shape != (*active.shape, 3) or not np.isfinite(position).all()
                or not np.issubdtype(surface.dtype, np.integer)
                or (surface[active.astype(bool)] < 0).any()):
            raise ValueError('Invalid raw contact observation')
        observed = np.where(active.astype(bool), surface, -1)
        if self.surface is None:
            self.surface = observed.copy()
            self.position = np.where(active[..., None], position, 0).copy()
            self.pending = observed.copy()
            self.count = np.zeros_like(surface, dtype=np.int64)
        else:
            if observed.shape != self.surface.shape:
                raise ValueError('Stream environment/part shape changed without episode reset')
            changed = observed != self.surface
            self.count = np.where(changed, np.where(observed == self.pending, self.count+1, 1), 0)
            self.pending = observed.copy()
            required = np.where(observed < 0, self.policy.off_frames,
                                np.where(self.surface < 0, self.policy.on_frames, self.policy.surface_frames))
            accept = changed & (self.count >= required)
            self.surface = np.where(accept, observed, self.surface)
            self.position = np.where(((self.surface == observed) & (observed >= 0))[..., None], position, self.position)
            self.position = np.where((self.surface >= 0)[..., None], self.position, 0)
        return dict(contact_part_mask=self.surface >= 0, contact_surface=self.surface.copy(),
                    contact_position_w=self.position.copy())


def process_contact_sequence(active, surface, position, *, fps, policy=DEFAULT_POLICY):
    if len(active) == 0:
        raise ValueError('Empty contact sequence')
    if len(surface) != len(active) or len(position) != len(active):
        raise ValueError('Contact timeline lengths disagree')
    stream = ContactLabelStream(fps=fps, policy=policy)
    rows = [stream.step(a, s, p) for a, s, p in zip(active, surface, position)]
    result = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    result['contact_label_contract'] = stream.contract
    return result


def process_independent_contacts(active, surface, position):
    """For pose batches with no timeline. Temporal filtering requires a stream."""
    if DEFAULT_POLICY != ContactLabelPolicy():
        raise ValueError('Filtered task contacts require full history or ContactLabelStream; isolated q is insufficient')
    return ContactLabelStream(fps=1).step(active, surface, position)
