"""Independent observed support metadata; contact roles remain intentions."""
import hashlib

import numpy as np
import torch

from somaforge_core.support_evidence import load_support_observations
from somaforge_core.support_semantics import SUPPORT_ASSESSMENT_SCHEMA
from somaforge_core.loaded_material_motion import unknown_material_path
from somaforge_core.published_data import load_published_json, resolve_published_path


def prepare_support_supervision(manifest_path, target, samples):
    manifest_path = resolve_published_path(manifest_path)
    entries = load_published_json(manifest_path)['motion_files']
    size = (len(samples), 6)
    arrays = {}
    for end in ('start', 'target'):
        for key in ('contact_activated','load_known','load_bearing','loaded_motion_known'):
            arrays[f'reference_{end}_observed_{key}'] = np.zeros(size, bool)
        for key in ('normal_force_n','loaded_tangent_speed_m_s'):
            arrays[f'reference_{end}_observed_{key}'] = np.full(size, np.nan, np.float32)
    grouped = {}
    for i, sample in enumerate(samples):
        motion_id = sample['motion_id'] if isinstance(sample, dict) else sample.motion_id
        grouped.setdefault(int(motion_id), []).append(i)
    def resolve(value):
        return resolve_published_path(value, manifest_path.parent)
    for motion_id, indices in grouped.items():
        entry = entries[motion_id]
        if 'support_observations_file' not in entry:
            continue  # Explicit unknown, never a continuous-contact surrogate.
        evidence_path = resolve(entry['support_observations_file'])
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != entry.get('support_observations_sha256'):
            raise ValueError('Missing or stale support observation declaration')
        observed = load_support_observations(resolve(entry['motion_file']), evidence_path)
        with np.load(resolve(entry['motion_file']), allow_pickle=False) as motion:
            if len(observed['load_known']) != len(motion['joint_pos']):
                raise ValueError('Observed support/motion clock mismatch')
        for i in indices:
            sample = samples[i]
            for end, frame_key in (('start','current_frame'),('target','target_frame')):
                frame = int(sample[frame_key] if isinstance(sample,dict) else getattr(sample,frame_key))
                if frame < 0 or frame >= len(observed['load_known']):
                    raise ValueError('Support supervision sample outside original clock')
                for key in ('contact_activated','load_known','load_bearing','loaded_motion_known',
                            'normal_force_n','loaded_tangent_speed_m_s'):
                    arrays[f'reference_{end}_observed_{key}'][i] = observed[key][frame]
    target.update({key:torch.as_tensor(value,device=target['q'].device) for key,value in arrays.items()})
    return dict(schema=SUPPORT_ASSESSMENT_SCHEMA,
        role_semantics='contact establish/keep/release intent, not observed support',
        observed_endpoint_samples=int(arrays['reference_target_observed_load_known'].all(-1).sum()),
        unknown_endpoint_samples=int((~arrays['reference_target_observed_load_known'].all(-1)).sum()),
        scope='same-solve observations; separate sampling from endpoint pose and contact query',
        predicted_support_status='unknown_until_actual_execution',
        predicted_material_motion=unknown_material_path())
