"""Keep recorded support evidence separate from edited targets and intentions."""
import hashlib
import json

import numpy as np

from .support_semantics import SUPPORT_ASSESSMENT_SCHEMA, assess_support
from .published_data import resolve_published_path


def reference_only_support(payload, *, reason, retain_force_targets=False):
    """Preserve source audit data, invalidate loads after changing the trajectory.

    Legacy support masks describe a requested role; they are never measured
    support. This mutates only an in-memory output payload, not its source file.
    """
    for key in tuple(payload):
        if key in ('support_part_mask', 'support_force_part_mask'):
            if 'contact_keep_intent_mask' not in payload:
                payload['contact_keep_intent_mask'] = np.asarray(payload[key]).copy()
        if (key.startswith(('observed_support_', 'solver_contact_', 'contact_force_'))
                or key in ('support_part_mask','support_force_part_mask','support_assessment_json')):
            if retain_force_targets and key.startswith('contact_force_'):
                payload.setdefault('source_reference_' + key, np.asarray(payload[key]).copy())
            else:
                value = payload.pop(key)
                payload.setdefault('source_reference_' + key, value)
    if retain_force_targets and 'contact_force_part_w' in payload:
        from .contact_schema import diagnostic_contact_provenance, encode_contact_force_provenance
        payload['contact_force_provenance_json'] = np.asarray(encode_contact_force_provenance(
            diagnostic_contact_provenance(source_backend='motion_edit_reference_target',
                metadata=dict(reason=reason,actual_support_status='unknown'))))
    payload['support_assessment_json'] = np.asarray(json.dumps(dict(
        schema=SUPPORT_ASSESSMENT_SCHEMA, status='unknown', reason=reason,
        source_evidence_scope='audit_reference_only',
        required='fresh actual execution; fixed-pose contact queries do not measure loads')))
    return payload


def load_support_observations(motion_path, evidence_path):
    """Load content-bound same-solve observations for a single executed motion.

    The sidecar explicitly binds source recording and motion. Pose/time mapping
    is producer-verified; edited trajectories require a new physical execution.
    """
    motion_path, evidence_path = resolve_published_path(motion_path), resolve_published_path(evidence_path)
    with np.load(evidence_path, allow_pickle=False) as z:
        meta = json.loads(z['support_assessment_json'].item())
        if meta.get('schema') != SUPPORT_ASSESSMENT_SCHEMA or meta.get('status') != 'observed':
            raise ValueError('Support observations are missing, legacy or reference-only')
        digest = hashlib.sha256(motion_path.read_bytes()).hexdigest()
        if meta.get('motion_sha256') != digest:
            raise ValueError('Support observations do not belong to this trajectory')
        recording = meta.get('recording', {})
        if not recording.get('path') or not recording.get('sha256') or not meta.get('sampling'):
            raise ValueError('Support observation provenance or solve sampling missing')
        if hashlib.sha256(resolve_published_path(recording['path']).read_bytes()).hexdigest() != recording['sha256']:
            raise ValueError('Stale original support recording')
        from .robot_assets import validate_g1_asset_metadata
        validate_g1_asset_metadata(meta.get('robot_asset'), context=str(evidence_path))
        from .newton_contact_data import require_current_newton_manifest
        require_current_newton_manifest(meta, context=str(evidence_path))
        values = assess_support(z['contact_activated'], z['normal_force_n'],
                                z['rms_tangent_speed_m_s'], load_known=z['load_known'])
        if values['contact_activated'].ndim != 2 or values['contact_activated'].shape[1] != 6:
            raise ValueError('Expected observed support [frames, six endpoint parts]')
        values['provenance'] = meta
        return values
