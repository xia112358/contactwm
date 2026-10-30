import json

import torch

from generator.support_supervision import prepare_support_supervision
from generator.endpoint_termination import endpoint_failure_reasons


def test_missing_load_labels_are_unknown_not_contact_imputed(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(motion_files=[{}])))
    target = dict(q=torch.zeros(1,36),current_contact=torch.ones(1,6,dtype=torch.bool))
    result = prepare_support_supervision(manifest,target,
        [dict(motion_id=0,current_frame=0,target_frame=1)])
    assert result['unknown_endpoint_samples']==1
    assert not target['reference_target_observed_load_known'].any()
    assert not target['reference_target_observed_load_bearing'].any()
    assert target['reference_target_observed_normal_force_n'].isnan().all()
    assert target['current_contact'].all()


def test_endpoint_contact_gate_does_not_certify_observed_support():
    record = dict(endpoint_contact_retention_valid=True,task_acceptance=dict(accepted=True))
    assert endpoint_failure_reasons(record)==[]
    assert endpoint_failure_reasons(record,require_observed_support=True)==['actual_support_evidence_unknown']
