"""Write versioned task labels from verified raw labels without changing motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from somaforge_core.newton_contact_data import load_contact_labels


def materialize(motion, raw_labels, output, consensus_evidence=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    labels = load_contact_labels(motion, raw_labels, rebuild_task_cache=True,
                                 consensus_evidence=consensus_evidence)
    with np.load(raw_labels, allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    arrays['contact_label_contract_json'] = np.array(json.dumps(labels['contact_label_contract']))
    if consensus_evidence is not None:
        arrays['contact_consensus_evidence_json'] = np.array(json.dumps(consensus_evidence))
    if 'contact_consensus_report' in labels:
        arrays['contact_consensus_report_json'] = np.array(json.dumps(labels['contact_consensus_report']))
    for key in ('contact_part_mask', 'contact_surface', 'contact_position_w'):
        arrays['task_'+key] = labels[key]
    for key in ('contact_pairs', 'abnormal_contact_pairs'):
        arrays['task_'+key+'_json'] = np.array(json.dumps(labels[key]))
    from somaforge_core.newton_contact_data import touchdown_events
    events = touchdown_events(labels['contact_part_mask'], surfaces=labels['contact_surface'])
    for key in ('touchdown_frames', 'touchdown_parts'):
        if key in arrays and 'legacy_'+key not in arrays:
            arrays['legacy_'+key] = arrays[key].copy()
    arrays['touchdown_frames'] = np.asarray([f for f, _ in events], np.int64)
    arrays['touchdown_parts'] = np.asarray([p for _, p in events], bool).reshape(-1, 6)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **arrays)
    load_contact_labels(motion, output)
    return dict(output=str(output), frames=len(labels['contact_part_mask']),
                contract=labels['contact_label_contract'], motion_modified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--raw-labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(materialize(a.motion,a.raw_labels,a.output),indent=2))


if __name__ == '__main__':
    main()
