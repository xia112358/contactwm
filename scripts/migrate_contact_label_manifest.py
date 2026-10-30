"""Materialize available audited task labels; report every missing motion.

The output is an inspection manifest, not automatic training acceptance.
"""
import argparse
import json
from pathlib import Path
from materialize_task_contact_labels import materialize
from somaforge_core.contact_dataset import contact_dataset_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--audit-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(a.manifest.read_text())
    accepted, excluded, contracts = [], [], []
    for entry in original['motion_files']:
        ident = entry['motion_id']
        raw = a.audit_directory/f'motion_{ident:03d}'/'contacts.npz'
        target = a.output/'labels'/f'motion_{ident:03d}.npz'
        try:
            result = materialize(entry['motion_file'], raw, target)
            accepted.append(dict(entry, newton_contact_file=str(target.resolve())))
            contracts.append(result['contract'])
        except Exception as exc:
            excluded.append(dict(motion_id=ident,reason=f'{type(exc).__name__}: {exc}'))
    manifest = dict(original, motion_files=accepted, training_ready=False,
        contact_migration=dict(source_manifest=str(a.manifest.resolve()),
            source_count=len(original['motion_files']),available=len(accepted),excluded=excluded,
            note='Label processing migrated; source layer, target realization and cache acceptance still required.'))
    output = a.output/'inspection_manifest.json'
    with output.open('x') as f:
        json.dump(manifest,f,indent=2)
    report = dict(available=len(accepted),excluded=excluded,
        contract_fingerprints=sorted({c['fingerprint'] for c in contracts}),
        contact_dataset_fingerprint=contact_dataset_fingerprint(output))
    with (a.output/'report.json').open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(dict(available=len(accepted),excluded=len(excluded),report=str(a.output/'report.json'))))


if __name__ == '__main__':
    main()
