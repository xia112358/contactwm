"""Audit every extracted case and attach native restart results without dropping failures."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); folder=a.library.absolute(); run=a.run.absolute()
    if (folder/'audit.json').exists():raise FileExistsError('Finalized output is immutable; use a fresh output')
    catalog=json.loads((folder/'catalogue.json').read_text())
    report=json.loads((a.validation/'report.json').read_text())
    config=json.loads((a.validation/'config.json').read_text())
    if Path(config['library']).resolve()!=folder.resolve():raise ValueError('Wrong validation library')
    recording=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=False)
    packed=torch.load(folder/'cases.pt',map_location='cpu',weights_only=False)
    data,meta=recording['data'],recording['metadata']; cases=catalog['cases']
    selection=json.loads(Path(meta['selection_manifest']).read_text())
    spans={s['motion']:s for s in meta['recording_spans']}
    assert set(spans)=={t['source_task_index'] for t in selection['tasks']}
    assert sha(meta['selection_manifest'])==meta['selection_manifest_sha256']
    assert sha(meta['source_checkpoint'])==meta['source_checkpoint_sha256']
    checkpoint=torch.load(meta['source_checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['iteration']==10000
    assert len(cases)==len(report['cases'])==len(packed['starts'])==report['total']
    assert report['max_reset_state_error']<1e-3 and report['max_restart_observation_error']<1e-3
    for span in spans.values():
        assert sha(span['source'])==span['source_sha256']
        b,e=span['begin'],span['end']
        assert not bool(data['term'][b:e].any())
        assert bool((data['f'][b+1:e]-data['f'][b:e-1]==1).all())
        assert all(bool(torch.isfinite(data[k][b:e]).all()) for k in ('s','s1','a','force','point'))
    for i,(case,result) in enumerate(zip(cases,report['cases'])):
        b,e,g=(int(packed[k][i]) for k in ('starts','stops','goals')); span=spans[case['motion']]
        assert case['case']==result['case']==i+1 and case['motion']==result['motion']
        assert span['begin']<=b<e<=span['end'] and span['begin']<=g<span['end']
        assert case['source_frames']==[b-span['begin'],e-span['begin']] and case['frames']==e-b
        assert g-span['begin']==case['original_goal_source_frame']==case['original_source_frames'][1]-1
        assert int(data['f'][b])==case['reference_start'] and int(data['f'][g])+1==case['reference_goal']
        assert bool((data['xi'][b:e,33].long()==case['motion']).all())
        assert bool((packed['history_indices'][i]>=span['begin']).all()) and bool((packed['history_indices'][i]<=b).all())
        assert span['begin']<=int(packed['previous_contact_indices'][i])<=b
        case['native_restart_passed']=result['native_restart_passed']
        case['validation']=result
    passed=torch.tensor([c['native_restart_passed'] for c in cases])
    assert int(passed.sum())==report['passed']
    packed['case_ids']=torch.arange(1,len(cases)+1)
    packed['native_restart_passed']=passed
    packed['metadata']['native_restart_report']=str((a.validation/'report.json').absolute())
    torch.save(packed,folder/'cases.pt')
    for name,mask in [('validated',passed),('review',~passed)]:
        subset={k:(v[mask] if isinstance(v,torch.Tensor) and v.shape[:1]==(len(cases),) else v) for k,v in packed.items()}
        torch.save(subset,folder/f'cases_{name}.pt')
        selected=[c for c,keep in zip(cases,mask.tolist()) if keep]
        (folder/f'{name}_cases.json').write_text(json.dumps(dict(count=len(selected),case_ids=[c['case'] for c in selected],
            motion_count=len({c['motion'] for c in selected}),cases=selected),indent=2)+'\n')
    catalog['native_restart_validation']=dict(report=str((a.validation/'report.json').absolute()),
        passed=report['passed'],failed=report['failed'],total=report['total'],criteria=config)
    (folder/'catalogue.json').write_text(json.dumps(catalog,indent=2)+'\n')
    for name in ('solver-collision.mjb','solver-collision.mjb.json'):
        if not (folder/name).exists():shutil.copyfile(run/'capture_00'/name,folder/name)
    collision=json.loads((folder/'solver-collision.mjb.json').read_text())
    assert sha(folder/'solver-collision.mjb')==collision['model_sha256']
    assert collision['colliders']==dict(mesh=29,sphere=0,cylinder=4,other=0)
    root=runtime_root()
    source_files=[*sorted((root/'supermimic').glob('*.py')),
                  *[root/'tools'/f for f in ('prepare_php_low_speed.py','extract_low_speed_contacts.py',
                    'validate_contact_slices.py','run_low_speed_extraction.py','finalize_contact_slices.py','run_native.sh')]]
    snapshot=run/'final_source';snapshot.mkdir(exist_ok=False)
    for source in source_files:
        target=snapshot/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    audit=dict(status='passed',checkpoint_iteration=checkpoint['iteration'],checkpoint_sha256=meta['source_checkpoint_sha256'],
        motions=len(spans),cases=len(cases),validated_cases=int(passed.sum()),review_cases=int((~passed).sum()),
        validated_motion_coverage=len({c['motion'] for c in cases if c['native_restart_passed']}),
        groups=dict(Counter(c['group'] for c in cases)),colliders=collision['colliders'],
        retained_frames=sum(c['frames'] for c in cases),removed_boundary_wait_frames=catalog['removed_wait_frames'],
        excluded_wait_cases=len(catalog['excluded_wait_cases']),
        checks=['every source fingerprint and complete native trajectory','all 109 selected original task identities',
                'every slice range, unchanged original goal and reference clock','preceding history stays inside own source',
                'matching terrain identity for every retained frame','all cases have matching native restart results',
                'saved solver model has no sphere colliders'],
        source_sha256={str(p.relative_to(root)):sha(p) for p in source_files},
        artifact_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    status=dict(status='complete',stage='extracted_segmented_and_audited',complete_motions=len(spans),
        cases=len(cases),validated_cases=int(passed.sum()),review_cases=int((~passed).sum()),
        library=str(folder),audit=str(folder/'audit.json'),validation=str((a.validation/'report.json').absolute()),
        updated_at=time.time(),training_started=False)
    (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
