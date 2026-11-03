"""Exclude long validated contact slices without truncating or changing their goals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--max_frames',type=int,default=60)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.max_frames<1:raise ValueError('Positive frame limit required')
    if a.output.exists():raise FileExistsError(a.output)
    source=a.library/'cases_validated.pt'
    packed=torch.load(source,map_location='cpu',weights_only=False)
    catalog=json.loads((a.library/'validated_cases.json').read_text())['cases']
    ids=packed['case_ids'].tolist()
    assert ids==[c['case'] for c in catalog]
    lengths=packed['stops']-packed['starts'];keep=lengths<=a.max_frames
    assert bool(packed['native_restart_passed'].all())
    selected=[c for c,k in zip(catalog,keep.tolist()) if k]
    excluded=[dict(c,exclusion_reason=f'demonstrated duration exceeds {a.max_frames} frames') for c,k in zip(catalog,keep.tolist()) if not k]
    if not selected:raise ValueError('No cases remain')
    child={k:(v[keep].clone() if isinstance(v,torch.Tensor) and v.shape[:1]==(len(ids),) else v) for k,v in packed.items()}
    selection=dict(schema='native_contact_duration_selection_v1',parent_slices=str(source.resolve()),
        parent_slices_sha256=sha(source),max_demonstrated_frames=a.max_frames,dt=packed['metadata']['dt'],
        rule='exclude whole overlong cases; never truncate an action or redefine its original goal',
        retained=len(selected),excluded=len(excluded),source_motions=len({c['motion'] for c in selected}),
        missing_source_motions=sorted({c['motion'] for c in catalog}-{c['motion'] for c in selected}),
        groups=dict(Counter(c['group'] for c in selected)),retained_case_ids=child['case_ids'].tolist(),
        excluded_case_ids=[c['case'] for c in excluded],training_restarted=False,
        deadline_note='This filters demonstrated lengths only; the previous +10-frame grace is a separate training setting.')
    child['duration_selection']=selection
    for c,b,e,g in zip(selected,child['starts'].tolist(),child['stops'].tolist(),child['goals'].tolist()):
        assert e-b==c['frames']<=a.max_frames and [b,e]==c['pooled_frames']
        original=catalog[ids.index(c['case'])]
        assert c==original and g==int(packed['goals'][ids.index(c['case'])])
    a.output.mkdir(parents=True)
    torch.save(child,a.output/'cases_validated.pt')
    (a.output/'selected_cases.json').write_text(json.dumps(dict(count=len(selected),cases=selected),indent=2)+'\n')
    (a.output/'excluded_cases.json').write_text(json.dumps(dict(count=len(excluded),cases=excluded),indent=2)+'\n')
    selection['output_sha256']=sha(a.output/'cases_validated.pt')
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps({k:selection[k] for k in ('retained','excluded','source_motions','missing_source_motions','groups','max_demonstrated_frames')},indent=2))


if __name__=='__main__':main()
