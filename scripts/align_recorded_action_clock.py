"""Align an inherited action clock to stable events in a fresh native recording.

Only action identity and order are inherited. Contact requirements come from
fresh observations; poses, labels and frame sampling are never changed. This
is source-clock construction, not candidate acceptance or sample filtering.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from somaforge_core.event_acceptance import (
    observed_source_contract, SourceEventEvidenceError)
from somaforge_core.demonstration_events import observations


def align(motion, labels, intervals):
    data=observations(motion,labels)
    rows=deepcopy(intervals)
    mask,faces=data['contact_part_mask'],data['contact_surface']
    changes=[]
    for _ in range(len(rows)*2):
        try:
            events,contract=observed_source_contract(data,rows)
            return rows,dict(schema='native_source_stable_action_clock_alignment_v1',
                shifts=changes,action_count=len(rows),omitted_action_count=0,
                source_pose_modified=False,contact_labels_modified=False,
                source_self_audit=contract['source_self_audit'],
                inherited_semantics='action identity and ordering only')
        except SourceEventEvidenceError as error:
            failures=error.report['failures']
            policy=error.contract['policy']
        identities={r['segment_id']:i for i,r in enumerate(rows)}
        changed=False
        for identity in dict.fromkeys(f['action'] for f in failures):
            i=identities[identity];row=rows[i];a,b=row['start_frame'],row['end_frame']
            if i+1>=len(rows) or rows[i+1]['start_frame']!=b:
                raise ValueError('Stable source event cannot be aligned across an explicitly removed span or terminal boundary')
            upper=rows[i+1]['end_frame']
            failed=[f for f in failures if f['action']==identity]
            if any(f['kind']!='establish' for f in failed):
                raise ValueError('Source clock alignment cannot reinterpret keep/release evidence')
            stable=int(policy['stable_frames'])
            candidates=[]
            for end in range(a+stable,upper):
                if all((mask[end-stable+1:end+1,f['part']] &
                        (faces[end-stable+1:end+1,f['part']]==f['surface'])).all() for f in failed):
                    candidates.append(end)
            if not candidates:
                raise ValueError('No stable same-face establishment in the adjacent native action span')
            end=min(candidates,key=lambda x:(abs(x-b),x))
            if end==b:
                raise ValueError('Source action alignment made no progress')
            changes.append(dict(action=identity,old_end_frame=b,new_end_frame=end,
                next_action=rows[i+1]['segment_id'],evidence_parts=[f['part'] for f in failed],
                stable_frames=stable))
            row['end_frame']=end;rows[i+1]['start_frame']=end
            changed=True
        if not changed:
            raise ValueError('Native source action clock could not be aligned')
    raise ValueError('Native source action alignment did not converge')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('motion','labels','intervals','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.intervals.read_text().splitlines() if line.strip()]
    rows,report=align(args.motion,args.labels,rows)
    report['inputs']={k:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for k,p in (('motion',args.motion),('labels',args.labels),('intervals',args.intervals))}
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    args.output.with_suffix('.alignment.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
