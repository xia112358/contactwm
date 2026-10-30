"""Action-level establish/keep/release checks on immutable Newton evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
from somaforge_core.contact_events import MAX_DROPOUT_FRAMES

SCHEMA = 'explicit_action_contact_acceptance_v2'
PARTS = ('left_foot', 'right_foot', 'left_hand', 'right_hand', 'left_knee', 'right_knee')
# Existing stable evidence rule from newton_contact_data.touchdown_events.
STABLE_FRAMES = 3


class SourceEventEvidenceError(ValueError):
    """Expose source-clock diagnostics without weakening prepared-source checks."""
    def __init__(self, report, contract):
        self.report, self.contract = report, contract
        super().__init__(f'Source observations do not satisfy event evidence policy: {report["failures"]}')


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def action_requirements(event):
    source, target = event['source_surfaces'], event['target_surfaces']
    if 'persistent_parts' not in event:
        raise ValueError('Explicit persistent_parts required; endpoints cannot infer support')
    active = {int(t['part_index']) for t in event.get('touchdown_events', [])}
    persistent = {PARTS.index(name) for name in event['persistent_parts']}
    if active & persistent:
        raise ValueError('An endpoint cannot both establish and keep within one action')
    requirements = []
    for p in sorted(persistent):
        if source[p] < 0 or source[p] != target[p]:
            raise ValueError('Explicit support must identify one unchanged surface')
        requirements.append(dict(kind='keep', part=p, surface=int(target[p])))
    for p in sorted(active):
        if target[p] < 0:
            raise ValueError('Touchdown requires an explicit target surface')
        requirements.append(dict(kind='establish', part=p, surface=int(target[p])))
    for release in event.get('release_events', []):
        if release.get('scope') != 'whole_endpoint':
            continue
        p = int(release['part_index'])
        if p in active | persistent:
            raise ValueError('Conflicting explicit release role')
        requirements.append(dict(kind='release', part=p, surface=-1))
    return requirements


def build_contract(mask, surface, events, fps):
    mask, surface = np.asarray(mask, bool), np.asarray(surface)
    if mask.ndim != 2 or mask.shape[1] != 6 or surface.shape != mask.shape:
        raise ValueError('Expected six-part contact evidence')
    if np.any(mask & (surface < 0)) or fps <= 0:
        raise ValueError('Invalid contact surface or fps')
    covered = np.zeros(len(mask), bool); actions=[]; noise=[]
    for index, e in enumerate(events):
        a,b=int(e['start_frame']),int(e['end_frame'])
        if not 0 <= a <= b < len(mask): raise ValueError('Action outside motion')
        covered[a:b+1]=True
        requirements=action_requirements(e)
        actions.append(dict(id=e.get('segment_id',str(index)),start=a,end=b,requirements=requirements))
        # Source provides one tolerance statistic, never a phase template.
        for r in requirements:
            if r['kind'] != 'keep': continue
            p,s=r['part'],r['surface']; raw=mask[a:b+1,p] & (surface[a:b+1,p]==s)
            for left,right in runs(~raw):
                if left > 0 and right < len(raw) and not mask[a+left:a+right,p].any():
                    noise.append(dict(action=index,part=p,start=a+left,stop=a+right,frames=right-left))
    if not actions: raise ValueError('No action requirements')
    tolerance=max([MAX_DROPOUT_FRAMES]+[x['frames'] for x in noise])
    return dict(schema=SCHEMA,frame_count=len(mask),fps=float(fps),retained_blocks=runs(covered),
        actions=actions,policy=dict(max_dropout_frames=tolerance,stable_frames=STABLE_FRAMES,
        tolerance_source='max_bounded_dropout_in_demonstrated_keep_actions',source_dropout_samples=noise,
        truth='unchanged_native_activation_and_allocation',endpoint='whole_part_same_surface'))


def observed_source_contract(data, intervals):
    """Bind roles to immutable source observations, preserving the edit clock.

    Removed standing intervals separate retained blocks. Each block uses the
    same event materializer as predictor demonstrations; evidence cannot cross
    a removed interval. Authored IDs identify edits, never contact requirements.
    """
    from somaforge_core.demonstration_events import materialize
    blocks = []
    frame_count = len(data['contact_part_mask'])
    identities = set()
    for index, row in enumerate(intervals):
        a, b = int(row['start_frame']), int(row['end_frame'])
        identity = row.get('segment_id', str(index))
        if not 0 <= a < b < frame_count or identity in identities:
            raise ValueError('Invalid or duplicate source action clock')
        identities.add(identity)
        if blocks and a < int(blocks[-1][-1]['end_frame']):
            raise ValueError('Overlapping or unordered source action clock')
        if not blocks or a != int(blocks[-1][-1]['end_frame']):
            blocks.append([])
        blocks[-1].append(dict(row, segment_id=identity))
    if not blocks:
        raise ValueError('No source action clock')
    events = []
    for block in blocks:
        start, stop = int(block[0]['start_frame']), int(block[-1]['end_frame']) + 1
        evidence = {key: np.asarray(data[key])[start:stop]
                    for key in ('contact_part_mask', 'contact_surface')}
        evidence['fps'] = data['fps']
        clock = [dict(start_frame=int(row['start_frame'])-start,
                      end_frame=int(row['end_frame'])-start) for row in block]
        observed, _ = materialize(evidence, clock)
        for row, source in zip(observed, block):
            row.update(segment_id=source['segment_id'],
                       start_frame=row['start_frame']+start, end_frame=row['end_frame']+start)
            for touchdown in row['touchdown_events']:
                touchdown['frame'] += start
            events.append(row)
    contract = build_contract(data['contact_part_mask'], data['contact_surface'], events, data['fps'])
    contract['event_binding'] = dict(schema='own_native_source_event_binding_v1',
        materializer='somaforge_core.demonstration_events.materialize',
        old_contact_requirements_used=False, boundary_shifts=0,
        action_count=len(events), authored_ids_used='edit_clock_identity_only')
    source_audit, _ = audit_events(contract, data['contact_part_mask'], data['contact_surface'])
    if not source_audit['passed']:
        raise SourceEventEvidenceError(source_audit, contract)
    contract['source_self_audit'] = source_audit
    return events, contract


def materialize_contract(motion, labels, event_path, support_phases=None):
    from somaforge_core.demonstration_events import observations
    data = observations(motion, labels)
    intervals = [json.loads(line) for line in Path(event_path).read_text().splitlines() if line.strip()]
    _, contract = observed_source_contract(data, intervals)
    contract['provenance'] = {key: dict(path=str(Path(path).resolve()),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        for key, path in [('motion', motion), ('labels', labels), ('events', event_path)]}
    if support_phases is not None:
        phases = json.loads(Path(support_phases).read_text())
        expected = {(r['event_id'], r['start'], r['end'], r['part'], r['surface']) for r in phases}
        actual = {(a['id'], a['start'], a['end'], r['part'], r['surface'])
                  for a in contract['actions'] for r in a['requirements'] if r['kind'] == 'keep'}
        if actual != expected or len(expected) != len(phases):
            raise ValueError('Support roles differ from the authoritative source phases')
        contract['provenance']['support_phases'] = dict(path=str(Path(support_phases).resolve()),
            sha256=hashlib.sha256(Path(support_phases).read_bytes()).hexdigest())
    with np.load(labels, allow_pickle=False) as z:
        contract['source_contact_semantics'] = json.loads(z['contact_semantics_json'].item())
    return contract


def optimization_events(contract):
    """Use action requirements, never source phase counts or exact boundaries."""
    from somaforge_core.newton_contacts import PARTS
    result=[]
    for action in contract['actions']:
        a,b=action['start'],action['end']
        block_stop=next(stop for start,stop in contract['retained_blocks'] if start<=b<stop)
        for r in action['requirements']:
            p,s=r['part'],r['surface']; faces=[-1]*6;faces[p]=s
            stop=min(block_stop,b+contract['policy']['stable_frames'])
            if r['kind']=='release':
                result.append(dict(start_frame=b,end_frame=stop-1,persistent_parts=[],
                    source_surfaces=faces,target_surfaces=faces,touchdown_events=[],
                    release_events=[dict(start_frame=b,end_frame_exclusive=stop,part_index=p,scope='whole_endpoint')]))
            else:
                result.append(dict(start_frame=a if r['kind']=='keep' else b,end_frame=b if r['kind']=='keep' else stop-1,
                    persistent_parts=[PARTS[p]],source_surfaces=faces,target_surfaces=faces,touchdown_events=[]))
    return result


def audit_events(contract, mask, surface):
    mask,surface=np.asarray(mask,bool),np.asarray(surface)
    if contract['schema']!=SCHEMA or mask.shape!=(contract['frame_count'],6) or surface.shape!=mask.shape:
        raise ValueError('Action contract/evidence mismatch')
    if np.any(mask & (surface<0)): raise ValueError('Missing actual contact surface')
    tolerance=contract['policy']['max_dropout_frames']; stable=contract['policy']['stable_frames']
    tolerated=np.zeros_like(mask); checks=[]
    for action in contract['actions']:
        a,b=action['start'],action['end']
        first,stop=next((x,y) for x,y in contract['retained_blocks'] if x<=a<=b<y)
        for r in action['requirements']:
            p,s=r['part'],r['surface']
            raw=mask[first:stop,p] & (surface[first:stop,p]==s)
            if r['kind']=='keep':
                continuous=raw.copy(); observed=np.flatnonzero(raw)
                for left,right in zip(observed[:-1],observed[1:]):
                    if (0 < right-left-1 <= tolerance and not mask[first+left+1:first+right,p].any()):
                        continuous[left+1:right]=True
                        tolerated[first+left+1:first+right,p]=True
                midpoint=(a+b)//2
                left_bound=a if a==first else min(a+tolerance,midpoint)
                right_bound=b if b==stop-1 else max(b-tolerance,midpoint)
                span=continuous[left_bound-first:right_bound-first+1]
                bad=[(x+left_bound,y+left_bound) for x,y in runs(~span)]
                passed=not bad and any(y-x>=stable for x,y in runs(raw[a-first:b-first+1]))
                if passed:
                    tolerated[a:b+1,p] |= ~raw[a-first:b-first+1]
                row=dict(action=action['id'],kind='keep',part=p,surface=s,passed=passed,interruptions=bad)
            else:
                # Completion tolerance plus a short stable observation, no exact touchdown frame.
                lo=max(a,b-tolerance); hi=min(stop,b+tolerance+stable)
                evidence=(~mask[lo:hi,p]) if r['kind']=='release' else (mask[lo:hi,p] & (surface[lo:hi,p]==s))
                stable_runs=[(x+lo,y+lo) for x,y in runs(evidence) if y-x>=stable]
                # Must reach the completion neighborhood, not just touch once early in the action.
                passed=bool(stable_runs)
                if passed:
                    end=min(stop,b+stable)
                    tolerated[b:end,p] |= (mask[b:end,p] if r['kind']=='release'
                        else ~(mask[b:end,p] & (surface[b:end,p]==s)))
                row=dict(action=action['id'],kind=r['kind'],part=p,surface=s,passed=passed,
                         observation_window=[lo,hi],stable_evidence=stable_runs)
            checks.append(row)
    failures=[r for r in checks if not r['passed']]
    return dict(schema=SCHEMA,passed=not failures,check_count=len(checks),failed_checks=len(failures),
        failures=failures,checks=checks,tolerated_dropout_part_frames=np.argwhere(tolerated).tolist()),tolerated
