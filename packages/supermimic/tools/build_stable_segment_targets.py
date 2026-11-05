"""Export fixed native targets aggregated within successful support placements."""
import argparse,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    r=torch.load(a.library/'rollout.pt',map_location='cpu',weights_only=False);x=torch.load(a.library/'cases.pt',map_location='cpu',weights_only=False)
    cat=json.loads((a.library/'catalogue.json').read_text());points=r['data']['point'][x['goals']].clone();accepted=[];rejected=[];proof=[]
    for i,c in enumerate(cat['cases']):
        if not c['collector_success']:raise ValueError('Failed source bypassed successful capture pool')
        new=x['target_masks'][i]&~x['keep_masks'][i]
        if not new.any():rejected.append(dict(case=c['case'],reason='no_next_contact_target',kind=c['kind']));continue
        sp=r['metadata']['recording_spans'][c['trace_id']];missing=[];case_proof=[]
        for part in new.nonzero().flatten().tolist():
            arrival=max(t for t,p in zip(c['arrival_frames'],c['arrival_parts']) if p==part)
            ep=next(e for e in sp['placement_target_evidence'] if e['part']==part and e['start']<=arrival<e['stop'])
            if not ep['samples']:missing.append(part);continue
            points[i,part]=torch.tensor(ep['point']);case_proof.append(dict(case=c['case'],motion=c['name'],source=ep['witness_source'],source_sha256=ep['witness_source_sha256'],**ep))
        if missing:rejected.append(dict(case=c['case'],reason='no_reliable_fixed_target',parts=missing));continue
        accepted.append(i);proof+=case_proof
    selected=torch.tensor(accepted,dtype=torch.long)
    packed={k:v[selected] if isinstance(v,torch.Tensor) and v.ndim and len(v)==len(cat['cases']) else v for k,v in x.items()}
    packed['fixed_goal_points']=points[selected]
    packed['metadata']=dict(x['metadata'],fixed_goal_contract=dict(version='successful_population_contact_targets_v4',target='shared equal_demo_patch_cloud medoid of matched stable native support evidence from multiple successful executions',completion='unchanged accumulated_target',loss_sampling='latest qualified stable student sample; missing retains original terminal surface guidance'))
    unresolved=[v for v in rejected if v['reason']=='no_reliable_fixed_target']
    report=dict(partition_cases=len(cat['cases']),contact_training_cases=len(accepted),source_success_required=True,targets=proof,excluded_training_cases=rejected,ready=not unresolved,raw_points_changed=False,partition_boundaries_changed=False)
    (a.output/'target-audit.json').write_text(json.dumps(report,indent=2))
    if unresolved:raise ValueError('Unresolved target quality; audit saved, training artifact not published')
    torch.save(packed,a.output/'cases.pt');print(json.dumps({k:v for k,v in report.items() if k not in ['targets','excluded_training_cases']}))


if __name__=='__main__':main()
