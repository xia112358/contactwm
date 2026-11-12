from pathlib import Path
import torch,json,numpy as np,hashlib
from supermimic.contact_release import native_release_history
import argparse
parser=argparse.ArgumentParser(description='Repair shared partition boundaries from aligned native target onsets; preserve fixed goals and population receipts.')
for name in ('recording','slices','partition','native_traces','source_pool','output'):parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();r=args.output;r.mkdir(exist_ok=False)
d=torch.load(args.slices,map_location='cpu',weights_only=False);record=torch.load(args.recording,map_location='cpu',weights_only=False);full=torch.load(args.partition,map_location='cpu',weights_only=False);traces=torch.load(args.native_traces,map_location='cpu',weights_only=False);spans=d['metadata']['recording_spans'];receipts=[];histories=[]
for z in traces:histories.append(native_release_history(z['active'],torch.ones_like(z['active']),z['position'],20))
for i,cid in enumerate(d['case_ids'].tolist()):
 ti=int(d['trace_ids'][i]);span=spans[ti];z=traces[ti];offset=span['begin'];a=int(d['starts'][i])-offset;b=int(d['stops'][i])-offset
 for part in (d['target_masks'][i]&~d['keep_masks'][i]).nonzero().flatten().tolist():
  ticks=z['stable'][a*10:b*10,part].nonzero().flatten()+a*10
  assert len(ticks),(cid,part,'no native stable evidence')
  if part<2:
   points=z['position'][ticks,part,None,:]+torch.einsum('tij,kj->tki',z['rotation'][ticks,part],d['fixed_material_local'][i,part]);error=(points-d['fixed_material_targets'][i,part]).square().sum(-1).mean(-1).sqrt()
  elif part<4:error=(z['point'][ticks,part]-d['fixed_goal_points'][i,part]).norm(dim=-1)
  else:
   valid=d['paired_material_valid'][i,part];points=z['position'][ticks,part,None,:]+torch.einsum('tij,kj->tki',z['rotation'][ticks,part],d['paired_material_local'][i,part,valid]);error=(points-d['paired_material_targets'][i,part,valid]).norm(dim=-1).min(-1).values
  tick=int(ticks[error.argmin()]);onset=int(histories[ti][2][tick,part]);assert onset>=0
  # Start before the solve that first activated the target phase. A tick-0
  # activation needs the preceding control state, never a first-frame exemption.
  latest=onset//10-int(onset%10==0)
  receipts.append(dict(case=cid,part=part,trace=ti,witness_tick=tick,onset_tick=onset,old_start=a,latest_start=latest,error_cm=float(error.min()*100)))
from supermimic.stable_contact_slices import move_partition_before_native_onsets
latest={cid:min(x['latest_start']+spans[x['trace']]['begin'] for x in receipts if x['case']==cid) for cid in d['case_ids'].tolist()}
full,changes=move_partition_before_native_onsets(full,latest)
for i,cid in enumerate(d['case_ids'].tolist()):
 j=int((full['case_ids']==cid).nonzero()[0]);d['starts'][i]=full['starts'][j];d['stops'][i]=full['stops'][j];d['goals'][i]=min(int(d['goals'][i]),int(d['stops'][i])-1)
# Require each preserved target's native stable witness to remain inside its new interval.
violations=[]
for x in receipts:
 i=int((d['case_ids']==x['case']).nonzero()[0]);offset=spans[x['trace']]['begin'];a=int(d['starts'][i])-offset;b=int(d['stops'][i])-offset
 if not a*10<=x['witness_tick']<b*10:
  span=spans[x['trace']];z=traces[x['trace']];part=x['part'];cap=args.source_pool/Path(span['trace']).parent;ct=json.loads((cap/'substeps/contract.json').read_text());w=ct['world_task_ids'].index(span['motion']);origin=torch.tensor(ct['world_origins'][w]);matches=[]
  for frame in range(a,b):
   native=torch.load(cap/'substeps'/f"{frame+span['source_frames'][0]:06d}.pt",weights_only=False)['substeps']
   for sub,row in enumerate(native):
    if not bool(row['features']['contact_stable'][w,part]):continue
    use=row['active']&row['allocated']&row['eligible']&(row['groups']==w*6+part)
    ids=use.nonzero().flatten();side=row['robot_side'][ids];force=row['rows']['force_on_body1_w'][ids,2];ids=ids[torch.where(side==1,force,-force)>0]
    pts=row['rows']['position_w'][ids]-origin
    if part<4:
     err=(pts-d['fixed_goal_points'][i,part]).norm(dim=-1);ok=err<=.05
    else:
     valid=d['paired_material_valid'][i,part];local=(pts-row['part_position'][w,part]+origin)@row['part_rotation'][w,part]
     match=(local[:,None]-d['paired_material_local'][i,part,valid]).norm(dim=-1)<=.01
     error=(pts[:,None]-d['paired_material_targets'][i,part,valid]).norm(dim=-1);ok=(match&(error<=.05)).any(-1);err=error.masked_fill(~match,float('inf')).min(-1).values
    for j in ok.nonzero().flatten().tolist():matches.append((float(err[j]),frame*10+sub,int(ids[j])))
  if matches:
   error,tick,row=min(matches);x['original_witness_tick']=x['witness_tick'];x['witness_tick']=tick;x['error_cm']=error*100;x['native_row']=row
  else:violations.append(x)
print('CHANGES',changes,flush=True);print('WITNESS_OUTSIDE',violations,flush=True)
(r/'boundary-audit.json').write_text(json.dumps(dict(changes=changes,targets=receipts,witness_outside=violations),indent=2))
# Candidate is written only after the partition and target ownership checks pass.
assert not violations,'A boundary would remove a predecessor target witness'
for ti in range(len(spans)):
 ids=(full['trace_ids']==ti).nonzero().flatten();assert torch.equal(full['stops'][ids[:-1]],full['starts'][ids[1:]])
for obj in (d,full):
 floors=torch.tensor([spans[int(t)]['history_begin'] for t in obj['trace_ids']]);obj['history_indices']=torch.maximum(obj['starts'][:,None]+torch.arange(-10,0),floors[:,None]);obj['previous_contact_indices']=torch.maximum(obj['starts']-1,floors)
 obj['metadata']['native_onset_boundary']='target native phase onset precedes reset; adjacent partition boundary shared; fixed targets and aligned population evidence preserved'
record['data']['substep_part_position']=torch.cat([z['position'].reshape(-1,10,6,3) for z in traces])
record['data']['substep_raw_contact_active']=torch.cat([z['active'].reshape(-1,10,6) for z in traces])
record['metadata']['release_motion_contract']=dict(native_absence=True,distance_m=.01,exit_seconds=.04)
torch.save(d,r/'cases.pt');torch.save(full,r/'partition-all.pt');torch.save(record,r/'rollout.pt');print('COMPLETE',len(d['starts']),flush=True)
