"""Summarize native penetration and unlabelled body contacts; do not redefine contact."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
results=[]
allowed={'left_ankle_roll_link','right_ankle_roll_link','left_ankle_pitch_link','right_ankle_pitch_link','left_knee_link','right_knee_link','left_sphere_hand_link','right_sphere_hand_link'}
for run in a.runs:
 frames=json.loads((run/'all-body-contacts.json').read_text());cfg=json.loads((run/'contact-response.json').read_text());base=json.loads((run/'diagnosis.json').read_text()) if (run/'diagnosis.json').exists() else {'incomplete':True}
 bodies={}
 for f in frames:
  for c in f['contacts']:
   if c['body'] in allowed:continue
   b=bodies.setdefault(c['body'],dict(frames=set(),loaded_frames=set(),peak_force_N=0.,peak_penetration_mm=0.))
   force=float(np.linalg.norm(c['force']));b['frames'].add(f['step']);b['peak_force_N']=max(b['peak_force_N'],force);b['peak_penetration_mm']=max(b['peak_penetration_mm'],-c['dist']*1000)
   if force>=10:b['loaded_frames'].add(f['step'])
 for b in bodies.values():
  b['frames']=sorted(b['frames']);b['loaded_frames']=sorted(b['loaded_frames']);b['duration_s']=len(b['frames'])*.02
  longest=current=0;previous=-2
  for f in b['loaded_frames']:current=current+1 if f==previous+1 else 1;previous=f;longest=max(longest,current)
  b['max_contiguous_loaded_frames']=longest
 depths=[max([-min(c['dist'],0)*1000 for c in f['contacts'] if 'ankle_roll_link' in c['body']],default=0.) for f in frames]
 def depth_summary(v):
  return dict(frames=len(v),mean_deepest_mm=float(np.mean(v)),p95_deepest_mm=float(np.quantile(v,.95)),max_mm=float(max(v)))
 results.append(dict(path=str(run.resolve()),response=cfg['response'],summary=base,unexpected_bodies=bodies,
    frame_weighted_foot_penetration=depth_summary(depths),matched_first100=depth_summary(depths[:100]),
    unexpected_support_detected=any(b['max_contiguous_loaded_frames']>=3 for b in bodies.values()),
    background_normal_anomalies=json.loads((run/'normal-anomalies.json').read_text()) if (run/'normal-anomalies.json').exists() else []))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(scope='Native constraint contacts only. Force >=10N for >=3 control frames flags sustained unlabelled support; it does not define contact activation.',runs=results),indent=2))
for r in results:print(r['response'],r['summary'], 'unexpected sustained support',r['unexpected_support_detected'])
