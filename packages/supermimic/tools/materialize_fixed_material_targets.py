"""Upgrade already event-aligned population targets without changing the partition."""
import argparse,json
from pathlib import Path
import torch
from supermimic.contact_material_targets import add_material_targets

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    x=torch.load(a.targets/'cases.pt',map_location='cpu',weights_only=False)
    report=json.loads((a.targets/'target-audit.json').read_text())
    packed,audit=add_material_targets(x,report['targets'])
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(packed,a.output/'cases.pt');(a.output/'material-target-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(cases=len(x['case_ids']),targets=len(audit),output=str(a.output))))
if __name__=='__main__':main()
