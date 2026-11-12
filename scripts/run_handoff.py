#!/usr/bin/env python3
"""Run the published ten-stage bottleneck recipe with the configured workflow runtime."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import run_workflow

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--gpu',default='0')
    p.add_argument('--python')
    p.add_argument('--resources',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    recipe=json.loads((ROOT/'configs/contact_learning/handoff-ten-stage.json').read_text())
    assets=args.assets.resolve();output=args.output.resolve()
    values=dict(assets=str(assets),output=str(output),iterations=args.iterations,root=str(ROOT))
    backend=[x.format(**values) for x in recipe['arguments']]
    options=argparse.Namespace(stage='contact',backend=None,python=args.python,resources=args.resources)
    launch=run_workflow.build_launch(options,backend)
    env=dict(launch.environment,CUDA_VISIBLE_DEVICES=args.gpu,SOMAFORGE_PUBLISHED_DATA_PATHS=str(assets/'paths.json'))
    launch=replace(launch,argv=(launch.argv[0],'-u',str(ROOT/'packages/supermimic/supermimic/handoff/entry.py'),*backend),environment=env)
    if args.dry_run:
        print(json.dumps(launch.as_dict(),indent=2));return
    for name in ['paths.json','contact-init.pt','contact-prior.pt','tracking-5k.pt','cases.pt','recording.pt','solver-collision.mjb']:
        if not (assets/name).is_file():p.error(f'Missing {assets/name}; run scripts/prepare_handoff.py first')
    run_workflow.execute(launch)

if __name__=='__main__':main()
