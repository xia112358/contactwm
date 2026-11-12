#!/usr/bin/env python3
"""Download and hash-check the exact inputs of the published handoff recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def download(url,path,digest):
    if not path.exists():
        with urllib.request.urlopen(url) as src,path.open('wb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError(f'SHA256 mismatch: {path}; remove the incomplete file before retrying')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);args=p.parse_args()
    assets=args.assets.resolve();assets.mkdir(parents=True,exist_ok=True)
    recipe=json.loads((ROOT/'configs/contact_learning/handoff-ten-stage.json').read_text())
    archive=assets/'handoff-data.tar.gz';download(recipe['bundle']['url'],archive,recipe['bundle']['sha256'])
    with tarfile.open(archive) as t:
        # Downloaded archive is a security boundary; never extract traversal/symlinks.
        for member in t.getmembers():
            if not (assets/member.name).resolve().is_relative_to(assets) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive entry: {member.name}')
        t.extractall(assets)
    models=json.loads((ROOT/'docs/core-starting-checkpoints-20261112.json').read_text())
    for row in models['files']:
        if row['asset'] in ('tracking-5k.pt','contact-init.pt','contact-prior.pt'):
            download('https://github.com/xia112358/contactwm/releases/download/core-starting-checkpoints-20261112-v1/'+row['asset'],assets/row['asset'],row['sha256'])
    template=json.loads((assets/'paths-template.json').read_text())
    rules=dict(schema='somaforge_published_data_paths_v1',files=[dict(source=x['source'],target=str(assets/x['target'])) for x in template['files']],prefixes=[dict(source=template['motion_source'],target=str(assets/'motions')),*[dict(source=x,target=str(ROOT)) for x in template['repository_sources']]])
    (assets/'paths.json').write_text(json.dumps(rules,indent=2)+'\n')
    print(f'Prepared and verified: {assets}')

if __name__=='__main__':main()
