"""Replace motion references while preserving the checkpoint's physical task slots."""
import hashlib
from pathlib import Path
from .task_manifest import load_task_manifest


def replace_motion_references(spec, manifest):
    references=load_task_manifest(manifest)
    fields=('terrain_usd_sha256','source_terrain_sha256','boxes')
    original=[[t[k] for k in fields] for t in spec['tasks']]
    requested=[[t[k] for k in fields] for t in references['tasks']]
    if original!=requested or references['robot_asset']!=spec['robot_asset']:
        raise ValueError('Reference replacement must preserve task order, robot and terrain')
    for task,reference in zip(spec['tasks'],references['tasks']):
        path=Path(reference['csv'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['csv_sha256']:
            raise ValueError('Replacement reference CSV changed')
        task['csv']=str(path)
        task['name']=reference['name']
    spec['motion_dir']=references['motion_dir']
    return dict(manifest=str(Path(manifest).resolve()),sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                reference_tasks=[dict(name=t['name'],csv_sha256=t['csv_sha256']) for t in references['tasks']])
