"""Content-addressed, integrity-checked CPU preprocessing; no contact fallback."""
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch

from somaforge_core.robot_assets import somaforge_root
ROOT = somaforge_root()
SCHEMA = 'certified_predictor_preprocessing_v1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def cache_identity(manifest_path, frames, build_dataset=None):
    from somaforge_core.contact_dataset import contact_dataset_fingerprint
    from somaforge_core.robot_assets import encode_robot_asset_json
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if not manifest.get('event_contract') or manifest.get('training_ready') is not True:
        raise ValueError('Preprocessing cache requires an accepted event manifest')
    from somaforge_core.newton_contact_data import require_current_newton_manifest
    require_current_newton_manifest(manifest, context=str(path))
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent/p
    extra = {}
    for entry in manifest['motion_files']:
        plan = resolve(entry['edit_plan_file'])
        extra[str(plan)] = file_hash(plan)
        catalog = resolve(json.loads(plan.read_text())['metadata']['target_surface_catalog'])
        extra[str(catalog)] = file_hash(catalog)
    modules = [Path(__file__), Path(__file__).with_name('contact_event_sample.py')]
    modules += list((ROOT/'packages/somaforge_core/somaforge_core').glob('*.py'))
    builder_identity = None
    if build_dataset is not None:
        source = inspect.getsourcefile(build_dataset)
        if source is None:
            raise ValueError('Preprocessing builder requires an inspectable source file')
        modules.append(Path(source).resolve())
        builder_identity = dict(module=build_dataset.__module__, function=build_dataset.__qualname__)
    identity = dict(schema=SCHEMA, frames=int(frames), manifest_sha256=file_hash(path),
                    dataset_fingerprint=contact_dataset_fingerprint(path), external_files=extra,
                    processing_code={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):file_hash(p) for p in sorted(modules)},
                    builder=builder_identity,
                    robot_asset_json=encode_robot_asset_json(),
                    versions=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)))
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return key, identity


def load_or_build_dataset(manifest_path, frames, *, build_dataset, cache_directory=None):
    from .contact_event_sample import ContactEventSample
    started = time.perf_counter()
    key, identity = cache_identity(manifest_path, frames, build_dataset)
    hash_seconds = time.perf_counter()-started
    directory = Path(cache_directory) if cache_directory else ROOT/'runtime/current/cache/predictor_preprocessing'
    directory.mkdir(parents=True,exist_ok=True)
    payload_path, metadata_path = directory/f'{key}.pt', directory/f'{key}.json'
    hit = metadata_path.exists()
    if hit:
        metadata = json.loads(metadata_path.read_text())
        if metadata['identity'] != identity or file_hash(payload_path) != metadata['payload_sha256']:
            raise ValueError('Preprocessing cache integrity mismatch; refusing cached supervision')
        payload = torch.load(payload_path, map_location='cpu', weights_only=False)
        data, summary = payload['data'], payload['summary']
        samples = [ContactEventSample(**row) for row in payload['samples']]
    else:
        data, samples, summary = build_dataset(manifest_path, frames)
        # Rehash after a cold build to catch changes during preprocessing.
        if cache_identity(manifest_path, frames, build_dataset) != (key, identity):
            raise ValueError('Input or processing code changed during preprocessing')
        payload = dict(data=data,samples=[dataclasses.asdict(s) for s in samples],summary=summary)
        # Exclusive publication: a partial cache is never accepted or overwritten.
        with payload_path.open('xb') as stream: torch.save(payload, stream)
        with metadata_path.open('x') as stream:
            json.dump(dict(identity=identity,payload_sha256=file_hash(payload_path)),stream,indent=2)
    report = dict(cache_hit=hit, cache_key=key, hash_seconds=hash_seconds,
                  elapsed_seconds=time.perf_counter()-started, samples=len(samples),
                  dataset_fingerprint=identity['dataset_fingerprint'])
    print(json.dumps(dict(preprocessing=report)),flush=True)
    return data, samples, summary, report
