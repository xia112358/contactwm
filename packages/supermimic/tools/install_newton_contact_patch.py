"""Build a project-local Newton runtime; never mutate a shared installation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def install(source: Path, output: Path) -> dict:
    patch_dir = Path(__file__).resolve().parents[1] / 'patches'
    name = 'newton-1.8dev-contact-writer-v4'
    manifest = json.loads((patch_dir / (name + '.json')).read_text())
    for relative, expected in manifest['files'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected['baseline_sha256']:
            raise ValueError(f'Patch requires the recorded baseline: {source / relative}')
    # A fresh destination preserves any already-running or user-managed runtime.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output / 'newton', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', str(output),
                    '-i', str(patch_dir / (name + '.patch'))], check=True)
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((output / 'newton' / relative).read_bytes()).hexdigest() == expected['patched_sha256']
    manifest.update(source=str(source), pythonpath=str(output))
    (output / 'contact-patch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Baseline newton package directory')
    parser.add_argument('--output', required=True, type=Path, help='New parent directory for the patched package')
    args = parser.parse_args()
    source = args.source or Path(importlib.util.find_spec('newton').origin).parent
    result = install(source.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
