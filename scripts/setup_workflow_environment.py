#!/usr/bin/env python3
"""Build an additive workflow environment without changing the simulation base."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', type=Path, help='Existing compatible SuperMimic simulation interpreter')
    parser.add_argument('--prefix', type=Path, default=ROOT/'runtime/environments/venv/workflow')
    args = parser.parse_args()
    resources = json.loads((ROOT/'configs/workflow_resources.json').read_text())
    base = args.base_python or Path(resources['supermimic']['base_python'])
    if not base.is_file():
        parser.error(f'Base simulation interpreter is missing: {base}; provide --base-python PATH')
    uv = shutil.which('uv')
    if uv is None:
        parser.error('uv must be installed and available on PATH')
    prefix = args.prefix.expanduser().absolute()
    if prefix.resolve() == base.resolve().parent.parent:
        parser.error('Overlay prefix must differ from the base simulation environment')
    if (prefix/'pyvenv.cfg').is_file() and 'include-system-site-packages = true' not in (prefix/'pyvenv.cfg').read_text().lower():
        parser.error('Existing prefix must be a venv with system-site-packages enabled')
    cache = Path(resources['supermimic']['runtime_root'])/'cache/uv-workflow'
    cache.mkdir(parents=True, exist_ok=True)
    def run(*command):
        subprocess.run([uv, '--cache-dir', str(cache), *map(str, command)], cwd=ROOT, check=True)
    if not (prefix/'pyvenv.cfg').is_file():
        run('venv', '--system-site-packages', '--python', base, prefix)
    python = prefix/'bin/python'
    run('pip', 'install', '--python', python, '--link-mode', 'copy', '-r', ROOT/'configs/environments/workflow-extras.lock.txt')
    packages = ['somaforge_core', 'contact_solver', 'generator', 'motion_edit', 'supermimic']
    editable = [item for name in packages for item in ('-e', str(ROOT/'packages'/name))]
    run('pip', 'install', '--python', python, '--link-mode', 'copy', '--no-deps', *editable, '-e', ROOT/'src/holosoma')
    print(f'Workflow interpreter: {python}')
    print('Set supermimic.python and ppo.python in configs/workflow_resources.json when using a custom prefix.')


if __name__ == '__main__':
    main()
