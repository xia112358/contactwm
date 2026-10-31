#!/usr/bin/env python3
"""Check a fresh ContactWM checkout without starting training or GPU simulation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

from run_workflow import ROOT, build_launch, parse_args

MODULE_CHECK = '''
import importlib.metadata as md, importlib.util, json
modules = ('torch', 'numpy', 'scipy', 'warp', 'newton', 'mujoco', 'mujoco_warp',
           'pxr', 'isaaclab', 'isaaclab_newton', 'somaforge_core', 'supermimic',
           'holosoma', 'tyro', 'loguru', 'wandb')
missing = [name for name in modules if importlib.util.find_spec(name) is None]
versions = {name: md.version(name) for name in ('torch', 'numpy', 'warp-lang', 'newton', 'mujoco', 'mujoco-warp')
            if importlib.util.find_spec({'warp-lang': 'warp', 'mujoco-warp': 'mujoco_warp'}.get(name, name)) is not None}
print(json.dumps(dict(missing_modules=missing, versions=versions)))
raise SystemExit(bool(missing))
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--source-only', action='store_true', help='Check source assets and launch plans without an installed simulator')
    parser.add_argument('--check-cli', action='store_true', help='Also run the four workflow stages with --help (no simulation)')
    args = parser.parse_args(argv)
    if args.source_only and args.check_cli:
        parser.error('--source-only cannot be combined with --check-cli')
    errors, plans = [], []
    try:
        for stage in ('tracking', 'collect', 'initialize', 'contact', 'predictor'):
            values = [stage, '--dry-run']
            if args.resources:
                values.extend(('--resources', str(args.resources)))
            options, _ = parse_args(values)
            plan = build_launch(options, ['--help'])
            plans.append(plan)
        contract_path = ROOT / 'packages/somaforge_core/somaforge_core/robot_assets.py'
        spec = importlib.util.spec_from_file_location('_contactwm_source_check', contract_path)
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        os.environ['SOMAFORGE_ROOT'] = str(ROOT)
        contract.canonical_g1_source_metadata()
        bundle = ROOT / 'assets/g1_spherehand_usd' / contract.G1_SPHEREHAND_USD_RELATIVE.name
        if contract._usd_bundle_sha256(bundle) != contract.G1_SPHEREHAND_USD_BUNDLE_SHA256:
            errors.append('Canonical USD bundle is missing or has a different hash')
    except (OSError, ValueError, KeyError) as error:
        errors.append(str(error))
    report = dict(scope='Source assets and launch routing; no GPU, training, datasets or checkpoint validation',
                  resources=str(plans[0].resources) if plans else None, errors=errors)
    if not args.source_only and plans:
        plan = plans[0]
        env = dict(os.environ, **plan.environment)
        try:
            result = subprocess.run([plan.argv[0], '-c', MODULE_CHECK], cwd=ROOT, env=env,
                                    text=True, capture_output=True, timeout=60)
            if result.returncode:
                errors.append('Workflow dependencies are incomplete. Run python3 scripts/setup_workflow_environment.py')
            if result.stdout.strip():
                report['dependencies'] = json.loads(result.stdout)
            if result.stderr.strip():
                report['dependency_stderr'] = result.stderr.strip()
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            errors.append(f'{error}; run python3 scripts/setup_workflow_environment.py')
        if args.check_cli and not errors:
            report['cli'] = {}
            for plan in plans[:4]:
                try:
                    result = subprocess.run(plan.argv, env=dict(os.environ, **plan.environment), cwd=ROOT,
                                            text=True, capture_output=True, timeout=60)
                    report['cli'][plan.stage] = dict(returncode=result.returncode)
                    if result.returncode:
                        errors.append(f'{plan.stage} --help failed: {result.stderr[-2000:]}')
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append(f'{plan.stage}: {error}')
    report['ok'] = not errors
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
