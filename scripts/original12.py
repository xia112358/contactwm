#!/usr/bin/env python3
"""Prepare and run the complete published original12 Newton 2ms experiment."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from run_workflow import ROOT, _path, build_launch, execute, parse_args


def plan(args, *, root=ROOT, environ=None):
    root = root.resolve()
    recipe = json.loads((root/'configs/experiments/original12_newton2ms.json').read_text())
    values = ['tracking']
    if args.resources:
        values += ['--resources', str(args.resources)]
    options, _ = parse_args(values)
    launch = build_launch(options, [], root=root, environ=environ)
    manifest = root / recipe['task_manifest']
    output = _path(args.output, root) if args.output else launch.runtime_root / 'logs' / ('original12-' + args.command)
    if args.command in ('prepare', 'check'):
        registry = json.loads(launch.resources.read_text())
        inherited = os.environ if environ is None else environ
        native = _path(inherited.get('SOMAFORGE_PPO_RUNTIME') or registry['ppo']['runtime_root'], root)
        commands = []
        if args.command == 'prepare':
            if args.skip_environment:
                commands.append([sys.executable, str(root/'scripts/bootstrap_runtime.py'), '--data-root', str(native)])
            else:
                commands.append([sys.executable, str(root/'scripts/setup_workflow_environment.py'),
                                 '--resources', str(launch.resources)])
        download = [sys.executable, str(root/'scripts/download_data.py')]
        modes = ('fetch', 'verify') if args.command == 'prepare' else ('verify',)
        for mode in modes:
            command = [*download, mode, recipe['release_component'], '--root', str(root),
                       '--catalog', str(root/recipe['release_catalog'])]
            if args.assets:
                command.extend(('--local-archives', str(args.assets.expanduser().absolute())))
            commands.append(command)
        if args.command == 'check':
            commands.append([launch.argv[0], '-m', 'supermimic.original12_check', str(root)])
        return dict(commands=commands, environment=launch.environment, resources=str(launch.resources))
    checkpoint = root / recipe['initialization' if args.resume_state == 'initialization' else 'saved_state']
    if args.command == 'tracking':
        native_args = [*recipe['tracking_args'], '--task_manifest', str(manifest),
                       '--log_dir', str(output), '--resume', str(checkpoint)]
        if args.startup_check:
            native_args.append('--startup_check')
        launch = build_launch(options, native_args, root=root, environ=environ)
    else:
        native_args = ['--run_dir', str(root/recipe['archived_run']), '--ckpt', str(root/recipe['saved_state']),
                       '--task_manifest', str(manifest), '--collision_profile', 'surface_v2',
                       '--contact_response', 'newton_contacts_v2', '--contact_model', 'force6', '--eval_seed', '0',
                       '--result_base', str(output/'checkpoint1500')]
        launch = replace(launch, stage='evaluate', argv=(launch.argv[0], '-m', 'supermimic.evaluate', *native_args))
    return dict(launch=launch, output=output, recipe=recipe, manifest=manifest, checkpoint=checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'tracking', 'evaluate'))
    parser.add_argument('--resources', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-environment', action='store_true', help='Prepare files only when the pinned workflow environment is already installed')
    parser.add_argument('--assets', type=Path, help='Offline directory containing both original12 data and training release assets')
    parser.add_argument('--output', type=Path, help='External output directory (defaults to the configured workflow runtime)')
    parser.add_argument('--resume-state', choices=('initialization', 'latest'), default='initialization',
                        help='Tracking: reproduce the original iteration-1000 start, or resume the saved iteration-1500 state')
    parser.add_argument('--startup-check', action='store_true', help='Tracking: one physics step without optimization')
    args = parser.parse_args(argv)
    if args.startup_check and args.command != 'tracking':
        parser.error('--startup-check applies only to tracking')
    try:
        result = plan(args)
        if 'commands' in result:
            if args.dry_run:
                print(json.dumps(result, indent=2))
            else:
                for command in result['commands']:
                    subprocess.run(command, cwd=ROOT, env=dict(os.environ, **result['environment']), check=True)
            return 0
        launch = result['launch']
        if args.dry_run:
            print(json.dumps(launch.as_dict(), indent=2))
            return 0
        if result['output'].is_relative_to(ROOT):
            raise ValueError('Run outputs must be outside the source checkout')
        # Verify distribution hashes before loading any published pickle/tensor file.
        verify = [sys.executable, str(ROOT/'scripts/download_data.py'), 'verify',
                  result['recipe']['release_component'], '--catalog', str(ROOT/result['recipe']['release_catalog'])]
        if args.assets:
            verify.extend(('--local-archives', str(args.assets.expanduser().absolute())))
        subprocess.run(verify, cwd=ROOT, check=True)
        result['output'].mkdir(parents=True, exist_ok=True)
        execute(launch)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
