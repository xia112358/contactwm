"""Reprocess retained executions; no new policy rollout or training."""
import argparse
import fcntl
from pathlib import Path

from somaforge_core import trajectory_refinement as workflow


from somaforge_core.rollout_reaggregation import reaggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--source-collection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case-keys', nargs='+')
    parser.add_argument('--failed-manifest', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--query-worlds', type=int, default=32)
    parser.add_argument('--case-workers', type=int, default=4, help='Concurrent cases sharing one synchronized Newton model')
    parser.add_argument('--reuse-results', type=Path, action='append', help='Prioritized prior outputs; repeat to retain finished work and older source evidence')
    parser.add_argument('--reuse-manifest', type=Path, help='Import indexed historical result evidence for a fresh audit')
    parser.add_argument('--rebuild', action='store_true', help='Exercise the current generation algorithm for every selected case; retain old evidence as audited candidates')
    parser.add_argument('--reassess-only', action='store_true', help='CPU audit only; leave pending cases for a later invocation')
    args = parser.parse_args()
    plan = workflow.read_json(args.training_plan)
    if args.case_keys and args.failed_manifest: parser.error('Choose explicit cases or failed manifest')
    keys = args.case_keys
    if args.failed_manifest:
        manifest = workflow.read_json(args.failed_manifest)
        failed = {r['reference_motion_file'] for r in manifest['motion_files'] if not r['task_acceptance']['passed']}
        keys = [c['key'] for c in plan['cases'] if c['reference']['motion_file'] in failed]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'controller.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reaggregate(plan, args.source_collection, args.output_dir, keys=keys, device=args.device, query_worlds=args.query_worlds, case_workers=args.case_workers,
                    reuse_results=args.reuse_results, reuse_manifest=args.reuse_manifest, reassess_only=args.reassess_only, rebuild=args.rebuild)


if __name__ == '__main__':
    main()
