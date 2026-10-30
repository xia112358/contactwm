"""Archive accepted trajectories, rematerialize native labels, and cut own events."""
import argparse
from pathlib import Path
import json

from materialize_task_contact_labels import materialize
from somaforge_core.refined_contact_dataset import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'dataset-root', 'baseline-index'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--display-name', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stable-frames', type=int, default=3)
    parser.add_argument('--merge-gap', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    summary = publish(args.source_manifest, args.dataset_root, args.version, args.display_name,
        args.baseline_index, materialize_labels=materialize, workers=args.workers,
        stable_frames=args.stable_frames, merge_gap=args.merge_gap, resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__': main()
