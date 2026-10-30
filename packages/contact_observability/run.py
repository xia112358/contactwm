"""Independent entry point for contact-target observation diagnostics."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tmp" / "contact_observability"


def output_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / "tmp").resolve()):
        raise argparse.ArgumentTypeError("Analysis outputs must stay inside the project's tmp/ directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-history", help="Copy historical evidence and record SHA256 provenance")
    importer.add_argument("--source-root", type=Path, default=ROOT / "tmp")
    importer.add_argument("--output", type=output_path, default=DEFAULT_OUTPUT)
    importer.add_argument("--json", action="store_true", help="Print the complete summary as JSON")
    summary = commands.add_parser("summarize", help="Read the imported start-frame and route evidence")
    summary.add_argument("--output", type=output_path, default=DEFAULT_OUTPUT)
    summary.add_argument("--write", action="store_true", help="Also write summary.json under --output")
    summary.add_argument("--json", action="store_true", help="Print the complete summary as JSON")
    route = commands.add_parser("route", help="Replay the full-motion ideal depth coverage diagnostic on CPU")
    route.add_argument("--history", type=Path, default=DEFAULT_OUTPUT / "history")
    route.add_argument("--output", type=output_path, default=DEFAULT_OUTPUT / "route")
    route.add_argument(
        "--training-manifest",
        type=Path,
        default=ROOT
        / "runtime/current/motions/climb00_contact_dataset/training/predictor_loaded_material207_joint_v12_20261004/manifest.json",
    )
    route.add_argument("--motion-index", type=int, default=144)
    route.add_argument("--scene-metadata", type=Path, help="Defaults to the imported route's scene metadata")
    route.add_argument("--profile", default="torso_depth_87x58_fov89p5_pitch27_v1")
    route.add_argument(
        "--coverage-distance-m",
        type=float,
        default=0.04,
        help="Geometric same-surface sampling tolerance; never a contact threshold",
    )
    args = parser.parse_args()
    # Revalidate defaults too, including any existing symlinks.
    args.output = output_path(args.output)
    if args.command == "route":
        from route_coverage import run_route

        run_route(
            output=args.output,
            history=args.history,
            profile=args.profile,
            motion_index=args.motion_index,
            training_manifest=args.training_manifest,
            scene_metadata=args.scene_metadata,
            coverage_distance_m=args.coverage_distance_m,
        )
    else:
        from history import import_history, summarize

        if args.command == "import-history":
            import_history(args.source_root, args.output)
        result = summarize(args.output)
        if args.command == "import-history" or args.write:
            (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            start = result["start_frame"]["metrics"]
            route = result["full_route"]["metrics"]
            print(f"Historical evidence: {result['source_files']} files; {args.output / 'history'}")
            print(
                f"Start-frame visibility: {start['visible']}/{start['targets']}; outside FOV: {start['reasons'].get('outside_fov', 0)}"
            )
            print(f"Full route ({result['full_route']['frames']} frames, {route['total']} targets):")
            for key in ("covered_at_start", "seen_by_start", "seen_before_target", "seen_anytime"):
                print(f"  {key}: {route[key]}/{route['total']}")
            for row in result["full_route"]["late_only_targets"]:
                print(
                    f"  Late-only target {row['sample_id']} ({row['part']}): touchdown {row['target_frame']}, first covered {row['first_seen_frame']}"
                )
            print(f"Index: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
