"""Reassess existing Newton-labeled trajectories without simulation or pose edits."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from somaforge_core import contact_order_acceptance
from somaforge_core import trajectory_refinement as workflow
from somaforge_core.newton_contact_data import load_contact_labels
from somaforge_core.rollout_dataset import task_check


def reassess(plan_path, manifest_path, output):
    plan = workflow.read_json(plan_path)
    original = workflow.read_json(manifest_path)
    manifest = copy.deepcopy(original)
    cases = {str(Path(c["reference"]["motion_file"]).resolve()): c for c in plan["cases"]}
    provenance = dict(schema=contact_order_acceptance.SCHEMA,
        training_plan=workflow.fingerprint(plan_path), source_manifest=workflow.fingerprint(manifest_path),
        implementation=[workflow.fingerprint(Path(contact_order_acceptance.__file__)),
                        workflow.fingerprint(Path(__file__)),
                        workflow.fingerprint(Path(__file__).resolve().parents[1]/
                            "packages/somaforge_core/somaforge_core/rollout_dataset.py")],
        changed="task acceptance only; original trajectories and Newton labels retained")
    manifest["acceptance_revision"] = provenance
    rows = []
    for entry in manifest["motion_files"]:
        case = cases[str(Path(entry["reference_motion_file"]).resolve())]
        workflow.verify_files([dict(path=entry["motion_file"], sha256=entry["motion_sha256"]),
                               dict(path=entry["contacts"], sha256=entry["contacts_sha256"])])
        labels = load_contact_labels(entry["motion_file"], entry["contacts"])
        with np.load(entry["contacts"], allow_pickle=False) as z:
            catalog = json.loads(z["contact_semantics_json"].item())["scene"]["surface_catalog"]
        old = entry["task_acceptance"]
        entry["task_acceptance"] = task_check(case, labels, catalog)
        result = entry["task_acceptance"]
        rows.append(dict(key=case["key"], identity=case["identity"],
                         old_passed=old["passed"], old_failed_checks=old["failed_checks"],
                         passed=result["passed"], failed_checks=result["failed_checks"],
                         check_count=result["check_count"]))
        print(f"Reassessed {len(rows)}/{len(manifest['motion_files'])}: {case['key']} "
              f"failed={result['failed_checks']} (old={old['failed_checks']})", flush=True)
    summary = dict(provenance=provenance, total=len(rows),
                   passed=sum(r["passed"] for r in rows),
                   old_passed=sum(r["old_passed"] for r in rows),
                   failed_checks=sum(r["failed_checks"] for r in rows),
                   check_count=sum(r["check_count"] for r in rows), motions=rows)
    output.mkdir(parents=True, exist_ok=True)
    workflow.write_json(output/"dataset_manifest.json", manifest, immutable=True)
    workflow.write_json(output/"acceptance_summary.json", summary, immutable=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reassess(args.training_plan, args.manifest, args.output_dir)
