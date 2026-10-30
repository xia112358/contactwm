#!/usr/bin/env python3
"""Inventory explicit training/demo releases without copying or changing inputs.

Only stdlib is used. Torch ZIP pickle opcodes are inspected, never executed.
Historical tmp references are resolved only through retained archive copies.
The output is a build input, not a new training-admission or contact certificate.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import pickletools
import struct
import zipfile


SCHEMA = "somaforge_data_release_inventory_v1"
SUFFIXES = {".json", ".jsonl", ".npz", ".npy", ".pt", ".pth", ".obj", ".urdf", ".xml", ".md", ".txt"}
SKIP_KEYS = {"output", "output_dir", "output_path", "output_motion_path", "log_dir", "source_record", "original_paths"}
MAX_METADATA = 16 * 1024 * 1024
OLD_SURFACE_MESH = "/runtime/current/terrains/climb00_pairwise48/multi_boxes_z_scale_1.0.obj"


def strings(value, key=""):
    """Yield possible file references, including paths used as dictionary keys."""
    if isinstance(value, dict):
        for name, child in value.items():
            if name in SKIP_KEYS or str(name).startswith("output_"):
                continue
            if isinstance(name, str) and "/" in name:
                yield name, "mapping_key"
            yield from strings(child, str(name))
    elif isinstance(value, list):
        for child in value:
            yield from strings(child, key)
    elif isinstance(value, str):
        if value.startswith(("{", "[")):
            try:
                yield from strings(json.loads(value), key)
                return
            except (ValueError, RecursionError):
                pass
        if "\n" not in value and len(value) < 4096 and "/" in value:
            yield value, key


def npy_strings(raw):
    """Decode metadata-only Unicode/bytes NPY arrays without importing NumPy."""
    if raw[:6] != b"\x93NUMPY":
        return []
    version = raw[6]
    offset = 10 if version == 1 else 12
    width = 2 if version == 1 else 4
    length = int.from_bytes(raw[8:8 + width], "little")
    header = ast.literal_eval(raw[offset:offset + length].decode("latin1").strip())
    dtype = header["descr"]
    body = raw[offset + length:]
    if isinstance(dtype, str) and dtype[1:2] == "U":
        unit = int(dtype[2:]) * 4
        codec = "utf-32-be" if dtype.startswith(">") else "utf-32-le"
    elif isinstance(dtype, str) and dtype[:2] in {"|S", "<S", ">S"}:
        unit, codec = int(dtype[2:]), "utf-8"
    else:
        return []
    if unit <= 0:
        return []
    return [body[i:i + unit].decode(codec, errors="replace").rstrip("\0")
            for i in range(0, len(body), unit)]


class Builder:
    def __init__(self, root):
        self.root = root.resolve()
        self.runtime = (root / "runtime/current").resolve()
        self.dataset = self.runtime / "motions/climb00_contact_dataset"
        self.archive = self.runtime / "archives/preserved_inputs_20261010"
        self.aliases = [
            {"source_root": str(self.runtime), "destination_root": "runtime/current"},
            {"source_root": str(self.root), "destination_root": "."},
        ]
        self.migration = {}
        migration = self.archive / "migration.json"
        if migration.is_file():
            for record in json.loads(migration.read_text()).get("records", {}).values():
                self.migration[record["source"]] = record["target"]
        self.components = []
        self.parse_cache = {}
        self.direct_reference_gaps = []
        self.file_aliases = defaultdict(set)
        self.evidence_alias_targets = {}
        self.alias_conflicts = defaultdict(set)

    def destination(self, path):
        for alias in self.aliases:
            try:
                tail = path.relative_to(alias["source_root"])
                return (Path(alias["destination_root"]) / tail).as_posix()
            except ValueError:
                pass
        raise ValueError(f"Input is outside declared source roots: {path}")

    def is_tmp(self, path):
        parts = path.parts
        return path == Path("/tmp") or str(path).startswith("/tmp/") or "tmp" in parts

    def alias(self, target, original):
        target = target.resolve()
        if Path(original).is_absolute():
            self.file_aliases[str(target)].add(str(original))
            previous = self.evidence_alias_targets.get(str(original))
            if previous is not None and previous != target:
                self.alias_conflicts[str(original)].update((str(previous), str(target)))
            else:
                self.evidence_alias_targets[str(original)] = target

    def register_evidence_aliases(self, value, parent):
        """Bind exact content-hash evidence to retained copies, never to changed labels."""
        records = []
        def walk(item):
            if isinstance(item, dict):
                if isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                    records.append(item)
                for child in item.values():
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)
        walk(value)
        available = {}
        for record in records:
            original = Path(record["path"])
            if original.is_absolute() and not self.is_tmp(original) and original.is_file():
                available.setdefault(record["sha256"], original.resolve())
        for record in records:
            original = record["path"]
            if not Path(original).is_absolute():
                continue
            target = available.get(record["sha256"])
            if target is None:
                candidate = self.dataset / "objects" / (record["sha256"] + Path(original).suffix)
                if candidate.is_file():
                    target = candidate
            if target is not None:
                # Sidecar declarations alone cannot establish equality of a copy.
                if hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
                    self.alias(target, original)

    def resolve(self, value, parent):
        if "://" in value or value.startswith(("/World/", "/env_", "app://")):
            return None, "not_artifact"
        raw = Path(value).expanduser()
        if raw.suffix.lower() not in SUFFIXES and not any(t in value for t in ("contact_layer", "/contact/")):
            return None, "not_artifact"
        candidates = []
        if value in self.evidence_alias_targets:
            candidates.append(self.evidence_alias_targets[value])
        if raw.is_absolute():
            if self.is_tmp(raw):
                target = self.migration.get(str(raw))
                if target:
                    candidates.append(Path(target))
                for marker in ("/runtime/current/holosoma/tmp/", "/holosoma/tmp/", "/tmp/"):
                    if marker in str(raw):
                        candidates.append(self.archive / str(raw).split(marker, 1)[1])
            else:
                candidates.append(raw)
        else:
            # dataset archive indexes explicitly resolve objects/ and versions/
            # against the dataset root, not against their index's directory.
            if value.startswith(("objects/", "versions/")):
                candidates.append(self.dataset / raw)
            candidates.extend([parent / raw, self.root / raw, self.archive / raw])
            # Normalize logical repo paths before following runtime/current's link.
            candidates.append(Path(os.path.abspath(parent / raw)))
            try:
                logical_parent = self.root / "runtime/current" / parent.relative_to(self.runtime)
                candidates.append(Path(os.path.abspath(logical_parent / raw)))
            except ValueError:
                pass
        for candidate in candidates:
            physical = candidate.resolve()
            if self.is_tmp(physical):
                continue
            if physical.exists():
                try:
                    self.destination(physical)
                except ValueError:
                    continue
                original = str(raw) if raw.is_absolute() else str(Path(os.path.abspath(parent / raw)))
                self.alias(physical, original)
                return physical, "resolved"
        return None, "historical_tmp_not_read" if self.is_tmp(raw) else "missing_or_external"

    def references(self, path, inspect_torch=False):
        cache_key = (str(path), inspect_torch)
        if cache_key in self.parse_cache:
            return self.parse_cache[cache_key]
        values = []
        errors = []
        try:
            if path.suffix == ".json":
                values = [json.loads(path.read_text())]
            elif path.suffix == ".jsonl":
                values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            elif path.suffix == ".npz":
                with zipfile.ZipFile(path) as archive:
                    for entry in archive.infolist():
                        name = entry.filename.lower()
                        if entry.file_size <= MAX_METADATA and any(s in name for s in ("json", "provenance", "path", "file")):
                            values.extend(npy_strings(archive.read(entry)))
            elif inspect_torch and path.suffix in {".pt", ".pth"} and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    entry = next((e for e in archive.infolist() if e.filename.endswith("/data.pkl")), None)
                    if entry and entry.file_size <= 128 * 1024 * 1024:
                        values = [arg for _, arg, _ in pickletools.genops(archive.read(entry)) if isinstance(arg, str)]
        except (ValueError, SyntaxError, UnicodeError, zipfile.BadZipFile, struct.error) as error:
            errors.append({"source": str(path), "error": str(error)})
        refs = [item for value in values for item in strings(value)]
        self.parse_cache[cache_key] = refs, errors
        return refs, errors

    def component(self, identity, purpose, status, seeds, dependencies=(), trees=(), inspect_torch=False, notes=(), recursive=True):
        files, missing, skipped, errors = {}, set(), set(), []
        reference_gaps, provenance_gaps = [], []
        queue = deque()
        for seed in seeds:
            p = Path(seed)
            p = p if p.is_absolute() else self.root / p
            if not p.is_file():
                missing.add(str(p))
            else:
                queue.append(p.resolve())
        for tree in trees:
            p = Path(tree)
            p = p if p.is_absolute() else self.root / p
            if not p.is_dir():
                missing.add(str(p))
            else:
                queue.extend(q.resolve() for q in p.rglob("*") if q.is_file()
                             and q.suffix.lower() in SUFFIXES and "generation_incomplete" not in q.parts
                             and "work" not in q.parts)
        while queue:
            p = queue.popleft()
            if str(p) in files:
                continue
            if self.is_tmp(p):
                raise ValueError(f"Refusing temporary input: {p}")
            destination = self.destination(p)
            files[str(p)] = {"source": str(p), "destination": destination, "size": p.stat().st_size}
            if not recursive:
                continue
            # Backup catalogs and training result reports are retained as evidence,
            # but must not recursively pull every unrelated experiment into a release.
            if p.name.startswith("catalog") or p.name in {
                "progress.json", "performance.json", "release_audit.json", "migration.json",
                "source_result.json", "source_continuity.json",
            }:
                continue
            refs, parse_errors = self.references(p, inspect_torch=inspect_torch and p.name == "corpus.pt")
            errors.extend(parse_errors)
            for value, key in refs:
                target, reason = self.resolve(value, p.parent)
                if target is not None and target.is_file():
                    queue.append(target)
                elif target is not None and target.is_dir() and "contact" in key:
                    queue.extend(q.resolve() for q in target.rglob("*") if q.is_file() and q.suffix.lower() in SUFFIXES)
                elif reason == "historical_tmp_not_read":
                    skipped.add(value)
                    reference_gaps.append({"declaration": str(p), "field": key, "value": value, "reason": reason})
                elif reason == "missing_or_external":
                    gap = {"declaration": str(p), "field": key, "value": value, "reason": reason}
                    if (identity.startswith("demonstrations_") and key == "mesh_path"
                            and p.suffix == ".jsonl" and "/surfaces/" in str(p)
                            and value.endswith(OLD_SURFACE_MESH)):
                        gap.update(classification="historical_surface_extraction_provenance",
                                   required_for_prepared_training_or_query=False,
                                   historical_lte_regeneration_complete=False)
                        provenance_gaps.append(gap)
                    else:
                        missing.add(value)
                        gap["classification"] = "unresolved_required_or_unclassified_input"
                    reference_gaps.append(gap)
        ordered = sorted(files.values(), key=lambda item: item["destination"])
        entry = dict(id=identity, component_id=identity, purpose=purpose, status=status,
                     dependencies=list(dependencies), files=ordered, file_count=len(ordered),
                     size=sum(f["size"] for f in ordered), source_roots_aliases=self.aliases,
                     missing_inputs=sorted(missing), historical_tmp_references_not_read=sorted(skipped),
                     reference_gaps=reference_gaps, metadata_errors=errors, provenance_notes=list(notes))
        entry["provenance_gaps"] = provenance_gaps
        if provenance_gaps:
            canonical = self.root / "configs/motion_matched/terrain_obj_cache/climb_00/multi_boxes_z_scale_1.0.obj"
            retained = self.runtime / "terrains/climb00_pairwise48/h100.obj"
            entry["provenance_notes"].append({
                "kind": "historical_surface_mesh_address_unrecoverable",
                "references": provenance_gaps,
                "evidence": "Surface JSONL records top-face polygons/bounds but no source mesh SHA. The retained h100.obj and tracked climb00 OBJ have the same SHA; this does not establish the missing original file's byte identity.",
                "available_h100_sha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
                "tracked_climb00_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
                "runtime_boundary": "Prepared demonstration arrays, surface catalogs and each index's actual query terrains are present. Predictor contact-cache validation hashes the catalog; query geometry comes from explicit scene/terrain manifests. Historical LTE regeneration samples metadata.mesh_path and is not declared complete.",
                "replacement_created": False,
            })
        self.components.append(entry)
        print(f"Built {identity}: {len(ordered)} files, {entry['size']} bytes", flush=True)
        return entry

    def declared_files(self, manifest):
        """WBT manifest fields are inputs; provenance inside NPZ is not a new recipe."""
        data = json.loads(manifest.read_text())
        result = [manifest]
        for group in ("motion_files", "terrains"):
            for entry in data.get(group, []):
                for key, value in entry.items():
                    if key.endswith("_file") and isinstance(value, str):
                        path, _ = self.resolve(value, manifest.parent)
                        if path is None:
                            raise FileNotFoundError(f"Declared manifest input: {manifest}: {key}={value}")
                        result.append(path)
        return result

    def corpus_inputs(self, corpus):
        """Inspect self-contained corpus provenance without executing Torch pickle."""
        refs, errors = self.references(corpus, inspect_torch=True)
        if errors:
            raise ValueError(errors)
        found, skipped = [corpus], []
        for value, _ in refs:
            path, reason = self.resolve(value, corpus.parent)
            if path is not None and path.is_file():
                found.append(path)
            elif reason != "not_artifact":
                skipped.append(value)
        return found, sorted(set(skipped))

    def direct_inputs(self, path):
        """One declaration level: retain hash evidence without chasing experiment history."""
        found = [path]
        if path.suffix == ".json" and path.name == "release.json":
            self.register_evidence_aliases(json.loads(path.read_text()), path.parent)
        refs, errors = self.references(path)
        if errors:
            raise ValueError(errors)
        for value, key in refs:
            target, reason = self.resolve(value, path.parent)
            if target is not None and target.is_file():
                found.append(target)
            elif target is not None and target.is_dir() and "contact" in key:
                found.extend(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
            elif target is None and reason != "not_artifact":
                self.direct_reference_gaps.append({"declaration": str(path), "field": key,
                                                   "value": value, "reason": reason})
        return found


def build(root):
    b = Builder(root)
    manifest_root = b.runtime / "manifests"
    b.component("wbt_baseline29", "Canonical 29-motion WBT baseline and original source evidence",
                "canonical_kinematics_no_new_contact_certification", b.declared_files(manifest_root / "omniretarget_baseline_29.json"), recursive=False)
    b.component("wbt_legacy8part_full_demonstrations", "Legacy eight-part references and complete original rollout recordings",
                "historical_force_threshold_masks_not_native_contact_truth", b.declared_files(manifest_root / "newton_contact_force_8part.json"),
                dependencies=["wbt_baseline29"], trees=[b.runtime / "rollout/newton_contact_force"], recursive=False)

    baseline_dir = root / "baselines/predictor_v1_latest_20261010"
    baseline = json.loads((baseline_dir / "baseline.json").read_text())
    predictor_seeds = [root / d["path"] for d in baseline["dependencies"].values()]
    dataset_manifest = root / baseline["dependencies"]["manifest"]["path"]
    dataset = json.loads(dataset_manifest.read_text())
    for row in dataset["motion_files"]:
        for key in ("motion_file", "newton_contact_file", "event_segments_file", "edit_plan_file"):
            predictor_seeds.append(Path(row[key]))
        plan = json.loads(Path(row["edit_plan_file"]).read_text())
        predictor_seeds.append(Path(plan["metadata"]["target_surface_catalog"]))
    for declaration in dataset["demonstration_event_evidence"]:
        certificate = Path(declaration["path"])
        predictor_seeds.append(certificate)
        predictor_seeds.extend(Path(record["path"]) for record in json.loads(certificate.read_text())["evidence"].values())
    predictor_seeds.extend(Path(row["terrain_file"]) for row in dataset["terrains"])
    predictor_seeds.extend(b.declared_files(root / baseline["dependencies"]["query_source_manifest"]["path"]))
    b.component("predictor_current_training_data", "Registered current Predictor demonstration/cache/query-scene input closure",
                "registered_historical_runtime_training_ready_unchanged", predictor_seeds,
                notes=["Original manifest says training_ready=false and retains its recorded Newton dev runtime.",
                       "No label, event, fingerprint or training-admission field is changed."], recursive=False)
    b.component("predictor_current_reference", "Current saved analysis checkpoint and recorded material geometry",
                baseline["status"], [root / baseline["checkpoint"]["path"],
                *(root / d["path"] for d in baseline["training_geometry_materials"].values()),
                *baseline_dir.glob("*.json"), baseline_dir / "README.md"],
                dependencies=["predictor_current_training_data"], recursive=False)

    prefix = b.runtime / "motion_edit/inputs/preserved_20261010/climb00"
    b.component("motion_edit_height110", "Retained height110 editing source, plan, layers, terrain and result",
                "reference_example_not_new_contact_admission", [prefix / "height110.json", prefix / "height110_motion.npz",
                root / "packages/motion_edit/data/motions/climb00_height110_current_main.json"],
                trees=[root / "packages/motion_edit/data/layers/contact/climb00_source",
                       root / "packages/motion_edit/data/layers/contact/task_variants/climb_00_height_1p100"])

    for name, folder in [("infiller_demo_latest", "infiller_latest_predictor_20261010"),
                         ("infiller_demo_recursive", "infiller_recursive_v2_20261011")]:
        corpus = b.runtime / "motions" / folder / "corpus.pt"
        inputs, skipped = b.corpus_inputs(corpus)
        b.component(name, "Prepared full Infiller demonstration corpus and embedded input references",
                    "prepared_corpus_preserve_original_acceptance", inputs,
                    dependencies=["predictor_current_training_data", "predictor_current_reference"], recursive=False,
                    notes=[{"historical_or_unavailable_embedded_references": skipped}])
    historical_inputs, historical_skipped = b.corpus_inputs(b.archive / "contact_shared_full_v1/corpus.pt")
    b.component("infiller_historical_shared_demo", "Retained complete 207-motion/4761-segment historical Infiller corpus",
                "historical_recipe_not_current_training_admission", [*historical_inputs,
                root / "docs/history/configs/direct_infiller_v1.json"], recursive=False,
                notes=["Embedded historical tmp inputs resolve only to retained migration/archive counterparts.",
                       {"historical_or_unavailable_embedded_references": historical_skipped}])

    catalog = json.loads((b.dataset / "catalog.json").read_text())
    archive_ids = []
    for version in catalog["versions"]:
        if version["name"] == "dynamics_corrected_207_20261017":
            continue
        identity = "demonstrations_" + version["name"]
        archive_ids.append(identity)
        index_path = b.dataset / version["index"]
        index = json.loads(index_path.read_text())
        for row in index.get("motion_files", []):
            original = row.get("source_record", {})
            for key, value in row.get("files", {}).items():
                target, _ = b.resolve(value, index_path.parent)
                if target is not None and isinstance(original.get(key), str):
                    b.alias(target, original[key])
        for row in index.get("terrains", []):
            target, _ = b.resolve(row["file"], index_path.parent)
            original = row.get("source_record", {}).get("terrain_file")
            if target is not None and isinstance(original, str):
                b.alias(target, original)
        if index.get("original_manifest") and index.get("source_manifest"):
            target, _ = b.resolve(index["original_manifest"], index_path.parent)
            if target is not None:
                b.alias(target, index["source_manifest"])
        b.component(identity, "Registered extracted demonstration variant and referenced content-addressed objects",
                    "archived_preserve_original_approval_and_semantics", [index_path],
                    notes=["Only this index's object references are included; source_record paths remain historical metadata."])

    standard = b.dataset / "versions/source_prefix_standard_20261014/edits207_joint_v14"
    standard_inputs = []
    for declaration in [standard / "manifest.json", standard / "preparation.json", standard / "status.json",
                        *sorted((standard / "motions").glob("*/reference_complete.json")),
                        *sorted((standard / "plans").glob("*.json"))]:
        standard_inputs.extend(b.direct_inputs(declaration))
    b.component("source_prefix_standard207", "207 source-prefix-standard kinematic references, plans and completion evidence",
                "kinematic_training_input_not_accepted_demonstration", standard_inputs, recursive=False,
                notes=["Failed generation directories and unreferenced PyRoki/taskspace/semantic-proxy work caches are excluded.",
                       "Generation reports are retained as evidence, not treated as an instruction to package every intermediate."])
    dynamics = b.dataset / "versions/dynamics_corrected_207_20261017"
    dynamics_inputs = b.declared_files(dynamics / "manifest.json")
    for declaration in sorted((dynamics / "provenance").glob("*/release.json")):
        # release.json declares baseline object identities and emitted evidence.
        # The baseline component supplies those objects' transitive source closure.
        dynamics_inputs.extend(b.direct_inputs(declaration))
    dynamics_component = b.component("dynamics_corrected207", "207 full dynamics-corrected demonstrations, native labels, 3149 events and provenance",
                "all_accepted_training_ready_false_unchanged", [*dynamics_inputs, dynamics / "index.json",
                dynamics / "summary.json", dynamics / "comparison.json", dynamics / "README.md", dynamics / "release_audit.json"],
                trees=[dynamics / "provenance"], dependencies=["demonstrations_full1000_fixed_timeline_newton_20260922_v3"],
                notes=["206 automatically accepted and 1 accepted by explicit user review; no training-start/admission field is changed."], recursive=False)
    old_contact_gaps = []
    for gap in b.direct_reference_gaps:
        if str(gap["declaration"]).startswith(str(dynamics)) and gap["value"].endswith("/contacts.npz"):
            gap.update(classification="unrecoverable_original_source_contact_provenance",
                       required_for_prepared_training_or_query=False)
            old_contact_gaps.append(gap)
    if old_contact_gaps:
        dynamics_component["provenance_gaps"].extend(old_contact_gaps)
        dynamics_component["provenance_notes"].append({
            "kind": "unrecoverable_original_contact_source_files",
            "count": len(old_contact_gaps),
            "references": old_contact_gaps,
            "evidence": "release.json retains hashes of nine original tmp contacts.npz sources. Their hashes differ from the published rematerialized labels and no exact-hash object copy was found. The old files are not read, replaced or aliased to changed data.",
            "runtime_boundary": "All 207 published motions, rematerialized contact labels, 3149 own events, scene catalogs and terrain inputs are included. The provenance gap concerns the original source files, not the complete prepared dynamics-corrected dataset. training_ready=false remains unchanged.",
            "replacement_created": False,
        })
    b.component("demonstration_catalog", "Source archive catalog and retained path-migration ledger",
                "inventory_only_not_training_admission", [b.dataset / "catalog.json", b.archive / "migration.json"],
                dependencies=archive_ids + ["dynamics_corrected207"])
    return b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/data_release_20261022"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    builder = build(root)
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    # Outputs may use tmp; inputs categorically may not.
    output.mkdir(parents=True, exist_ok=True)
    filelists = output / "filelists"
    filelists.mkdir(exist_ok=True)
    union = {}
    for component in builder.components:
        for item in component["files"]:
            item["aliases"] = sorted(builder.file_aliases[item["source"]] |
                                     {item["source"], str(root / item["destination"])})
            prior = union.setdefault(item["destination"], item)
            if prior["source"] != item["source"]:
                raise ValueError(f"Destination collision: {item['destination']}")
        (filelists / (component["id"] + ".json")).write_text(json.dumps(component, indent=2, ensure_ascii=False) + "\n")
        print(f"{component['id']}: {component['file_count']} files, {component['size']} bytes, {len(component['missing_inputs'])} unresolved references")
    inventory = dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
                     repo_root=str(root), source_roots_aliases=builder.aliases,
                     component_count=len(builder.components), components=builder.components,
                     unique_file_count=len(union), unique_size=sum(f["size"] for f in union.values()),
                     direct_reference_gaps=builder.direct_reference_gaps,
                     alias_conflicts={key: sorted(value) for key, value in builder.alias_conflicts.items()},
                     operation="inventory_only_no_copy_upload_or_source_modification",
                     policy="Preserve original status/contact provenance; never use tmp as a source; no recursive all-model/ablation sweep.")
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(f"UNIQUE: {len(union)} files, {inventory['unique_size']} bytes; {output / 'inventory.json'}")


if __name__ == "__main__":
    main()
