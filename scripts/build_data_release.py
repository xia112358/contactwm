#!/usr/bin/env python3
"""Build immutable data-release assets from explicitly enumerated inventories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from download_data import SCHEMA, check_file, checked_relative, selected_components, sha256


def build(inventories, output, repository, tag, catalog_path, part_bytes=1024**3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base_url = f"https://github.com/{repository}/releases/download/{tag}"
    components, aliases, checkouts = {}, {}, []
    for filename in inventories:
        data = json.loads(Path(filename).read_text())
        checkout = data.get("repo_root", data.get("source_roots", {}).get("checkout"))
        if checkout:
            checkouts.append(Path(checkout).resolve())
        for component in data["components"]:
            if component["id"] in components:
                raise ValueError(f"Duplicate component ID: {component['id']}")
            if component.get("missing_inputs"):
                raise ValueError(f"Unresolved inputs in {component['id']}: {component['missing_inputs']}")
            components[component["id"]] = component
        rows = data.get("source_roots_aliases", [])
        if not rows:
            roots = data.get("source_roots", {})
            rows = [{"source": roots[key], "target": target} for key, target in (
                ("checkout", "."), ("runtime", "runtime/current")) if key in roots]
        for row in rows:
            source = row.get("source", row.get("source_root"))
            target = row.get("target", row.get("destination_root"))
            if not source or not Path(source).is_absolute():
                raise ValueError(f"Source root must be absolute: {source}")
            checked_relative(target)
            if source in aliases and aliases[source] != target:
                raise ValueError(f"Conflicting source root: {source}")
            aliases[source] = target
    graph = {"components": components}
    order = selected_components(graph, ["all"])
    owners, hashes, plans, repository_files = {}, {}, {}, {}
    catalog = {"schema": SCHEMA, "repository": repository, "tag": tag,
               "source_roots_aliases": [{"source": source, "target": target}
                                       for source, target in sorted(aliases.items())],
               "repository_files": [], "components": {}}
    def asset(path):
        return {"name": path.name, "url": f"{base_url}/{path.name}",
                "size": path.stat().st_size, "sha256": sha256(path)}
    def tracked_input(source, destination, digest):
        # Small inputs already distributed by Git retain their original location.
        # Require both Git tracking and identical committed bytes before omitting.
        for checkout in checkouts:
            if source.resolve() != (checkout / destination).resolve():
                continue
            result = subprocess.run(["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--", destination],
                                    capture_output=True)
            if result.returncode:
                continue
            committed = subprocess.run(["git", "-C", str(checkout), "show", f"HEAD:{destination}"],
                                       capture_output=True, check=True).stdout
            if hashlib.sha256(committed).hexdigest() != digest:
                raise ValueError(f"Tracked data differs from committed input: {destination}")
            return True
        return False

    # Resolve all ownership and aliases before writing any index. Later components
    # can share a payload while adding historical source names to its owner.
    for name in order:
        component = components[name]
        dependencies = set(component.get("dependencies", []))
        records, sources = [], {}
        for row in sorted(component["files"], key=lambda item: item["destination"]):
            destination = checked_relative(row["destination"]).as_posix()
            source = Path(row["source"])
            if not source.is_absolute():
                raise ValueError(f"Inventory source must be absolute: {source}")
            stat = source.stat()
            if stat.st_size != row["size"] or not source.is_file():
                raise ValueError(f"Inventory changed: {source}")
            key = (str(source.resolve()), stat.st_size, stat.st_mtime_ns)
            digest = hashes.setdefault(key, None)
            if digest is None:
                digest = hashes[key] = sha256(source)
            record = {"path": destination, "size": stat.st_size, "sha256": digest,
                      "sources": sorted(set([str(source), str(source.resolve()), *row.get("aliases", [])]))}
            if any(not Path(value).is_absolute() for value in record["sources"]):
                raise ValueError(f"Historical source aliases must be absolute: {destination}")
            if not destination.startswith("runtime/current/"):
                if not tracked_input(source, destination, digest):
                    raise ValueError(f"Untracked data payload outside runtime/current: {destination}")
                previous = repository_files.setdefault(destination, record)
                previous["sources"] = sorted(set(previous["sources"]) | set(record["sources"]))
                continue
            if destination in owners:
                owner, previous = owners[destination]
                if (previous["sha256"], previous["size"]) != (digest, stat.st_size):
                    raise ValueError(f"Conflicting data versions at {destination}")
                if owner != name:
                    dependencies.add(owner)
                previous["sources"] = sorted(set(previous["sources"]) | set(record["sources"]))
                continue
            owners[destination] = (name, record)
            records.append(record)
            sources[destination] = source
        plans[name] = (records, sources, dependencies)
    catalog["repository_files"] = [repository_files[key] for key in sorted(repository_files)]
    for name in order:
        component = components[name]
        records, sources, dependencies = plans[name]
        groups, current, size = [], [], 0
        for record in records:
            if record["size"] > part_bytes:
                raise ValueError(f"Single file exceeds configured archive part limit: {record['path']}")
            if current and size + record["size"] > part_bytes:
                groups.append(current)
                current, size = [], 0
            current.append(record)
            size += record["size"]
        if current:
            groups.append(current)
        archives = []
        for number, group in enumerate(groups, 1):
            path = output / f"{name}.part-{number:03d}.tar.gz"
            print(f"Packing {path.name}: {len(group)} files", flush=True)
            with tarfile.open(path, "x:gz", compresslevel=3) as bundle:
                for record in group:
                    source = sources[record["path"]]
                    info = tarfile.TarInfo(record["path"])
                    info.size, info.mode, info.mtime = record["size"], 0o644, 0
                    before = source.stat()
                    with source.open("rb") as stream:
                        bundle.addfile(info, stream)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f"Source changed while packing: {source}")
            if path.stat().st_size >= 2 * 1024**3:
                raise ValueError(f"Release asset exceeds GitHub's per-file limit: {path}")
            archives.append(asset(path))
        index = output / f"{name}.files.json"
        index.write_text(json.dumps({"schema": "somaforge_data_component_files_v1", "component": name,
                                    "files": records}, indent=2) + "\n")
        catalog["components"][name] = {
            "purpose": component["purpose"], "status": component["status"],
            "dependencies": sorted(dependencies), "file_count": len(records),
            "bytes": sum(row["size"] for row in records), "index": asset(index), "archives": archives,
            "provenance_notes": component.get("provenance_notes", [])}
    selected_components(catalog, ["all"])
    Path(catalog_path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(catalog, indent=2) + "\n"
    Path(catalog_path).write_text(text)
    (output / "data_releases.json").write_text(text)
    checksums = [f"{sha256(path)}  {path.name}" for path in sorted(output.iterdir()) if path.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return catalog


def verify_release(catalog, output):
    """Stream every declared member without making a second installed data copy."""
    output = Path(output)
    total = 0
    for name in selected_components(catalog, ["all"]):
        component = catalog["components"][name]
        index = output / component["index"]["name"]
        if not check_file(index, component["index"]):
            raise ValueError(f"Corrupt release index: {index}")
        listing = json.loads(index.read_text())
        expected = {record["path"]: record for record in listing["files"]}
        if (listing["component"] != name or len(expected) != component["file_count"]
                or len(expected) != len(listing["files"])):
            raise ValueError(f"Inconsistent release index: {name}")
        observed = set()
        for record in component["archives"]:
            path = output / record["name"]
            if not check_file(path, record):
                raise ValueError(f"Corrupt release asset: {path}")
            with tarfile.open(path, "r|gz") as bundle:
                for member in bundle:
                    checked_relative(member.name)
                    if not member.isfile() or member.name not in expected or member.name in observed:
                        raise ValueError(f"Invalid release archive member: {member.name}")
                    row = expected[member.name]
                    digest = hashlib.sha256()
                    with bundle.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if member.size != row["size"] or digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"Release member checksum mismatch: {member.name}")
                    observed.add(member.name)
        if observed != set(expected):
            raise ValueError(f"Release archives are incomplete: {name}")
        total += len(observed)
        print(f"Stream-verified {name}: {len(observed)} files", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/data_releases.json"))
    parser.add_argument("--repository", default="xia112358/somaforge")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Stream-check existing assets without extracting")
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(10)
    if args.verify_only:
        catalog = json.loads(args.catalog.read_text())
    else:
        if not args.inventories:
            parser.error("--inventories is required when building")
        catalog = build(args.inventories, args.output, args.repository, args.tag, args.catalog)
    verify_release(catalog, args.output)


if __name__ == "__main__":
    main()
