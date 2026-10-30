#!/usr/bin/env python3
"""Fetch and verify explicitly selected SomaForge data-release components.

Only the standard library is required. Payloads are immutable; conflicting
existing files are rejected. Training, simulation and checkpoint selection are
never launched by this command.

Run only one installer per data directory at a time. Interrupted temporary
downloads/imports remain available for inspection; retries do not delete them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "somaforge_data_release_catalog_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_relative(value):
    if not isinstance(value, str):
        raise ValueError(f"Unsafe release path: {value!r}")
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or "\x00" in value or path.as_posix() != value):
        raise ValueError(f"Unsafe release path: {value!r}")
    return path


def absolute_alias(value):
    if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute():
        raise ValueError(f"Published data alias must be an absolute path: {value!r}")
    return Path(os.path.normpath(value))


def alias_target(root, value):
    relative = checked_relative(value)
    target = (root / relative).resolve()
    if not target.is_relative_to(root) and not target.is_relative_to((root / "runtime/current").resolve()):
        raise ValueError(f"Published data alias target escapes the checkout/data directory: {value}")
    return target


def target_path(root, value):
    relative = checked_relative(value)
    if relative.parts[:2] != ("runtime", "current") or len(relative.parts) < 3:
        raise ValueError(f"Release payload must be inside runtime/current: {value}")
    data_root = (root / "runtime/current").resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(data_root):
        raise ValueError(f"Release path escapes the configured data directory: {value}")
    return target


def repository_path(root, value):
    """Locate an already Git-distributed input; never an extraction target."""
    relative = checked_relative(value)
    if not relative.parts or relative.parts[:2] == ("runtime", "current"):
        raise ValueError(f"Repository input must be a checkout file outside runtime/current: {value}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Repository input escapes the checkout: {value}")
    return target


def check_file(path, record):
    return path.is_file() and path.stat().st_size == record["size"] and sha256(path) == record["sha256"]


def acquire(record, cache, local_archives=None):
    name = checked_relative(record["name"])
    if len(name.parts) != 1:
        raise ValueError("Release asset names must be basenames")
    if local_archives is not None:
        path = Path(local_archives) / name
        if not check_file(path, record):
            raise ValueError(f"Local release asset is missing or corrupt: {path}")
        return path
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if path.exists():
        if not check_file(path, record):
            raise ValueError(f"Conflicting cached asset, move it aside before retrying: {path}")
        return path
    if not record["url"].startswith("https://"):
        raise ValueError("Published data URLs must use HTTPS")
    partial = cache / f".partial-{uuid.uuid4().hex}"
    request = urllib.request.Request(record["url"], headers={"User-Agent": "SomaForge-data/1"})
    print(f"Downloading {name} ({record['size']:,} bytes)", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if not check_file(partial, record):
        raise ValueError(f"Download checksum mismatch; partial evidence retained: {partial}")
    partial.rename(path)
    return path


def selected_components(catalog, names):
    components = catalog["components"]
    requested = list(components) if names == ["all"] else names
    ordered, visiting = [], set()
    def visit(name):
        if name not in components:
            raise ValueError(f"Unknown component {name!r}; use the list command")
        if name in visiting:
            raise ValueError(f"Data dependency cycle at {name}")
        if name in ordered:
            return
        visiting.add(name)
        for dependency in components[name].get("dependencies", []):
            visit(dependency)
        visiting.remove(name)
        ordered.append(name)
    for name in requested:
        visit(name)
    return ordered


def extract_part(archive, root, expected, observed):
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile() or member.name not in expected:
                raise ValueError(f"Unexpected or nonregular archive member: {member.name}")
            if member.name in observed:
                raise ValueError(f"Duplicate archive member: {member.name}")
            observed.add(member.name)
            record = expected[member.name]
            if member.size != record["size"]:
                raise ValueError(f"Archive member size mismatch: {member.name}")
            target = target_path(root, member.name)
            if target.exists():
                if not check_file(target, record):
                    raise FileExistsError(f"Refusing to replace existing data: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # PHP raw release basenames can already be 217 bytes. The staging
            # name must fit NAME_MAX independently of the final basename.
            partial = target.with_name(".import-" + uuid.uuid4().hex)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with source, partial.open("xb") as stream:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
            if not check_file(partial, record):
                raise ValueError(f"Extracted data checksum mismatch: {partial}")
            partial.rename(target)


def write_local_mapping(root, catalog, records):
    path = target_path(root, "runtime/current/manifests/published_data_paths.json")
    mapping = {"schema": "somaforge_published_data_paths_v1", "prefixes": [], "files": []}
    if path.exists():
        mapping = json.loads(path.read_text())
        if mapping.get("schema") != "somaforge_published_data_paths_v1":
            raise ValueError(f"Incompatible existing path mapping: {path}")
    rules = {kind: {} for kind in ("prefixes", "files")}
    def add(kind, source, target):
        source, target = str(absolute_alias(source)), str(absolute_alias(target))
        if source in rules[kind] and rules[kind][source] != target:
            raise ValueError(f"Conflicting installed data alias: {source}")
        rules[kind][source] = target
    for kind in rules:
        rows = mapping.get(kind, [])
        if not isinstance(rows, list):
            raise ValueError(f"Published data {kind} mapping must be a list")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"source", "target"}:
                raise ValueError(f"Published data {kind} mapping requires source and target")
            add(kind, row["source"], row["target"])
    for row in catalog.get("source_roots_aliases", []):
        add("prefixes", row["source"], str(alias_target(root, row["target"])))
    prefixes = sorted(((Path(source), Path(target)) for source, target in rules["prefixes"].items()),
                      key=lambda pair: len(pair[0].parts), reverse=True)
    mapped_records = [(record, target_path(root, record["path"])) for record in records]
    mapped_records.extend((record, repository_path(root, record["path"]))
                          for record in catalog.get("repository_files", []))
    for record, target in mapped_records:
        for source in record.get("sources", []):
            original = absolute_alias(source)
            if str(original) in rules["files"]:
                add("files", str(original), str(target))
                continue
            matched = next(((old, new) for old, new in prefixes if original.is_relative_to(old)), None)
            covered = matched is not None and (matched[1] / original.relative_to(matched[0])).resolve() == target
            if not covered:
                add("files", str(original), str(target))
    result = {"schema": mapping["schema"], **{
        kind: [{"source": source, "target": target} for source, target in sorted(rows.items())]
        for kind, rows in rules.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    partial.write_text(json.dumps(result, indent=2) + "\n")
    # Replace only the installer's own derived mapping, never evidence/data.
    partial.replace(path)


def install(catalog, root, names, *, local_archives=None, verify=False):
    if catalog.get("schema") != SCHEMA:
        raise ValueError("Unsupported data-release catalog")
    root = Path(root).expanduser().resolve()
    repository_files = catalog.get("repository_files", [])
    paths = [record["path"] for record in repository_files]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate Git-distributed repository input in catalog")
    for record in repository_files:
        source = repository_path(root, record["path"])
        if not check_file(source, record):
            raise ValueError(f"Missing or corrupt Git-distributed repository input: {record['path']}")
    cache = target_path(root, "runtime/current/cache/data-releases/" + catalog["tag"])
    for row in catalog.get("source_roots_aliases", []):
        absolute_alias(row["source"])
        alias_target(root, row["target"])
    all_records = []
    for name in selected_components(catalog, names):
        component = catalog["components"][name]
        index = acquire(component["index"], cache, local_archives)
        listing = json.loads(index.read_text())
        if listing.get("schema") != "somaforge_data_component_files_v1" or listing["component"] != name:
            raise ValueError(f"Wrong file index for {name}")
        records = listing["files"]
        expected = {record["path"]: record for record in records}
        if len(expected) != len(records):
            raise ValueError(f"Duplicate declared data destination in {name}")
        # Preflight conflicts before extracting any member of this component.
        for relative, record in expected.items():
            target = target_path(root, relative)
            if target.exists() and not check_file(target, record):
                raise FileExistsError(f"Refusing to replace existing data: {target}")
        if not verify:
            observed = set()
            for part in component["archives"]:
                extract_part(acquire(part, cache, local_archives), root, expected, observed)
            if observed != set(expected):
                raise ValueError(f"Release archives omit {len(set(expected)-observed)} files in {name}")
        for relative, record in expected.items():
            if not check_file(target_path(root, relative), record):
                raise ValueError(f"Missing or corrupt installed data: {relative}")
        all_records.extend(records)
        print(f"Verified {name}: {len(records)} immutable files; status={component['status']}", flush=True)
    if not verify:
        write_local_mapping(root, catalog, all_records)
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "fetch", "verify"))
    parser.add_argument("components", nargs="*", help="Component IDs or all")
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/data_releases.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--local-archives", type=Path, help="Use already built assets instead of downloading")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    if args.command == "list":
        for name, component in catalog["components"].items():
            print(f"{name}: {component['bytes']:,} bytes, {component['status']}\n  {component['purpose']}")
        return
    if not args.components:
        parser.error("Select component IDs or all; no data is fetched automatically")
    install(catalog, args.root, args.components, local_archives=args.local_archives, verify=args.command == "verify")


if __name__ == "__main__":
    main()
