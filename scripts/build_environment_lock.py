#!/usr/bin/env python3
"""Build the Linux CPython 3.12 pip lock from official GA and a pip report.

No uv executable is used. The upstream lock is read as immutable TOML data;
the selected dependency graph and project resolution are recorded separately.
Run seed, resolve its generated requirements with pip --dry-run --report, then
finish. This tool never installs or modifies a Python environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
LAB_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
LOCAL = ("packages/somaforge_core", "packages/contact_solver", "packages/generator",
         "packages/motion_edit", "src/holosoma")


def official_packages(lock):
    by_name = defaultdict(list)
    for package in lock["package"]:
        by_name[package["name"]].append(package)
    seen, selected = set(), {}
    pending = [({"name": "isaaclab-dev"}, ("test", "isaacsim"))]
    while pending:
        edge, extras = pending.pop()
        if edge.get("marker") and not Marker(edge["marker"]).evaluate({"extra": ""}):
            continue
        candidates = [p for p in by_name[edge["name"]]
                      if (not edge.get("version") or p.get("version") == edge["version"])
                      and (not edge.get("source") or p["source"] == edge["source"])
                      and (not p.get("resolution-markers") or any(
                          Marker(marker).evaluate({"extra": ""}) for marker in p["resolution-markers"]))]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous active package in official lock: {edge}")
        package = candidates[0]
        key = (package["name"], package.get("version"), extras)
        if key in seen:
            continue
        seen.add(key)
        if package["name"] in selected and selected[package["name"]] != package:
            raise ValueError(f"Multiple active versions: {package['name']}")
        selected[package["name"]] = package
        dependencies = list(package.get("dependencies", []))
        for extra in extras:
            dependencies += package.get("optional-dependencies", {}).get(extra, [])
        pending.extend((dependency, tuple(dependency.get("extra", []))) for dependency in dependencies)
    return selected


def wheel_record(package):
    ranks = {tag: number for number, tag in enumerate(sys_tags())}
    candidates = []
    for wheel in package.get("wheels", []):
        filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
        _, _, _, tags = parse_wheel_filename(filename)
        compatible = tags & ranks.keys()
        if compatible:
            candidates.append((min(ranks[tag] for tag in compatible), wheel["url"], wheel))
    if not candidates:
        if not package.get("sdist"):
            raise ValueError(f"No compatible locked artifact: {package['name']}=={package.get('version')}")
        wheel = package["sdist"]
    else:
        wheel = min(candidates, key=lambda row: row[:2])[2]
    return dict(name=package["name"], version=package["version"], url=wheel["url"],
                sha256=wheel["hash"].removeprefix("sha256:"))


def requirement_files(path):
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            yield from requirement_files(path.parent / line[3:])
        else:
            yield line


def seed(lab, output):
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Generate this lock with Linux x86_64 CPython 3.12")
    head = subprocess.run(['git', '-C', str(lab), 'rev-parse', 'HEAD'],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != LAB_COMMIT:
        raise ValueError(f"Expected pinned Isaac Lab commit {LAB_COMMIT}, got {head}")
    raw = (lab / "uv.lock").read_bytes()
    committed = subprocess.run(['git', '-C', str(lab), 'show', f'{LAB_COMMIT}:uv.lock'],
                               check=True, capture_output=True).stdout
    if raw != committed:
        raise ValueError("Isaac Lab uv.lock differs from the pinned Git source")
    selected = official_packages(tomllib.loads(raw.decode()))
    records = {name: wheel_record(package) for name, package in selected.items()
               if "registry" in package["source"]}
    # One provider owns each ONNX/OpenCV namespace. Isaac Sim requires GPU ONNX
    # and its official OpenCV build; exact API substitutions live in profiles.
    records.pop("onnxruntime", None)
    overrides = {}
    for value in requirement_files(ROOT / "configs/isaaclab-ga-overlay-requirements.txt"):
        requirement = Requirement(value)
        overrides[canonicalize_name(requirement.name)] = value
    external = []
    projects = [tomllib.loads((ROOT / path / "pyproject.toml").read_text())["project"] for path in LOCAL]
    local_names = {canonicalize_name(project["name"]) for project in projects}
    for project in projects:
        for value in project.get("dependencies", []):
            requirement = Requirement(value)
            name = canonicalize_name(requirement.name)
            if name in local_names or name in {"onnxruntime", "opencv-python"}:
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            external.append(value)
    external += list(overrides.values()) + ["onnxruntime-gpu==1.26.0", "pip==26.2.2", "wheel==0.45.1", "setuptools==81.0.0"]
    for name in overrides:
        records.pop(name, None)
    # Constraints retain every upstream pin but never request conflicting Sim
    # roots during pip resolution of the added/changed dependency closure.
    constraints = []
    for name, record in sorted(records.items()):
        if name in {"torch", "torchvision"}:
            constraints.append(f"{name} @ {record['url']}#sha256={record['sha256']}")
        else:
            constraints.append(f"{name}=={record['version']}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "main-constraints.txt").write_text("\n".join(constraints) + "\n")
    (output / "main-roots.txt").write_text("\n".join(sorted(set(external))) + "\n")
    (output / "main-seed.json").write_text(json.dumps(dict(
        schema="somaforge_pip_lock_seed_v1", isaaclab_commit=LAB_COMMIT,
        upstream_lock_sha256=hashlib.sha256(raw).hexdigest(), extras=["test", "isaacsim"],
        records=records, overridden=overrides), indent=2) + "\n")
    print(f"Selected {len(records)} upstream distributions; resolve {len(set(external))} project requirements")


def finish(seed_file, report_file, destination):
    data = json.loads(seed_file.read_text())
    records = data["records"]
    report = json.loads(report_file.read_text())
    for item in report["install"]:
        name = canonicalize_name(item["metadata"]["name"])
        if name in records and records[name]['version'] != item['metadata']['version']:
            raise ValueError(f"Project resolution changed a retained upstream pin: {name}")
        download = item["download_info"]
        if not download["url"].startswith("https://"):
            raise ValueError(f"Expected an immutable HTTPS distribution artifact: {download['url']}")
        digest = download["archive_info"]["hashes"]["sha256"]
        records[name] = dict(name=name, version=item["metadata"]["version"],
                             url=download["url"], sha256=digest)
    if "onnxruntime" in records or "opencv-python" in records:
        raise ValueError("Conflicting ONNX/OpenCV API providers must not co-install")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SomaForge main: Linux x86_64 / CPython 3.12 / Miniforge + pip.",
             f"# Official Isaac Lab commit: {data['isaaclab_commit']}",
             f"# Official lock SHA256: {data['upstream_lock_sha256']}",
             "# Complete external closure. Install with --no-deps; profiles.json audits exact upstream overrides.",
             "# Local project and Isaac Lab packages are installed separately by manage_environment.py."]
    for name, record in sorted(records.items()):
        if not record['url'].startswith('https://') or not re.fullmatch(r'[0-9a-f]{64}', record['sha256']):
            raise ValueError(f"Expected an HTTPS artifact with a SHA256 digest: {name}")
        lines += [f"# {name}=={record['version']}",
                  f"{name} @ {record['url']}#sha256={record['sha256']}"]
    destination.write_text("\n".join(lines) + "\n")
    provenance = {key: value for key, value in data.items() if key != "records"}
    provenance.update(package_count=len(records), records=records,
                      pip_report_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest())
    destination.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Locked {len(records)} external distributions: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("seed")
    first.add_argument("--isaaclab", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("finish")
    second.add_argument("--seed", type=Path, required=True)
    second.add_argument("--report", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seed":
        seed(args.isaaclab, args.output)
    else:
        finish(args.seed, args.report, args.output)


if __name__ == "__main__":
    main()
