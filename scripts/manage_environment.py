#!/usr/bin/env python3
"""Manage independent Miniforge/pip profiles without changing legacy runtimes.

Only ``create`` installs anything. Run one creator per profile at a time;
interrupted installations retain their files and matching ownership state.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlsplit


SCHEMA = "somaforge_environment_profiles_v1"
STATE_SCHEMA = "somaforge_managed_environment_v1"
ROOT = Path(__file__).resolve().parents[1]
LAB_PACKAGES = (
    "isaaclab", "isaaclab_assets", "isaaclab_contrib", "isaaclab_experimental",
    "isaaclab_newton", "isaaclab_ov", "isaaclab_physx", "isaaclab_ppisp",
    "isaaclab_rl", "isaaclab_tasks", "isaaclab_tasks_experimental", "isaaclab_visualizers",
)
BOOTSTRAP_REQUIREMENTS = ("pip==26.2.2", "setuptools==81.0.0", "wheel==0.45.1", "packaging==26.0")
COVERAGE_STARTUP = 'import os\n\nif os.getenv("COVERAGE_PROCESS_START") or os.getenv("COVERAGE_PROCESS_CONFIG"):\n try:\n  import coverage\n except:\n  pass\n else:\n  coverage.process_startup(slug="pth")'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bootstrap_locks(bootstrap, locks):
    """Reject contradictory bootstrap/runtime versions before any installation.

    Profiles use platform-specific complete locks; reading their exact pins and
    artifact filenames requires no installed packaging library.
    """
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    wanted = {canonical(value.split("==", 1)[0]): value.split("==", 1)[1] for value in bootstrap}
    for lock in locks:
        for raw in lock.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "--")):
                continue
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?\s*(.*)$", line)
            if not match or canonical(match[1]) not in wanted:
                continue
            name, rest = canonical(match[1]), match[2]
            if rest.startswith("=="):
                version = re.split(r"[\s;]", rest[2:].strip(), maxsplit=1)[0]
                if "*" in version:
                    raise ValueError(f"Bootstrap package must have an exact runtime lock: {line}")
            elif rest.startswith("@"):
                filename = unquote(urlsplit(rest[1:].strip().split()[0]).path.rsplit("/", 1)[-1])
                if filename.endswith(".whl"):
                    pieces = filename.split("-")
                    if len(pieces) < 5 or canonical(pieces[0]) != name:
                        raise ValueError(f"Invalid bootstrap wheel lock: {line}")
                    version = pieces[1]
                else:
                    stem = re.sub(r"\.(?:tar\.gz|zip|tar\.bz2|tar\.xz)$", "", filename)
                    artifact = stem.rsplit("-", 1)
                    if len(artifact) != 2 or canonical(artifact[0]) != name:
                        raise ValueError(f"Bootstrap source needs a versioned runtime lock: {line}")
                    version = artifact[1]
            else:
                raise ValueError(f"Bootstrap package must have an exact runtime lock: {line}")
            if version != wanted[name]:
                raise ValueError(f"Bootstrap/runtime lock conflict for {name}: {wanted[name]} != {version} ({lock})")


def repository_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid repository path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"Expected a safe repository-relative path: {value!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Repository path escapes the checkout: {value}")
    return target


def find_conda(explicit=None, *, required=False):
    if explicit:
        candidates = [explicit]
    else:
        candidates = [os.environ.get("CONDA_EXE")]
        if os.environ.get("MINIFORGE_ROOT"):
            candidates.append(str(Path(os.environ["MINIFORGE_ROOT"]).expanduser() / "bin/conda"))
        candidates.extend([shutil.which("conda"), str(Path.home() / "miniforge3/bin/conda")])
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    if explicit or required:
        raise RuntimeError("Miniforge conda was not found; pass --conda or set MINIFORGE_ROOT")
    return None


def assert_safe_prefix(root, prefix, *, writing=False):
    anchor = (root / "runtime/environments/conda").resolve()
    if prefix == anchor or not prefix.is_relative_to(anchor):
        raise ValueError("Managed prefixes must be below runtime/environments/conda")
    legacy_root = root / "runtime/environments"
    if legacy_root.is_dir():
        for old in legacy_root.iterdir():
            if old.name != "conda" and not old.name.startswith(".") and old.is_dir():
                old = old.resolve()
                if prefix.is_relative_to(old) or old.is_relative_to(prefix):
                    raise ValueError(f"Managed prefix overlaps a legacy runtime: {old}")
    if writing:
        protected = [Path(sys.prefix).resolve()]
        if os.environ.get("CONDA_PREFIX"):
            protected.append(Path(os.environ["CONDA_PREFIX"]).resolve())
        for active in protected:
            if prefix.is_relative_to(active) or active.is_relative_to(prefix):
                raise ValueError(f"Refusing to install into or overlap the active Python/Conda prefix: {active}")


def make_plan(root, profile, *, catalog=None, conda=None, isaaclab=None):
    root = Path(root).expanduser().resolve()
    catalog = Path(catalog).expanduser().resolve() if catalog else root / "configs/environments/profiles.json"
    payload = json.loads(catalog.read_text())
    if payload.get("schema") != SCHEMA or profile not in payload.get("profiles", {}):
        raise ValueError(f"Unknown environment profile or catalog schema: {profile}")
    spec = payload["profiles"][profile]
    python = spec["python"]
    if not isinstance(python, str) or not re.fullmatch(r"3\.\d+(?:\.\d+)?", python):
        raise ValueError("Profile python must specify Python 3 major/minor, optionally patch")
    relative_prefix = PurePosixPath(spec["prefix"])
    if relative_prefix.parts[:3] != ("runtime", "environments", "conda"):
        raise ValueError("Profile prefix must be runtime/environments/conda/<name>")
    prefix = repository_path(root, spec["prefix"])
    assert_safe_prefix(root, prefix)
    lock_names = [spec["requirements"], *spec.get("source_requirements", [])]
    locks = [repository_path(root, name) for name in lock_names]
    for lock in locks:
        if not lock.is_file():
            raise FileNotFoundError(f"Profile lock file is missing: {lock}")
    local = [repository_path(root, name) for name in spec["local_packages"]]
    for package in local:
        if not (package / "pyproject.toml").is_file() and not (package / "setup.py").is_file():
            raise FileNotFoundError(f"Local Python package is missing: {package}")
    bootstrap = spec.get("bootstrap_requirements", list(BOOTSTRAP_REQUIREMENTS))
    if not isinstance(bootstrap, list) or any(not isinstance(value, str) or
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*==[^ *;]+", value) for value in bootstrap):
        raise ValueError("bootstrap_requirements must be explicitly version-pinned packages")
    validate_bootstrap_locks(bootstrap, locks)
    fingerprint = hashlib.sha256(json.dumps({"profile": profile, "config": spec,
        "bootstrap_requirements": bootstrap,
        "locks": {name: digest(path) for name, path in zip(lock_names, locks)}}, sort_keys=True).encode()).hexdigest()
    executable = str(prefix / "bin/python")
    conda = find_conda(conda)
    pip = [executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
           "--no-deps", "--no-build-isolation"]
    stages = [{"id": "conda", "argv": [conda or "conda", "create", "--yes", "--no-default-packages",
        "--override-channels", "--channel", "conda-forge", "--prefix", str(prefix), f"python={python}", "pip"]}]
    lab = None
    if spec.get("isaaclab_commit"):
        commit = spec["isaaclab_commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError("isaaclab_commit must be an exact 40-character Git SHA")
        source = Path(isaaclab).expanduser().resolve() if isaaclab else root / "runtime/dependencies/isaaclab-v3.0.0"
        snapshot = root / "runtime/environments/.sources" / (profile + "-" + fingerprint[:12]) / "isaaclab"
        names = spec.get("isaaclab_packages", list(LAB_PACKAGES))
        if not isinstance(names, list) or not names or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names):
            raise ValueError("isaaclab_packages must list explicit source-module basenames")
        lab = dict(source=str(source), external=bool(isaaclab), snapshot=str(snapshot), commit=commit, packages=names)
        if not source.exists():
            if isaaclab:
                raise FileNotFoundError(f"Explicit read-only Isaac Lab source is missing: {source}")
            stages.insert(0, {"id": "clone-lab", "argv": ["git", "clone", "--branch", "v3.0.0", "--depth", "1",
                "https://github.com/isaac-sim/IsaacLab.git", str(source)]})
    elif isaaclab:
        raise ValueError("--isaaclab is only valid for a profile declaring isaaclab_commit")
    if bootstrap:
        stages.append({"id": "bootstrap-packaging", "argv": [*pip, *bootstrap]})
    for index, lock in enumerate(locks):
        stages.append({"id": "dependencies" if index == 0 else f"source-dependencies-{index}",
                       "argv": [*pip, "-r", str(lock)]})
    if lab:
        stages.append({"id": "lab-source-snapshot", "action": "git-archive-read-only", **lab})
        stages.append({"id": "lab-packages", "argv": [*pip, *[
            str(Path(lab["snapshot"]) / "source" / name) for name in lab["packages"]]]})
    if local:
        stages.append({"id": "local-packages", "argv": [*pip, *[value for package in local for value in ("-e", str(package))]]})
    state = root / "runtime/environments/.profile-state" / (profile + "-" + hashlib.sha256(str(prefix).encode()).hexdigest()[:16] + ".json")
    return dict(schema=STATE_SCHEMA, root=str(root), catalog=str(catalog), profile=profile, prefix=str(prefix), python=python,
                fingerprint=fingerprint, conda=conda, state_file=str(state), requirements=[str(path) for path in locks],
                local_packages=[str(path) for path in local], lab=lab, config=spec, stages=stages)


def clean_environment(plan):
    env = dict(os.environ)
    inherited = [Path(plan["root"]) / "runtime/environments"]
    for name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        if env.get(name):
            inherited.append(Path(env[name]).resolve())
    if (Path(sys.prefix) / "conda-meta").is_dir() or (Path(sys.prefix) / "pyvenv.cfg").is_file():
        inherited.append(Path(sys.prefix).resolve())
    def scrub(value):
        return os.pathsep.join(part for part in value.split(os.pathsep) if part and
            not any(Path(part).resolve().is_relative_to(old.resolve()) for old in inherited))
    env["PATH"] = scrub(env.get("PATH", os.defpath))
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = scrub(env["LD_LIBRARY_PATH"])
    for name in list(env):
        if name.startswith(("CONDA_", "PIP_")) or name in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
            "PYTHONPYCACHEPREFIX", "VIRTUAL_ENV", "ISAACLAB_PATH",
            "COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG",
            "SOMAFORGE_NEWTON_ENV", "SOMAFORGE_NEWTON_PROFILE", "SOMAFORGE_ISAACLAB_PATH"}:
            env.pop(name, None)
    prefix = Path(plan["prefix"])
    env.update(PATH=str(prefix / "bin") + os.pathsep + env.get("PATH", os.defpath),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_INPUT="1", PIP_CONFIG_FILE=os.devnull,
               PIP_CACHE_DIR=str(Path(plan["root"]) / "runtime/current/cache/pip"),
               SOMAFORGE_ROOT=plan["root"], SOMAFORGE_ENV_PROFILE=plan["profile"], SOMAFORGE_ENV_PREFIX=str(prefix),
               SOMAFORGE_ENV_CATALOG=plan["catalog"])
    if plan["conda"]:
        env["SOMAFORGE_CONDA_EXE"] = plan["conda"]
    else:
        env.pop("SOMAFORGE_CONDA_EXE", None)
    # The new Conda interpreter and installed editable packages supply imports;
    # no base .pth or legacy activation script is inherited.
    python_paths = [plan["root"]]
    if "newton" in plan["config"].get("pins", {}):
        env["SOMAFORGE_NEWTON_ENV"] = str(prefix)
    if plan["lab"]:
        env["ISAACLAB_PATH"] = plan["lab"]["snapshot"]
        env["SOMAFORGE_ISAACLAB_PATH"] = plan["lab"]["snapshot"]
        python_paths.extend(str(Path(plan["lab"]["snapshot"]) / "source" / name) for name in plan["lab"]["packages"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _write_state(path, state):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def _claim(plan):
    root, prefix = Path(plan["root"]), Path(plan["prefix"])
    assert_safe_prefix(root, prefix, writing=True)
    path = Path(plan["state_file"])
    identity = {name: plan[name] for name in ("schema", "root", "profile", "prefix", "fingerprint")}
    if path.exists():
        state = json.loads(path.read_text())
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing ownership state has a different profile, prefix or lock fingerprint")
    else:
        if prefix.exists() or prefix.is_symlink():
            raise FileExistsError(f"Refusing an existing prefix without matching ownership state: {prefix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump({**identity, "status": "creating"}, stream, indent=2)
    return identity


def _run(argv, *, env, cwd=None):
    return subprocess.run(argv, env=env, cwd=cwd, check=True, stdout=sys.stderr, stderr=sys.stderr)


def _verify_lab(lab):
    result = subprocess.run(["git", "-C", lab["source"], "rev-parse", "HEAD"], check=True,
                            text=True, capture_output=True)
    if result.stdout.strip() != lab["commit"]:
        raise ValueError(f"Isaac Lab source does not match pinned commit {lab['commit']}: {lab['source']}")


def _stage_lab(lab):
    _verify_lab(lab)
    result = subprocess.run(["git", "-C", lab["source"], "archive", "--format=tar", lab["commit"]],
                            check=True, capture_output=True)
    target = Path(lab["snapshot"])
    if target.resolve() != target:
        raise ValueError(f"Managed Lab snapshot must not redirect through existing symlinks: {target}")
    excluded = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Unsafe path in pinned Isaac Lab source snapshot")
            if relative.parts[0] in {".agents", ".claude"}:
                excluded.append(member.name)
                continue
            output = target / relative
            if not output.resolve().is_relative_to(target) or output.is_symlink():
                raise ValueError(f"Existing Lab snapshot path redirects outside its owned files: {output}")
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                with archive.extractfile(member) as source:
                    content = source.read()
                if output.exists():
                    if not output.is_file() or output.read_bytes() != content:
                        raise ValueError(f"Managed Lab snapshot differs from the pinned source: {output}")
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("xb") as stream:
                        stream.write(content)
                    output.chmod(member.mode & 0o777)
            else:
                raise ValueError("Pinned Isaac Lab snapshot must contain only regular files/directories")
    for package in lab["packages"]:
        if not (target / "source" / package / "pyproject.toml").is_file():
            raise ValueError(f"Pinned Isaac Lab source package is missing: {package}")
    manifest = dict(schema="somaforge_isaaclab_source_snapshot_v1", commit=lab["commit"],
                    source=lab["source"], excluded_development_paths=sorted(excluded))
    manifest_path = target / ".somaforge-source-snapshot.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("commit") != manifest["commit"] or previous.get("excluded_development_paths") != manifest["excluded_development_paths"]:
            raise ValueError("Managed source snapshot manifest differs from the pinned commit/exclusions")
    else:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)


def create(plan):
    if plan["lab"] and Path(plan["lab"]["source"]).exists():
        _verify_lab(plan["lab"])
    identity = _claim(plan)
    conda = find_conda(plan["conda"], required=True)
    env = clean_environment(plan)
    state_file = Path(plan["state_file"])
    completed = []
    try:
        for stage in plan["stages"]:
            if stage["id"] == "lab-source-snapshot":
                _stage_lab(plan["lab"])
            else:
                argv = list(stage["argv"])
                if stage["id"] == "conda":
                    argv[0] = conda
                    prefix = Path(plan["prefix"])
                    if (prefix / "bin/python").is_file() and (prefix / "conda-meta/history").is_file():
                        completed.append(stage["id"])
                        continue
                if stage["id"] == "clone-lab":
                    Path(plan["lab"]["source"]).parent.mkdir(parents=True, exist_ok=True)
                    if Path(plan["lab"]["source"]).exists():
                        _verify_lab(plan["lab"])
                    else:
                        _run(argv, env=env)
                        _verify_lab(plan["lab"])
                else:
                    _run(argv, env=env, cwd=plan["root"])
            completed.append(stage["id"])
            _write_state(state_file, {**identity, "status": "creating", "completed": completed})
        if plan["lab"]:
            _verify_lab(plan["lab"])
        _write_state(state_file, {**identity, "status": "installed", "completed": completed})
        audit = check(plan)
        if not audit["ok"]:
            raise ValueError("Post-install environment audit failed: " + "; ".join(audit["errors"]))
    except BaseException as exc:
        _write_state(state_file, {**identity, "status": "interrupted", "completed": completed, "error": str(exc)})
        raise
    return {**identity, "status": "installed", "completed": completed}


def recorded_startup(site, package, paths):
    """Known startup hooks must belong to an unmodified wheel in this prefix."""
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    for record in site.glob("*.dist-info/RECORD"):
        metadata = record.with_name("METADATA")
        if not metadata.is_file() or not any(line.startswith("Name: ") and
                normalize(line[6:]) == normalize(package) for line in metadata.read_text().splitlines()):
            continue
        rows = {row[0]: row[1] for row in csv.reader(io.StringIO(record.read_text())) if len(row) >= 2}
        for relative in paths:
            target = site / relative
            if not target.is_file() or not target.resolve().is_relative_to(site.resolve()):
                return False
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest()).decode().rstrip("=")
            if rows.get(relative) != expected:
                return False
        return True
    return False


def inspect_pth(plan):
    prefix = Path(plan["prefix"])
    errors = []
    if (prefix / "pyvenv.cfg").exists():
        errors.append("Managed Conda prefix must not inherit a venv/base pyvenv.cfg")
    allowed = [prefix, *(Path(path) for path in plan["local_packages"])]
    source_roots = [Path(path) for path in plan["local_packages"]]
    if plan["lab"]:
        source_roots.append(Path(plan["lab"]["snapshot"]) / "source")
    for site in prefix.glob("lib/python*/site-packages"):
        if (site / "sitecustomize.py").exists() or (site / "usercustomize.py").exists():
            errors.append(f"Unexpected Python startup customization in {site}")
        for path in site.glob("*.pth"):
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "addsitedir" in line:
                    errors.append(f"External site inheritance is forbidden: {path}")
                elif line.startswith(("import ", "import\t")):
                    finder = re.fullmatch(r"import (__editable___[A-Za-z0-9_]+_finder); \1\.install\(\)", line)
                    if finder:
                        module = site / (finder[1] + ".py")
                        try:
                            if not module.resolve().is_relative_to(prefix):
                                raise ValueError("finder redirects outside the prefix")
                            values = {}
                            for node in ast.parse(module.read_text()).body:
                                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                                    values[node.target.id] = node.value
                                elif isinstance(node, ast.Assign):
                                    for name in node.targets:
                                        if isinstance(name, ast.Name):
                                            values[name.id] = node.value
                            mapping = ast.literal_eval(values["MAPPING"])
                            namespaces = ast.literal_eval(values["NAMESPACES"])
                            if not isinstance(mapping, dict) or not mapping or not isinstance(namespaces, dict):
                                raise ValueError("finder mappings are not literal dictionaries")
                            paths = list(mapping.values())
                            for group in namespaces.values():
                                if not isinstance(group, list):
                                    raise ValueError("finder namespaces are not literal path lists")
                                paths.extend(group)
                            for value in paths:
                                if not isinstance(value, str) or not Path(value).is_absolute() or not any(
                                        Path(value).resolve().is_relative_to(root) for root in source_roots):
                                    raise ValueError("finder maps to an unconfigured source: " + str(value))
                        except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
                            errors.append(f"Invalid editable finder in {path}: {exc}")
                    elif line == "import cmeel_pth":
                        if not recorded_startup(site, "cmeel", [path.name, "cmeel_pth.py"]):
                            errors.append(f"Missing/unverified owned cmeel startup module: {path}")
                    elif line == "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; enabled and __import__('_distutils_hack').add_shim();":
                        if not recorded_startup(site, "setuptools", [path.name, "_distutils_hack/__init__.py"]):
                            errors.append(f"Missing/unverified setuptools startup module: {path}")
                    elif line.startswith("import sys; exec("):
                        try:
                            statements = ast.parse(line).body
                            valid = (len(statements) == 2 and isinstance(statements[0], ast.Import) and
                                len(statements[0].names) == 1 and statements[0].names[0].name == "sys" and
                                statements[0].names[0].asname is None and isinstance(statements[1], ast.Expr) and
                                isinstance(statements[1].value, ast.Call) and
                                isinstance(statements[1].value.func, ast.Name) and statements[1].value.func.id == "exec" and
                                len(statements[1].value.args) == 1 and not statements[1].value.keywords and
                                ast.literal_eval(statements[1].value.args[0]) == COVERAGE_STARTUP)
                        except (SyntaxError, ValueError):
                            valid = False
                        if not valid or not recorded_startup(site, "coverage", [path.name, "coverage/__init__.py"]):
                            errors.append(f"Unrecognized/unverified coverage startup hook: {path}")
                    else:
                        errors.append(f"Unrecognized executable Python startup line in {path}: {line}")
                else:
                    target = (site / line).resolve()
                    if not any(target.is_relative_to(parent) for parent in allowed):
                        errors.append(f"External path in {path}: {target}")
    return errors


def ownership_errors(plan):
    path = Path(plan["state_file"])
    if not path.is_file():
        return ["Matching managed ownership state is missing"]
    state = json.loads(path.read_text())
    if any(state.get(name) != plan[name] for name in ("schema", "root", "profile", "prefix", "fingerprint")):
        return ["Managed ownership state does not match profile/lock fingerprint"]
    if state.get("status") != "installed":
        return ["Managed environment installation is not complete: " + str(state.get("status"))]
    return []


CHECK_CODE = r'''
import ast, importlib.metadata as md, importlib.util, json, pathlib, re, sys, tomllib
from urllib.parse import unquote, urlsplit
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename, parse_sdist_filename
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename, parse_sdist_filename
request = json.load(sys.stdin)
prefix = pathlib.Path(request['prefix']).resolve()
errors = []
if pathlib.Path(sys.prefix).resolve() != prefix or pathlib.Path(sys.base_prefix).resolve() != prefix:
    errors.append('Python prefix/base_prefix does not identify the independent Conda environment')
for entry in sys.path:
    path = pathlib.Path(entry).resolve()
    if 'site-packages' in path.parts and not path.is_relative_to(prefix):
        errors.append('Inherited external site-packages: ' + str(path))
dists = {}
for dist in md.distributions():
    name = canonicalize_name(dist.metadata['Name'])
    if name in dists and dists[name].version != dist.version:
        errors.append('Multiple installed versions of ' + name)
    dists[name] = dist
versions = {name: dist.version for name, dist in dists.items()}
for module, providers in {
    'cv2': ('opencv-python', 'opencv-contrib-python', 'opencv-python-headless',
            'opencv-contrib-python-headless', 'opencv-python-headless-noffmpeg'),
    'onnxruntime': ('onnxruntime', 'onnxruntime-gpu', 'onnxruntime-openvino'),
}.items():
    installed_providers = sorted(name for name in providers if name in dists)
    if len(installed_providers) > 1:
        errors.append('Conflicting providers for ' + module + ': ' + ', '.join(installed_providers))
extras = {}
locked = {}
for lock in request.get('locks', []):
    for raw in lock['text'].splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith(('--index-url ', '--extra-index-url ', '--find-links ')):
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(req.name)
        extras.setdefault(name, set()).update(req.extras)
        expected = None
        if req.url:
            if req.url.startswith('git+'):
                if name not in dists:
                    errors.append('Locked source distribution missing: ' + name)
                if not any(canonicalize_name(pin['package']) == name for pin in request['source_pins']):
                    errors.append('Source lock entry lacks an exact Git commit: ' + name)
                continue  # Exact commit is checked separately against direct_url.json.
            filename = unquote(urlsplit(req.url).path.rsplit('/', 1)[-1])
            parsed = parse_wheel_filename(filename) if filename.endswith('.whl') else parse_sdist_filename(filename)
            if canonicalize_name(parsed[0]) != name:
                errors.append('Lock URL distribution name mismatch: ' + name)
            expected = str(parsed[1])
        elif len(list(req.specifier)) == 1:
            pin = next(iter(req.specifier))
            if pin.operator == '==' and '*' not in pin.version:
                expected = pin.version
        if expected is None:
            errors.append('Lock entry is not exact: ' + line)
        elif name in locked and locked[name] != expected:
            errors.append('Conflicting lock versions: ' + name)
        else:
            locked[name] = expected
            if versions.get(name) != expected:
                errors.append('Lock mismatch: ' + name + ' expected ' + expected + ', got ' + str(versions.get(name)))
for raw in request.get('bootstrap_requirements', []):
    req = Requirement(raw)
    name = canonicalize_name(req.name)
    if name not in versions or not req.specifier.contains(versions[name], prereleases=True):
        errors.append('Bootstrap distribution mismatch: ' + raw)

def module_origin(name, roots):
    # Only inspect top-level modules: find_spec does not import their CUDA/runtime code.
    if '.' in name:
        return
    spec = importlib.util.find_spec(name)
    paths = ([spec.origin] if spec and spec.origin and spec.origin not in ('built-in', 'frozen') else [])
    if spec and spec.submodule_search_locations:
        paths.extend(spec.submodule_search_locations)
    if not paths or any(not any(pathlib.Path(path).resolve().is_relative_to(root) for root in roots) for path in paths):
        errors.append('Module origin is outside configured source: ' + name)

source_roots = [pathlib.Path(path).resolve() for path in request.get('local_packages', [])]
if request.get('lab'):
    source_roots.append(pathlib.Path(request['lab']['snapshot']) / 'source')
for site in prefix.glob('lib/python*/site-packages') if source_roots else []:
    for finder in site.glob('__editable___*_finder.py'):
        for node in ast.parse(finder.read_text()).body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'MAPPING':
                for module in ast.literal_eval(node.value):
                    module_origin(module, source_roots)
            elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MAPPING' for target in node.targets):
                for module in ast.literal_eval(node.value):
                    module_origin(module, source_roots)

for package in request.get('local_packages', []):
    root = pathlib.Path(package).resolve()
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    name = canonicalize_name(project['name'])
    dist = dists.get(name)
    if not dist or ('version' in project and dist.version != project['version']):
        errors.append('Local distribution missing/version mismatch: ' + name)
        continue
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    url = urlsplit(direct.get('url', ''))
    if url.scheme != 'file' or pathlib.Path(unquote(url.path)).resolve() != root or not direct.get('dir_info', {}).get('editable'):
        errors.append('Local distribution is not editable from the configured checkout: ' + name)
    top = dist.read_text('top_level.txt') or ''
    for module in top.splitlines():
        if module.strip():
            module_origin(module.strip(), [root])

lab = request.get('lab')
if lab:
    for module in lab['packages']:
        root = pathlib.Path(lab['snapshot']) / 'source' / module
        project_file = root / 'pyproject.toml'
        if not project_file.is_file():
            errors.append('Pinned Lab source package missing: ' + module)
            continue
        project = tomllib.loads(project_file.read_text()).get('project', {})
        name = canonicalize_name(project.get('name', module))
        if name not in dists or ('version' in project and versions[name] != project['version']):
            errors.append('Lab distribution missing/version mismatch: ' + name)
        module_origin(module, [root])

# Propagate requested extras through all active dependencies until the graph is stable.
# This includes extras introduced by project metadata, such as smart_open[s3].
changed = True
while changed:
    changed = False
    for name, dist in dists.items():
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in {'', *extras.get(name, set())}):
                continue
            requested = extras.setdefault(canonicalize_name(req.name), set())
            before = len(requested)
            requested.update(req.extras)
            changed |= len(requested) != before
for name, version in request['config'].get('pins', {}).items():
    if versions.get(canonicalize_name(name)) != str(version):
        errors.append('Pin mismatch: ' + name + ' expected ' + str(version) + ', got ' + str(versions.get(canonicalize_name(name))))
def key(requirement):
    req = Requirement(requirement)
    return canonicalize_name(req.name) + ('[' + ','.join(sorted(req.extras)) + ']' if req.extras else '') + str(req.specifier)
conflicts, allowed = [], []
providers = request['config'].get('provider_substitutions', [])
provider_ok = {}
for rule in providers:
    name, provider = canonicalize_name(Requirement(rule['requirement']).name), canonicalize_name(rule['provider'])
    valid = bool(rule.get('reason')) and name not in dists and versions.get(provider) == rule['provider_version']
    spec = importlib.util.find_spec(rule['module'])
    origin = pathlib.Path(spec.origin).resolve() if spec and spec.origin else None
    valid = valid and origin is not None and origin.is_relative_to(prefix)
    provider_ok[id(rule)] = valid
    if not valid:
        errors.append('Invalid provider substitution for ' + rule['requirement'])
for name, dist in dists.items():
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in {'', *extras.get(name, set())}):
            continue
        installed = versions.get(canonicalize_name(req.name))
        if installed is not None and (not req.specifier or req.specifier.contains(installed, prereleases=True)):
            continue
        conflict = dict(package=name, version=dist.version, requirement=key(raw), installed=installed)
        accepted = False
        for rule in request['config'].get('dependency_overrides', []):
            if (rule.get('reason') and canonicalize_name(rule['package']) == name and rule['version'] == dist.version
                    and key(rule['requirement']) == conflict['requirement'] and rule['installed'] == installed):
                accepted = True
        for rule in providers:
            if (provider_ok[id(rule)] and canonicalize_name(rule['package']) == name and rule['version'] == dist.version
                    and key(rule['requirement']) == conflict['requirement'] and installed is None):
                accepted = True
        (allowed if accepted else conflicts).append(conflict)
for row in request['source_pins']:
    dist = dists.get(canonicalize_name(row['package']))
    direct = json.loads(dist.read_text('direct_url.json') or '{}') if dist else {}
    if direct.get('vcs_info', {}).get('commit_id') != row['commit']:
        errors.append('Source commit mismatch: ' + row['package'])
print(json.dumps(dict(versions=versions, locked_count=len(locked), requested_extras={name: sorted(value) for name, value in extras.items() if value}, conflicts=conflicts, allowed_conflicts=allowed, errors=errors)))
'''


def source_pins(plan):
    pins = []
    for path in plan["requirements"]:
        for match in re.finditer(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*@\s*git\+\S+@([0-9a-fA-F]{40})(?:\s|$|#)", Path(path).read_text()):
            pins.append(dict(package=match[1], commit=match[2]))
    return pins


def check(plan):
    executable = Path(plan["prefix"]) / "bin/python"
    errors = [*inspect_pth(plan), *ownership_errors(plan)]
    if not executable.is_file():
        errors.append(f"Profile Python is missing: {executable}")
    result = dict(profile=plan["profile"], prefix=plan["prefix"], fingerprint=plan["fingerprint"], errors=errors)
    if errors:
        return {**result, "ok": False}
    env = clean_environment(plan)
    request = dict(prefix=plan["prefix"], config=plan["config"], source_pins=source_pins(plan),
        locks=[dict(path=path, text=Path(path).read_text()) for path in plan["requirements"]],
        bootstrap_requirements=plan["config"].get("bootstrap_requirements", list(BOOTSTRAP_REQUIREMENTS)),
        local_packages=plan["local_packages"], lab=plan["lab"])
    metadata = subprocess.run([str(executable), "-c", CHECK_CODE], input=json.dumps(request),
        text=True, capture_output=True, env=env, cwd=plan["root"])
    pip = subprocess.run([str(executable), "-m", "pip", "check", "--disable-pip-version-check"],
        text=True, capture_output=True, env=env, cwd=plan["root"])
    result["pip_check"] = dict(returncode=pip.returncode, stdout=pip.stdout, stderr=pip.stderr)
    if metadata.returncode:
        errors.append("Metadata audit failed: " + metadata.stderr.strip())
    else:
        audit = json.loads(metadata.stdout)
        result.update(audit)
        errors.extend(audit["errors"])
        if audit["conflicts"]:
            errors.append("Unapproved dependency conflicts: " + json.dumps(audit["conflicts"], sort_keys=True))
        if pip.returncode:
            unmatched = []
            for line in pip.stdout.splitlines():
                matched = False
                for row in audit["allowed_conflicts"]:
                    head = row["package"] + " " + row["version"]
                    dep = re.split(r"[\[<>=!~ ]", row["requirement"], maxsplit=1)[0]
                    expected = (head + " requires " + dep + ", which is not installed." if row["installed"] is None else
                        head + " has requirement " + row["requirement"] + ", but you have " + dep + " " + row["installed"] + ".")
                    if line.casefold() == expected.casefold():
                        matched = True
                        break
                if not matched:
                    unmatched.append(line)
            if unmatched or not pip.stdout.strip():
                errors.append("Unapproved pip check output: " + "\n".join(unmatched or [pip.stderr]))
            if pip.stderr.strip():
                errors.append("Unapproved pip check stderr: " + pip.stderr.strip())
    return {**result, "errors": errors, "ok": not errors}


def run(plan, command):
    audit = check(plan)
    if not audit["ok"]:
        raise ValueError("; ".join(audit["errors"]))
    executable = Path(plan["prefix"]) / "bin/python"
    if not executable.is_file():
        raise FileNotFoundError(f"Profile Python is missing: {executable}")
    if not command:
        raise ValueError("run requires an explicit command after --")
    argv = list(command)
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", argv[0]):
        argv[0] = str(executable)
    return subprocess.run(argv, env=clean_environment(plan), cwd=plan["root"]).returncode


def shell_environment(plan):
    """Emit a safely quoted activation script using the managed run environment."""
    import shlex
    env = clean_environment(plan)
    lines = ["unset " + name for name in os.environ if name not in env]
    lines.extend("export " + name + "=" + shlex.quote(value)
                 for name, value in env.items() if os.environ.get(name) != value)
    lines.append("hash -r")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "create", "check", "run", "shell"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--conda")
    parser.add_argument("--isaaclab", type=Path, help="Read-only pinned Lab source; installation builds a separate snapshot")
    arguments = sys.argv[1:]
    split = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:split])
    command = arguments[split + 1:] if split < len(arguments) else []
    if command and args.command != "run":
        parser.error("Only run accepts a command after --")
    try:
        plan = make_plan(args.root, args.profile, catalog=args.catalog, conda=args.conda, isaaclab=args.isaaclab)
        if args.command == "shell":
            print(shell_environment(plan))
            return 0
        if args.command == "run":
            return run(plan, command)
        result = plan if args.command == "plan" else create(plan) if args.command == "create" else check(plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok", True) else 1
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
