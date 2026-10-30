"""Independent profile planning/ownership without installing or importing a GPU stack."""
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manager(monkeypatch):
    spec = importlib.util.spec_from_file_location("_test_manage_environment", ROOT / "scripts/manage_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    config = root / "configs/environments"
    config.mkdir(parents=True)
    (config / "fixture.lock.txt").write_text("dependency==1.2.3\n")
    (config / "fixture-sources.lock.txt").write_text("source @ git+https://example.test/repo.git@" + "a" * 40 + "\n")
    package = root / "packages/example"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text('[project]\nname="example"\nversion="1.0"\n')
    profile = dict(python="3.12", prefix="runtime/environments/conda/fixture",
        requirements="configs/environments/fixture.lock.txt", local_packages=["packages/example"])
    catalog = config / "profiles.json"
    catalog.write_text(json.dumps(dict(schema="somaforge_environment_profiles_v1", profiles={"fixture": profile})))
    conda = tmp_path / "miniforge/bin/conda"
    conda.parent.mkdir(parents=True)
    conda.write_text("#!/bin/sh\nexit 0\n")
    conda.chmod(0o755)
    monkeypatch.setenv("CONDA_EXE", str(conda))
    return SimpleNamespace(root=root, catalog=catalog, profile=profile, conda=conda)


def update_catalog(project, **fields):
    project.profile.update(fields)
    project.catalog.write_text(json.dumps(dict(schema="somaforge_environment_profiles_v1", profiles={"fixture": project.profile})))


def fake_installer(manager, monkeypatch, *, fail_stage=False):
    calls = []
    monkeypatch.setattr(manager, "check", lambda plan: {"ok": True, "errors": []})
    def install(argv, *, env, cwd=None):
        calls.append((list(argv), dict(env)))
        if "create" in argv:
            prefix = Path(argv[argv.index("--prefix") + 1])
            (prefix / "bin").mkdir(parents=True)
            (prefix / "bin/python").write_text("fixture interpreter, never executed")
            (prefix / "conda-meta").mkdir()
            (prefix / "conda-meta/history").write_text("fixture")
        elif fail_stage:
            raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(manager, "_run", install)
    return calls


def test_plan_is_read_only_and_never_uses_uv_or_a_legacy_python(manager, project, monkeypatch):
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: pytest.fail("plan cannot execute commands"))
    plan = manager.make_plan(project.root, "fixture")
    assert plan["conda"] == str(project.conda)
    assert not Path(plan["prefix"]).exists()
    assert not Path(plan["state_file"]).exists()
    commands = [stage["argv"] for stage in plan["stages"]]
    assert commands[0][:2] == [str(project.conda), "create"]
    assert all(command[0] == str(Path(plan["prefix"]) / "bin/python") for command in commands[1:])
    assert all("uv" not in command for command in commands)
    assert all("--no-deps" in command and "--no-build-isolation" in command for command in commands[1:])


def test_create_claims_only_new_prefix_and_same_lock_can_resume(manager, project, monkeypatch):
    update_catalog(project, source_requirements=["configs/environments/fixture-sources.lock.txt"])
    plan = manager.make_plan(project.root, "fixture")
    calls = fake_installer(manager, monkeypatch)
    assert manager.create(plan)["status"] == "installed"
    assert len(calls) == 5
    assert manager.create(plan)["status"] == "installed"
    assert len(calls) == 9  # Existing owned Conda interpreter is preserved.
    assert all(str(project.root / "runtime/environments/conda/fixture") in " ".join(argv) for argv, _ in calls)
    assert manager.source_pins(plan) == [dict(package="source", commit="a" * 40)]


def test_interrupted_owned_install_retains_state_and_can_resume(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    fake_installer(manager, monkeypatch, fail_stage=True)
    with pytest.raises(subprocess.CalledProcessError):
        manager.create(plan)
    assert json.loads(Path(plan["state_file"]).read_text())["status"] == "interrupted"
    before = (Path(plan["prefix"]) / "bin/python").read_bytes()
    calls = fake_installer(manager, monkeypatch)
    manager.create(plan)
    assert calls and all("create" not in argv for argv, _ in calls)
    assert (Path(plan["prefix"]) / "bin/python").read_bytes() == before


def test_create_audits_completed_install_and_retains_failed_prefix(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    fake_installer(manager, monkeypatch)
    def audit(selected):
        assert json.loads(Path(selected["state_file"]).read_text())["status"] == "installed"
        return {"ok": False, "errors": ["Lock mismatch: fixture"]}
    monkeypatch.setattr(manager, "check", audit)
    with pytest.raises(ValueError, match="Post-install environment audit failed"):
        manager.create(plan)
    state = json.loads(Path(plan["state_file"]).read_text())
    assert state["status"] == "interrupted" and "Lock mismatch" in state["error"]
    assert (Path(plan["prefix"]) / "bin/python").exists()


def test_existing_unowned_prefix_and_changed_lock_are_never_modified(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    prefix = Path(plan["prefix"])
    prefix.mkdir(parents=True)
    note = prefix / "user-file"
    note.write_text("keep")
    monkeypatch.setattr(manager, "_run", lambda *a, **k: pytest.fail("Cannot install into an unowned prefix"))
    with pytest.raises(FileExistsError, match="ownership"):
        manager.create(plan)
    assert note.read_text() == "keep"
    assert not Path(plan["state_file"]).exists()
    # Ownership may be claimed only for a separately named fresh profile.
    update_catalog(project, prefix="runtime/environments/conda/another")
    fresh = manager.make_plan(project.root, "fixture")
    manager._claim(fresh)
    (project.root / project.profile["requirements"]).write_text("dependency==2.0\n")
    changed = manager.make_plan(project.root, "fixture")
    with pytest.raises(ValueError, match="fingerprint"):
        manager.create(changed)


@pytest.mark.parametrize("key", ["CONDA_PREFIX", "sys.prefix"])
def test_active_prefix_is_protected_even_with_matching_state(manager, project, monkeypatch, key):
    plan = manager.make_plan(project.root, "fixture")
    if key == "CONDA_PREFIX":
        monkeypatch.setenv(key, plan["prefix"])
    else:
        monkeypatch.setattr(manager, "sys", SimpleNamespace(prefix=plan["prefix"]))
    with pytest.raises(ValueError, match="active Python/Conda"):
        manager.create(plan)
    assert not Path(plan["state_file"]).exists()


@pytest.mark.parametrize("prefix", ["runtime/environments/old", "../escape", "/outside"])
def test_profile_cannot_target_legacy_or_external_prefix(manager, project, prefix):
    update_catalog(project, prefix=prefix)
    with pytest.raises(ValueError):
        manager.make_plan(project.root, "fixture")


def test_conda_locator_obeys_explicit_miniforge_and_does_not_install_tools(manager, project, monkeypatch):
    assert manager.find_conda(str(project.conda)) == str(project.conda)
    monkeypatch.delenv("CONDA_EXE")
    monkeypatch.setenv("MINIFORGE_ROOT", str(project.conda.parent.parent))
    assert manager.find_conda() == str(project.conda)
    with pytest.raises(RuntimeError, match="Miniforge"):
        manager.find_conda(str(project.conda.parent / "missing"))


def test_clean_run_paths_drop_legacy_shell_and_use_only_new_python(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    old = project.root / "runtime/environments/isaaclab-ga-3.0.0-base"
    monkeypatch.setenv("PYTHONPATH", str(old / "lib/site-packages"))
    monkeypatch.setenv("PYTHONHOME", str(old))
    monkeypatch.setenv("CONDA_PREFIX", str(old))
    monkeypatch.setenv("PATH", str(old / "bin") + os.pathsep + "/usr/bin")
    env = manager.clean_environment(plan)
    assert env["PYTHONPATH"] == str(project.root)
    assert env["SOMAFORGE_CONDA_EXE"] == str(project.conda)
    assert str(project.conda.parent) not in env["PATH"].split(os.pathsep)
    assert "PYTHONHOME" not in env and "CONDA_PREFIX" not in env
    assert str(old) not in env["PATH"]
    executable = Path(plan["prefix"]) / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture")
    calls = []
    monkeypatch.setattr(manager, "check", lambda selected: {"ok": True, "errors": []})
    monkeypatch.setattr(manager.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(returncode=7))
    assert manager.run(plan, ["python", "-m", "example"]) == 7
    assert calls[0][0][0] == str(executable)
    assert calls[0][1]["env"]["PYTHONPATH"] == str(project.root)


def test_pip_configuration_cannot_redirect_installs_to_an_existing_environment(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_CONFIG_FILE", "PIP_INDEX_URL", "PYTHONUSERBASE"):
        monkeypatch.setenv(key, "/old/active/environment")
    env = manager.clean_environment(plan)
    assert not {"PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_INDEX_URL", "PYTHONUSERBASE"} & env.keys()
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PIP_NO_INPUT"] == env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["PIP_CACHE_DIR"] == str(project.root / "runtime/current/cache/pip")
    bootstrap = next(stage for stage in plan["stages"] if stage["id"] == "bootstrap-packaging")
    dependencies = next(stage for stage in plan["stages"] if stage["id"] == "dependencies")
    assert plan["stages"].index(bootstrap) < plan["stages"].index(dependencies)
    assert all(pin in bootstrap["argv"] for pin in manager.BOOTSTRAP_REQUIREMENTS)
    update_catalog(project, bootstrap_requirements=["setuptools==80.0.0"])
    assert manager.make_plan(project.root, "fixture")["fingerprint"] != plan["fingerprint"]


def test_pth_base_inheritance_is_rejected_before_interpreter_execution(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    site = Path(plan["prefix"]) / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    (site / "legacy.pth").write_text("import site; site.addsitedir('/old/base/site-packages')\n")
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: pytest.fail("Tainted startup must not run"))
    result = manager.check(plan)
    assert not result["ok"] and any("inheritance" in error for error in result["errors"])


def test_lab_external_source_is_verified_and_only_snapshot_gets_installed(manager, project, monkeypatch):
    lab = project.root.parent / "read-only-lab"
    lab.mkdir()
    update_catalog(project, isaaclab_commit="a" * 40, isaaclab_packages=["isaaclab"])
    monkeypatch.setenv("SOMAFORGE_ISAACLAB_PATH", "/old/legacy/isaaclab")
    default = manager.make_plan(project.root, "fixture")
    assert default["lab"]["source"] == str(project.root / "runtime/dependencies/isaaclab-v3.0.0")
    plan = manager.make_plan(project.root, "fixture", isaaclab=lab)
    command = next(stage["argv"] for stage in plan["stages"] if stage["id"] == "lab-packages")
    assert "--no-deps" in command and "isaaclab-dev" not in " ".join(command)
    assert str(lab) not in " ".join(command)
    assert Path(plan["lab"]["snapshot"]).is_relative_to(project.root / "runtime/environments/.sources")
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="b" * 40 + "\n"))
    with pytest.raises(ValueError, match="pinned commit"):
        manager.create(plan)
    assert not Path(plan["prefix"]).exists()


def test_lab_snapshot_retains_runtime_resources_and_records_only_development_exclusions(manager, project, monkeypatch):
    lab = project.root.parent / "external-lab"
    lab.mkdir()
    sentinel = lab / "do-not-change"
    sentinel.write_bytes(b"read-only upstream checkout")
    update_catalog(project, isaaclab_commit="a" * 40, isaaclab_packages=["isaaclab"])
    plan = manager.make_plan(project.root, "fixture", isaaclab=lab)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:") as bundle:
        for name, content in {"source/isaaclab/pyproject.toml": b"[project]\n", "apps/runtime.kit": b"runtime resource",
                              "tools/launch.sh": b"#!/bin/sh\n"}.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            bundle.addfile(member, io.BytesIO(content))
        for name in (".agents/skills/development", ".claude/skills"):
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = "../nonruntime-developer-tools"
            bundle.addfile(member)
    def git(argv, **kwargs):
        assert argv[:3] == ["git", "-C", str(lab)]
        return SimpleNamespace(stdout=archive.getvalue() if "archive" in argv else "a" * 40 + "\n")
    monkeypatch.setattr(manager.subprocess, "run", git)
    manager._stage_lab(plan["lab"])
    target = Path(plan["lab"]["snapshot"])
    assert (target / "apps/runtime.kit").read_bytes() == b"runtime resource"
    assert (target / "tools/launch.sh").stat().st_mode & 0o111
    manifest = json.loads((target / ".somaforge-source-snapshot.json").read_text())
    assert manifest["excluded_development_paths"] == [".agents/skills/development", ".claude/skills"]
    assert not (target / ".agents").exists() and not (target / ".claude").exists()
    assert sentinel.read_bytes() == b"read-only upstream checkout"
    assert sorted(path.name for path in lab.iterdir()) == ["do-not-change"]
    before = (target / "apps/runtime.kit").stat().st_mtime_ns
    manager._stage_lab(plan["lab"])
    assert (target / "apps/runtime.kit").stat().st_mtime_ns == before
    env = manager.clean_environment(plan)
    assert str(target / "source/isaaclab") in env["PYTHONPATH"].split(os.pathsep)
    assert str(lab) not in env["PYTHONPATH"]
    assert env["SOMAFORGE_ISAACLAB_PATH"] == env["ISAACLAB_PATH"] == str(target)


def test_lab_snapshot_rejects_runtime_symlinks(manager, project, monkeypatch):
    lab = project.root.parent / "external-lab"
    lab.mkdir()
    update_catalog(project, isaaclab_commit="a" * 40, isaaclab_packages=["isaaclab"])
    plan = manager.make_plan(project.root, "fixture", isaaclab=lab)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:") as bundle:
        member = tarfile.TarInfo("apps/escape")
        member.type = tarfile.SYMTYPE
        member.linkname = "/old/host"
        bundle.addfile(member)
    monkeypatch.setattr(manager.subprocess, "run", lambda argv, **kw:
        SimpleNamespace(stdout=archive.getvalue() if "archive" in argv else "a" * 40 + "\n"))
    with pytest.raises(ValueError, match="regular files/directories"):
        manager._stage_lab(plan["lab"])


def test_metadata_policy_allows_only_exact_conflict_and_valid_provider(manager, monkeypatch):
    def distribution(name, version, requires=()):
        return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires, read_text=lambda key: None)
    dists = [distribution("consumer", "1.0", ["dependency==1.0"]), distribution("dependency", "2.0")]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
    request = dict(prefix=sys.prefix, config=dict(dependency_overrides=[dict(package="consumer", version="1.0",
        requirement="dependency==1.0", installed="2.0", reason="fixture exact compatibility pin")]), source_pins=[])
    def audit():
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(manager.CHECK_CODE, {})
        return json.loads(output.getvalue())
    assert audit()["conflicts"] == []
    request["config"]["dependency_overrides"][0]["installed"] = "1.9"
    assert audit()["conflicts"] == [dict(package="consumer", version="1.0", requirement="dependency==1.0", installed="2.0")]
    dists[:] = [distribution("holosoma", "0.0.1", ["onnxruntime"]), distribution("onnxruntime-gpu", "1.26.0")]
    request["config"] = dict(provider_substitutions=[dict(package="holosoma", version="0.0.1", requirement="onnxruntime",
        provider="onnxruntime-gpu", provider_version="1.26.0", module="onnxruntime", reason="one provider only")])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(Path(sys.prefix) / "lib/onnxruntime/__init__.py")))
    assert audit()["conflicts"] == []
    dists.append(distribution("onnxruntime", "1.26.0"))
    assert any("Invalid provider" in error for error in audit()["errors"])


def test_check_preserves_raw_pip_output_and_rejects_unknown_failures(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    identity = manager._claim(plan)
    manager._write_state(Path(plan["state_file"]), {**identity, "status": "installed"})
    executable = Path(plan["prefix"]) / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture")
    def audit(argv, **kwargs):
        assert argv[0] == str(executable)
        if "-c" in argv:
            return SimpleNamespace(returncode=0, stdout=json.dumps(dict(versions={}, errors=[], conflicts=[],
                allowed_conflicts=[dict(package="consumer", version="1.0", requirement="dependency==1.0", installed="2.0")])), stderr="")
        return SimpleNamespace(returncode=1, stdout="consumer 1.0 has requirement dependency==1.0, but you have dependency 2.0.\n", stderr="")
    monkeypatch.setattr(manager.subprocess, "run", audit)
    result = manager.check(plan)
    assert result["ok"] and result["pip_check"]["returncode"] == 1
    monkeypatch.setattr(manager.subprocess, "run", lambda argv, **kw: audit(argv, **kw) if "-c" in argv else
                        SimpleNamespace(returncode=1, stdout="unexpected dependency failure\n", stderr=""))
    assert not manager.check(plan)["ok"]


def metadata_audit(manager, monkeypatch, request, distributions):
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(manager.CHECK_CODE, {})
    return json.loads(output.getvalue())


def dist(name, version, requires=(), files=None):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires,
                           read_text=lambda key: (files or {}).get(key))


def test_full_lock_checks_unpinned_by_catalog_packages_and_url_versions(manager, monkeypatch):
    request = dict(prefix=sys.prefix, config={}, source_pins=[], locks=[dict(text=(
        "dependency==1.2.3\n"
        "wheel-package @ https://example.test/wheel_package-2.0%2Bcpu-py3-none-any.whl#sha256=abc\n"
        "sdist-package @ https://example.test/sdist_package-4.0.tar.gz#sha256=def\n"))])
    distributions = [dist("dependency", "1.2.3"), dist("wheel-package", "2.0+cpu"), dist("sdist-package", "4.0")]
    audit = metadata_audit(manager, monkeypatch, request, distributions)
    assert audit["locked_count"] == 3
    assert not any("Lock mismatch" in error for error in audit["errors"])
    distributions[0] = dist("dependency", "1.2.4")
    distributions.pop()
    errors = metadata_audit(manager, monkeypatch, request, distributions)["errors"]
    assert any("dependency expected 1.2.3, got 1.2.4" in error for error in errors)
    assert any("sdist-package expected 4.0, got None" in error for error in errors)


def test_bootstrap_and_final_lock_versions_are_both_enforced(manager, monkeypatch):
    request = dict(prefix=sys.prefix, config={}, source_pins=[], locks=[dict(text="packaging==26.3\n")],
                   bootstrap_requirements=["packaging==26.0", "pip==26.2.2"])
    distributions = [dist("packaging", "26.3"), dist("pip", "26.2.2")]
    errors = metadata_audit(manager, monkeypatch, request, distributions)["errors"]
    assert "Bootstrap distribution mismatch: packaging==26.0" in errors
    distributions[1] = dist("pip", "27.0")
    errors = metadata_audit(manager, monkeypatch, request, distributions)["errors"]
    assert "Bootstrap distribution mismatch: pip==26.2.2" in errors


@pytest.mark.parametrize("lock", ["packaging==26.3\n",
    "packaging @ https://example.test/packaging-26.3-py3-none-any.whl#sha256=abc\n",
    "packaging @ https://example.test/packaging-26.3.tar.gz#sha256=abc\n"])
def test_plan_rejects_bootstrap_runtime_conflicts_before_writes(manager, project, lock, monkeypatch):
    (project.root / project.profile["requirements"]).write_text(lock)
    monkeypatch.setattr(manager.subprocess, "run", lambda *args, **kwargs: pytest.fail("plan is read-only"))
    with pytest.raises(ValueError, match="Bootstrap/runtime lock conflict for packaging"):
        manager.make_plan(project.root, "fixture")
    assert not (project.root / "runtime/environments").exists()
    update_catalog(project, bootstrap_requirements=["packaging==26.3"])
    assert manager.make_plan(project.root, "fixture")["profile"] == "fixture"


def test_requested_extras_propagate_through_project_dependency_graph(manager, monkeypatch):
    request = dict(prefix=sys.prefix, config={}, source_pins=[], locks=[])
    distributions = [dist("project", "1.0", ["storage[s3]==2.0"]),
                     dist("storage", "2.0", ['cloud[cuda]==3.0; extra == "s3"']),
                     dist("cloud", "3.0", ['plugin==4.0; extra == "cuda"'])]
    audit = metadata_audit(manager, monkeypatch, request, distributions)
    assert audit["requested_extras"] == {"storage": ["s3"], "cloud": ["cuda"]}
    assert audit["conflicts"] == [dict(package="cloud", version="3.0", requirement="plugin==4.0", installed=None)]
    distributions.append(dist("plugin", "4.0"))
    assert not metadata_audit(manager, monkeypatch, request, distributions)["conflicts"]


@pytest.mark.parametrize("module,providers", [
    ("cv2", ["opencv-python", "opencv-contrib-python", "opencv-python-headless",
             "opencv-contrib-python-headless", "opencv-python-headless-noffmpeg"]),
    ("onnxruntime", ["onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino"]),
])
def test_shared_module_providers_are_mutually_exclusive(manager, monkeypatch, module, providers):
    request = dict(prefix=sys.prefix, config={}, source_pins=[])
    for provider in providers:
        audit = metadata_audit(manager, monkeypatch, request, [dist(provider, "1.0")])
        assert not any("Conflicting providers" in error for error in audit["errors"])
    # Check each pair, including variants not selected by the current catalog.
    import itertools
    for pair in itertools.combinations(providers, 2):
        audit = metadata_audit(manager, monkeypatch, request, [dist(name, "1.0") for name in pair])
        assert "Conflicting providers for " + module + ": " + ", ".join(sorted(pair)) in audit["errors"]


def test_local_distribution_and_pinned_source_identity_are_required(manager, project, monkeypatch):
    request = dict(prefix=sys.prefix, config={}, source_pins=[dict(package="source", commit="a" * 40)],
                   local_packages=[str(project.root / "packages/example")])
    distributions = [dist("example", "1.0", files={"direct_url.json": json.dumps({
        "url": "file:///old/checkout/packages/example", "dir_info": {"editable": True}})}),
        dist("source", "1.0", files={"direct_url.json": json.dumps({"vcs_info": {"commit_id": "b" * 40}})})]
    errors = metadata_audit(manager, monkeypatch, request, distributions)["errors"]
    assert any("not editable from the configured checkout" in error for error in errors)
    assert "Source commit mismatch: source" in errors
    errors = metadata_audit(manager, monkeypatch, request, [])["errors"]
    assert "Local distribution missing/version mismatch: example" in errors


@pytest.mark.parametrize("status", [None, "creating", "interrupted", "installed"])
def test_check_and_run_reject_unowned_incomplete_or_changed_profiles(manager, project, monkeypatch, status):
    plan = manager.make_plan(project.root, "fixture")
    if status:
        identity = manager._claim(plan)
        manager._write_state(Path(plan["state_file"]), {**identity, "status": status})
    executable = Path(plan["prefix"]) / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture")
    if status == "installed":
        (project.root / project.profile["requirements"]).write_text("dependency==9.0\n")
        plan = manager.make_plan(project.root, "fixture")
    monkeypatch.setattr(manager.subprocess, "run", lambda *args, **kwargs: pytest.fail("Invalid profile must not execute Python"))
    assert not manager.check(plan)["ok"]
    with pytest.raises(ValueError, match="ownership|complete|fingerprint"):
        manager.run(plan, ["python", "-m", "example"])


def test_editable_finder_cannot_redirect_to_an_old_checkout(manager, project):
    plan = manager.make_plan(project.root, "fixture")
    site = Path(plan["prefix"]) / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    finder = "__editable___example_1_0_finder"
    (site / "example.pth").write_text(f"import {finder}; {finder}.install()\n")
    source = site / (finder + ".py")
    source.write_text("MAPPING: dict = {'example': '/old/checkout/example'}\nNAMESPACES: dict = {}\n")
    assert any("unconfigured source" in error for error in manager.inspect_pth(plan))
    source.write_text("MAPPING: dict = " + repr({"example": str(project.root / "packages/example/example")}) + "\nNAMESPACES: dict = {}\n")
    assert manager.inspect_pth(plan) == []
    (site / "arbitrary.pth").write_text("import old_environment_bootstrap\n")
    assert any("Unrecognized executable" in error for error in manager.inspect_pth(plan))


def test_known_wheel_startup_hook_requires_matching_record_hash(manager, project):
    import base64
    import hashlib
    plan = manager.make_plan(project.root, "fixture")
    site = Path(plan["prefix"]) / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    hook = site / "cmeel.pth"
    hook.write_text("import cmeel_pth\n")
    module = site / "cmeel_pth.py"
    module.write_text("# owned hook fixture\n")
    info = site / "cmeel-0.60.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: cmeel\nVersion: 0.60.1\n")
    (info / "RECORD").write_text("".join(path.name + ",sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=") + ",\n" for path in (hook, module)))
    assert not manager.inspect_pth(plan)
    module.write_text("import old_environment_bootstrap\n")
    assert any("unverified" in error for error in manager.inspect_pth(plan))


def test_shell_activation_quotes_values_and_matches_clean_run(manager, project, monkeypatch):
    plan = manager.make_plan(project.root, "fixture")
    monkeypatch.setenv("PYTHONHOME", "/old/python")
    monkeypatch.setenv("VIRTUAL_ENV", "/old/venv")
    # Shell substitutions in a path must stay literal on activation.
    plan["prefix"] += " spaces ' $(echo unsafe)"
    script = manager.shell_environment(plan)
    probe = script + "\n/usr/bin/env -0"
    result = subprocess.run(["/bin/bash", "-c", probe], capture_output=True, check=True)
    actual = dict(item.split(b"=", 1) for item in result.stdout.split(b"\0") if b"=" in item)
    expected = manager.clean_environment(plan)
    for key in ("PATH", "PYTHONPATH", "SOMAFORGE_ENV_PREFIX", "SOMAFORGE_ENV_PROFILE"):
        assert actual[key.encode()].decode() == expected[key]
    assert b"PYTHONHOME" not in actual and b"VIRTUAL_ENV" not in actual
