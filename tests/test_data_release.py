"""Exercise the actual publisher and installer with portable immutable fixtures."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_tools(monkeypatch):
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(__import__("sys").modules, name, module)
        spec.loader.exec_module(module)
        return module
    download = load("download_data", "download_data.py")
    return download, load("_test_build_data_release", "build_data_release.py")


def fixture_release(tmp_path, tools):
    download, publisher = tools
    original = tmp_path / "original"
    original.mkdir()
    a, b = original / "motion.npz", original / "checkpoint.pt"
    a.write_bytes(b"immutable motion fixture\0" * 10)
    b.write_bytes(b"immutable model fixture\0" * 12)
    def row(path):
        return dict(source=str(path), destination="runtime/current/examples/"+path.name, size=path.stat().st_size)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(dict(source_roots_aliases=[dict(source=str(original), target="runtime/current/examples")],
        components=[dict(id="motions", purpose="test motions", status="fixture", dependencies=[], files=[row(a)]),
                    dict(id="models", purpose="test model", status="fixture", dependencies=[], files=[row(a),row(b)])])))
    assets = tmp_path / "assets"
    catalog = publisher.build([inventory], assets, "owner/repo", "data-fixture", tmp_path / "catalog.json", part_bytes=512)
    return catalog, assets, a, b


def test_publish_install_dependencies_cross_root_and_verify_without_mutating_sources(tmp_path, release_tools, monkeypatch):
    download, _ = release_tools
    catalog, assets, a, b = fixture_release(tmp_path, release_tools)
    before = [download.sha256(p) for p in (a,b)]
    assert catalog["components"]["models"]["dependencies"] == ["motions"]
    root = tmp_path / "new-machine"
    root.mkdir()
    assert download.install(catalog, root, ["models"], local_archives=assets) == 2
    assert download.install(catalog, root, ["models"], local_archives=assets, verify=True) == 2
    installed = [root / "runtime/current/examples" / p.name for p in (a,b)]
    assert [download.sha256(p) for p in installed] == before
    mtimes = [p.stat().st_mtime_ns for p in installed]
    download.install(catalog, root, ["models"], local_archives=assets)
    assert [p.stat().st_mtime_ns for p in installed] == mtimes
    assert [download.sha256(p) for p in (a,b)] == before
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root))
    from somaforge_core.published_data import resolve_published_path
    assert resolve_published_path(a) == installed[0]


def test_conflicting_existing_file_and_corrupt_archive_are_rejected(tmp_path, release_tools):
    download, _ = release_tools
    catalog, assets, a, _ = fixture_release(tmp_path, release_tools)
    root = tmp_path / "new-machine"
    target = root / "runtime/current/examples/motion.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        download.install(catalog, root, ["motions"], local_archives=assets)
    assert target.read_bytes() == b"existing user data"
    archive = assets / catalog["components"]["motions"]["archives"][0]["name"]
    archive.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="corrupt"):
        download.install(catalog, tmp_path / "another-machine", ["motions"], local_archives=assets)


@pytest.mark.parametrize("name,typeflag", [("../escape",tarfile.REGTYPE),
    ("runtime/current/escape",tarfile.SYMTYPE)])
def test_unexpected_members_and_links_are_never_extracted(tmp_path, release_tools, name, typeflag):
    download, _ = release_tools
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.type = typeflag
        member.size = 1 if typeflag == tarfile.REGTYPE else 0
        member.linkname = "/outside"
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unexpected"):
        download.extract_part(path, tmp_path / "root", {}, set())
    assert not (tmp_path / "escape").exists()


def test_dependency_cycles_and_symlink_escape_are_rejected(tmp_path, release_tools):
    download, _ = release_tools
    with pytest.raises(ValueError, match="cycle"):
        download.selected_components({"components":{"a":{"dependencies":["b"]},"b":{"dependencies":["a"]}}}, ["a"])
    root = tmp_path / "root"
    (root/"runtime/current").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root/"runtime/current/escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        download.target_path(root, "runtime/current/escape/unsafe")


@pytest.mark.parametrize("length", [216, 217, 255])
def test_long_release_basenames_import_without_long_temporary_names(tmp_path, release_tools, length):
    download, _ = release_tools
    name = "x" * (length - 4) + ".npz"
    relative = "runtime/current/motions/" + name
    content = b"unchanged canonical motion"
    archive = tmp_path / "long-name.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    expected = {relative: dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())}
    root = tmp_path / "new-machine"
    observed = set()
    download.extract_part(archive, root, expected, observed)
    target = root / relative
    assert target.read_bytes() == content
    assert observed == {relative}
    assert list(target.parent.glob(".import-*")) == []


def test_longest_prefix_wins_and_existing_prefixes_require_exact_exception(tmp_path, release_tools, monkeypatch):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    mapping = root / "runtime/current/manifests/published_data_paths.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps(dict(schema="somaforge_published_data_paths_v1", files=[],
        prefixes=[dict(source="/old/branch", target=str(root / "runtime/current/incorrect"))])))
    catalog = dict(source_roots_aliases=[dict(source="/old", target="runtime/current")])
    records = [dict(path="runtime/current/branch/motion.npz", sources=["/old/branch/motion.npz"])]
    download.write_local_mapping(root, catalog, records)
    result = json.loads(mapping.read_text())
    target = root / "runtime/current/branch/motion.npz"
    assert dict(source="/old/branch/motion.npz", target=str(target)) in result["files"]
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root))
    from somaforge_core.published_data import resolve_published_path
    assert resolve_published_path("/old/branch/motion.npz") == target


def test_conflicting_exact_alias_is_not_hidden_by_matching_prefix(tmp_path, release_tools):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    mapping = root / "runtime/current/manifests/published_data_paths.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps(dict(schema="somaforge_published_data_paths_v1", prefixes=[],
        files=[dict(source="/old/motion.npz", target=str(root / "runtime/current/different.npz"))])))
    before = mapping.read_bytes()
    catalog = dict(source_roots_aliases=[dict(source="/old", target="runtime/current")])
    records = [dict(path="runtime/current/motion.npz", sources=["/old/motion.npz"])]
    with pytest.raises(ValueError, match="Conflicting installed data alias"):
        download.write_local_mapping(root, catalog, records)
    assert mapping.read_bytes() == before


@pytest.mark.parametrize("target", ["/outside", "../outside", "runtime/current/../outside", "a//b", "a\\b"])
def test_unsafe_catalog_alias_target_is_rejected_before_mapping_write(tmp_path, release_tools, target):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    with pytest.raises(ValueError, match="Unsafe release path"):
        download.write_local_mapping(root, dict(source_roots_aliases=[dict(source="/old", target=target)]), [])
    assert not (root / "runtime/current/manifests/published_data_paths.json").exists()


def test_alias_source_must_be_absolute_and_target_cannot_follow_escape_symlink(tmp_path, release_tools):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    root.mkdir()
    with pytest.raises(ValueError, match="absolute path"):
        download.write_local_mapping(root, dict(source_roots_aliases=[dict(source="relative", target=".")]), [])
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        download.write_local_mapping(root, dict(source_roots_aliases=[dict(source="/old", target="escape")]), [])
    assert not list(outside.iterdir())


def test_bad_import_checksum_keeps_partial_and_retry_preserves_it(tmp_path, release_tools):
    download, _ = release_tools
    relative = "runtime/current/motions/a.npz"
    content = b"original immutable bytes"
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    root = tmp_path / "new-machine"
    bad = {relative: dict(size=len(content), sha256="0" * 64)}
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.extract_part(archive, root, bad, set())
    assert not (root / relative).exists()
    partials = list((root / relative).parent.glob(".import-*"))
    assert len(partials) == 1 and partials[0].read_bytes() == content
    good = {relative: dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())}
    download.extract_part(archive, root, good, set())
    assert (root / relative).read_bytes() == content
    assert partials[0].read_bytes() == content


def test_interrupted_download_retains_partial_and_retry_does_not_delete_it(tmp_path, release_tools, monkeypatch):
    download, _ = release_tools
    content = b"release archive fixture"
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("fixture connection interrupted")
            return super().read(size)
    record = dict(name="fixture.tar.gz", url="https://example.test/fixture.tar.gz",
                  size=len(content), sha256=hashlib.sha256(content).hexdigest())
    cache = tmp_path / "cache"
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Interrupted(content))
    with pytest.raises(ConnectionResetError, match="interrupted"):
        download.acquire(record, cache)
    partials = list(cache.glob(".partial-*"))
    assert len(partials) == 1 and partials[0].read_bytes() == content
    assert not (cache / record["name"]).exists()
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    assert download.acquire(record, cache).read_bytes() == content
    assert partials[0].read_bytes() == content


def test_later_component_aliases_survive_payload_deduplication(tmp_path, release_tools, monkeypatch):
    download, publisher = release_tools
    original = tmp_path / "original"
    original.mkdir()
    first, second = original / "a.npz", original / "b.npz"
    first.write_bytes(b"same immutable motion")
    second.write_bytes(first.read_bytes())
    destination = "runtime/current/examples/shared.npz"
    def row(path, aliases=()):
        return dict(source=str(path), destination=destination, size=path.stat().st_size, aliases=list(aliases))
    historical = "/historical/independent/motion.npz"
    inventory = tmp_path / "inventory.json"
    # Also cover the alternate alias-field spelling produced by the full
    # demonstration inventory, rather than only the PHP inventory's spelling.
    inventory.write_text(json.dumps(dict(source_roots_aliases=[
        dict(source_root=str(original), destination_root="runtime/current/examples")], components=[
        dict(id="owner", purpose="owner", status="fixture", dependencies=[], files=[row(first)]),
        dict(id="later", purpose="alias owner", status="fixture", dependencies=[], files=[row(second, [historical])]) ])))
    assets = tmp_path / "assets"
    catalog = publisher.build([inventory], assets, "owner/repo", "fixture", tmp_path / "catalog.json")
    assert catalog["components"]["later"]["dependencies"] == ["owner"]
    assert catalog["components"]["later"]["file_count"] == 0
    assert catalog["components"]["later"]["archives"] == []
    owner_index = json.loads((assets / catalog["components"]["owner"]["index"]["name"]).read_text())
    assert set(owner_index["files"][0]["sources"]) == {str(first), str(second), historical}
    root = tmp_path / "new-machine"
    assert download.install(catalog, root, ["later"], local_archives=assets) == 1
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root))
    from somaforge_core.published_data import resolve_published_path
    assert resolve_published_path(historical) == root / destination
    assert first.read_bytes() == second.read_bytes() == b"same immutable motion"


def test_publisher_stream_verifier_checks_inner_member_hashes(tmp_path, release_tools, monkeypatch):
    download, publisher = release_tools
    catalog, assets, _, _ = fixture_release(tmp_path, release_tools)
    modes = []
    open_archive = publisher.tarfile.open
    def stream_open(path, mode, **kwargs):
        modes.append(mode)
        return open_archive(path, mode, **kwargs)
    monkeypatch.setattr(publisher.tarfile, "open", stream_open)
    assert publisher.verify_release(catalog, assets) == 2
    assert modes and set(modes) == {"r|gz"}
    # A valid outer archive checksum cannot certify different inner bytes.
    record = catalog["components"]["motions"]["archives"][0]
    index = json.loads((assets / catalog["components"]["motions"]["index"]["name"]).read_text())
    declared = index["files"][0]
    path = assets / "changed-inner-member.tar.gz"
    with open_archive(path, "w:gz") as bundle:
        member = tarfile.TarInfo(declared["path"])
        member.size = declared["size"]
        bundle.addfile(member, io.BytesIO(b"x" * member.size))
    record.update(name=path.name, size=path.stat().st_size, sha256=download.sha256(path))
    with pytest.raises(ValueError, match="member checksum mismatch"):
        publisher.verify_release(catalog, assets)


def test_publisher_cli_invokes_stream_verification_after_build(tmp_path, release_tools, monkeypatch):
    _, publisher = release_tools
    calls = []
    catalog = {"fixture": True}
    monkeypatch.setattr(publisher, "build", lambda *a, **k: calls.append("build") or catalog)
    monkeypatch.setattr(publisher, "verify_release", lambda c, p: calls.append(("verify", c, p)))
    monkeypatch.setattr(publisher.os, "nice", lambda amount: None)
    output = tmp_path / "assets"
    monkeypatch.setattr(__import__("sys"), "argv", ["publisher", "--inventories", str(tmp_path / "input.json"),
        "--output", str(output), "--catalog", str(tmp_path / "catalog.json"), "--tag", "fixture"])
    publisher.main()
    assert calls == ["build", ("verify", catalog, output)]


def test_only_byte_identical_committed_inputs_are_omitted_from_runtime_payload(tmp_path, release_tools):
    _, publisher = release_tools
    checkout = tmp_path / "source-checkout"
    source = checkout / "configs/surface.obj"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"committed terrain bytes")
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "configs/surface.obj"], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=fixture", "-c", "user.email=fixture@example.test",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture asset"], check=True)
    inventory = tmp_path / "inventory.json"
    def write_inventory():
        inventory.write_text(json.dumps(dict(repo_root=str(checkout), components=[dict(id="tracked", purpose="tracked terrain",
            status="fixture", dependencies=[], files=[dict(source=str(source), destination="configs/surface.obj",
                                                          size=source.stat().st_size)])])))
    write_inventory()
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    catalog = publisher.build([inventory], tmp_path / "assets", "owner/repo", "fixture", tmp_path / "catalog.json")
    assert catalog["components"]["tracked"]["file_count"] == 0
    assert catalog["components"]["tracked"]["archives"] == []
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    source.write_bytes(b"uncommitted changed terrain")
    write_inventory()
    with pytest.raises(ValueError, match="Tracked data differs from committed"):
        publisher.build([inventory], tmp_path / "new-assets", "owner/repo", "fixture", tmp_path / "new-catalog.json")


def test_repository_files_map_historical_tmp_alias_without_writing_checkout_input(tmp_path, release_tools, monkeypatch):
    download, _ = release_tools
    catalog, assets, _, _ = fixture_release(tmp_path, release_tools)
    root = tmp_path / "new-machine"
    terrain = root / "configs/terrain.obj"
    terrain.parent.mkdir(parents=True)
    terrain.write_bytes(b"Git-distributed terrain bytes")
    old_alias = "/old-machine/tmp/generated-terrain/terrain.obj"
    catalog["repository_files"] = [dict(path="configs/terrain.obj", size=terrain.stat().st_size,
        sha256=download.sha256(terrain), sources=[old_alias])]
    before = (terrain.read_bytes(), terrain.stat().st_mtime_ns)
    assert download.install(catalog, root, ["motions"], local_archives=assets) == 1
    assert download.install(catalog, root, ["motions"], local_archives=assets, verify=True) == 1
    assert (terrain.read_bytes(), terrain.stat().st_mtime_ns) == before
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root))
    from somaforge_core.published_data import resolve_published_path
    assert resolve_published_path(old_alias) == terrain
    mapping = json.loads((root / "runtime/current/manifests/published_data_paths.json").read_text())
    assert dict(source=old_alias, target=str(terrain)) in mapping["files"]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_repository_files_missing_or_changed_fail_before_fetch(tmp_path, release_tools, existing, verify):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    target = root / "configs/terrain.obj"
    target.parent.mkdir(parents=True)
    if existing:
        target.write_bytes(b"local user terrain")
    catalog = dict(schema=download.SCHEMA, tag="fixture", components={}, repository_files=[
        dict(path="configs/terrain.obj", size=5, sha256=hashlib.sha256(b"valid").hexdigest(), sources=[])])
    with pytest.raises(ValueError, match="Missing or corrupt Git-distributed"):
        download.install(catalog, root, ["all"], local_archives=tmp_path / "no-downloads", verify=verify)
    if existing:
        assert target.read_bytes() == b"local user terrain"
    else:
        assert not target.exists()
    assert not (root / "runtime").exists()


@pytest.mark.parametrize("value", ["../outside", "/absolute", "runtime/current/terrain.obj", "configs//a.obj", "."])
def test_repository_files_reject_unsafe_or_runtime_paths(tmp_path, release_tools, value):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    catalog = dict(schema=download.SCHEMA, tag="fixture", components={}, repository_files=[
        dict(path=value, size=0, sha256="0" * 64, sources=[])])
    with pytest.raises(ValueError, match="Unsafe release path|outside runtime/current"):
        download.install(catalog, root, ["all"])
    assert not root.exists()


def test_repository_files_cannot_follow_symlink_outside_checkout(tmp_path, release_tools):
    download, _ = release_tools
    root = tmp_path / "new-machine"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    terrain = outside / "terrain.obj"
    terrain.write_bytes(b"external terrain")
    (root / "configs").symlink_to(outside, target_is_directory=True)
    catalog = dict(schema=download.SCHEMA, tag="fixture", components={}, repository_files=[
        dict(path="configs/terrain.obj", size=terrain.stat().st_size, sha256=download.sha256(terrain), sources=[])])
    with pytest.raises(ValueError, match="escapes the checkout"):
        download.install(catalog, root, ["all"])
    assert terrain.read_bytes() == b"external terrain"
