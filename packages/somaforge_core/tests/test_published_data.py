"""Path relocation never changes published evidence or broadens its meaning."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from somaforge_core.published_data import (
    SCHEMA,
    load_published_json,
    relocate_published_values,
    resolve_published_path,
)


@pytest.fixture
def mapping(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMAFORGE_ROOT", str(tmp_path / "checkout"))
    path = tmp_path / "checkout/runtime/current/manifests/published_data_paths.json"
    path.parent.mkdir(parents=True)

    def install(**rules):
        # Match the installer's atomic update, including filesystems whose
        # timestamps do not change during rapid same-size in-place writes.
        replacement = path.with_name(path.name + ".new")
        replacement.write_text(json.dumps({"schema": SCHEMA, **rules}))
        replacement.replace(path)
        return path

    return install


def test_missing_mapping_preserves_normal_paths_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMAFORGE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert resolve_published_path("motion.npz") == tmp_path / "motion.npz"
    assert resolve_published_path("motion.npz", tmp_path / "scene") == tmp_path / "scene/motion.npz"
    assert resolve_published_path("/unavailable/original/motion.npz") == Path("/unavailable/original/motion.npz")
    data = {"path": "/unavailable/original/motion.npz", "nested": ["relative.npz", "CONSTRAINT"], "hash": "abc"}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(data))
    assert load_published_json(source) == data
    assert relocate_published_values(data) == data


def test_exact_file_then_longest_directory_prefix_and_component_boundaries(mapping, tmp_path):
    target = tmp_path / "restored"
    mapping(prefixes=[{"source": "/old/repo", "target": str(target / "repo")},
                      {"source": "/old/repo/runtime/current", "target": str(target / "data")}],
            files=[{"source": "/old/repo/runtime/current/special.pt", "target": str(target / "selected.pt")}])
    assert resolve_published_path("/old/repo/runtime/current/special.pt") == target / "selected.pt"
    assert resolve_published_path("/old/repo/runtime/current/motions/a.npz") == target / "data/motions/a.npz"
    assert resolve_published_path("/old/repo/configs/a.json") == target / "repo/configs/a.json"
    assert resolve_published_path("/old/repo2/a.json") == Path("/old/repo2/a.json")
    assert resolve_published_path("/old/repo/runtime/current/special.pt.backup") == target / "data/special.pt.backup"
    assert resolve_published_path("relative.npz", "/old/repo/runtime/current") == target / "data/relative.npz"


def test_relocation_is_single_pass_even_with_cycles(mapping):
    mapping(prefixes=[{"source": "/a", "target": "/b"}, {"source": "/b", "target": "/a"}])
    assert resolve_published_path("/a/motion.npz") == Path("/b/motion.npz")
    assert relocate_published_values(["/a/a.npz", "/b/b.npz"]) == ["/b/a.npz", "/a/b.npz"]


def test_relative_reference_crosses_original_data_and_checkout_layout(mapping, tmp_path):
    old_repo = Path("/old/workstation/somaforge")
    old_data = Path("/old/workstation/somaforge-data/current")
    new_repo = tmp_path / "checkout"
    new_data = tmp_path / "another_disk/data"
    original_base = old_data / "motions/wbt/group"
    original_terrain = old_repo / "configs/terrain.obj"
    relative = os.path.relpath(original_terrain, original_base)
    # The broad source parent deliberately has a shorter target prefix. The
    # specific physical data alias must determine the original relative base.
    mapping(prefixes=[{"source": "/old/workstation", "target": str(tmp_path)},
                      {"source": str(old_repo), "target": str(new_repo)},
                      {"source": str(old_data), "target": str(new_data)}])
    assert resolve_published_path(relative, new_data / "motions/wbt/group") == new_repo / "configs/terrain.obj"
    assert resolve_published_path(relative, original_base) == new_repo / "configs/terrain.obj"
    assert resolve_published_path("motion.npz", new_data / "motions/wbt/group") == new_data / "motions/wbt/group/motion.npz"


def test_relative_reference_without_declared_original_destination_keeps_normal_base(mapping, tmp_path):
    new_data = tmp_path / "separate/data"
    mapping(prefixes=[{"source": "/old/work/data", "target": str(new_data)}])
    relative = "../../../unmapped/terrain.obj"
    assert resolve_published_path(relative, new_data) == (new_data / relative).resolve()


def test_relative_reference_uses_new_symlink_target_without_old_filesystem(mapping, tmp_path):
    repo = tmp_path / "restored_checkout"
    external = tmp_path / "external_disk"
    external.mkdir()
    (repo / "runtime").mkdir(parents=True, exist_ok=True)
    (repo / "runtime/current").symlink_to(external, target_is_directory=True)
    mapping(prefixes=[{"source": "/old/work/repo", "target": str(repo)},
                      {"source": "/old/work/data", "target": str(repo / "runtime/current")}])
    old_base = Path("/old/work/data/motions/group")
    relative = os.path.relpath("/old/work/repo/configs/terrain.obj", old_base)
    assert resolve_published_path(relative, external / "motions/group") == repo / "configs/terrain.obj"


def test_json_aliases_nested_metadata_without_touching_source_bytes(mapping, tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    mapping(prefixes=[{"source": "/original/data", "target": str(target)}])
    source = target / "manifest.json"
    document = {"motion": "/original/data/motion.npz", "relative": "motions/relative.npz",
                "description": "source was /original/data/motion.npz", "url": "https://example.com/a",
                "native_semantics": "CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc",
                "nested": [{"evidence": "/original/data/source.npz", "sha256": "fixed-hash"}],
                "/original/data/key": "/undeclared/path"}
    source.write_text(json.dumps(document))
    original = source.read_bytes()
    before_hash = hashlib.sha256(original).hexdigest()
    loaded = load_published_json("/original/data/manifest.json")
    assert loaded["motion"] == str(target / "motion.npz")
    assert loaded["nested"][0] == {"evidence": str(target / "source.npz"), "sha256": "fixed-hash"}
    for key in ("relative", "description", "url", "native_semantics", "/original/data/key"):
        assert loaded[key] == document[key]
    assert source.read_bytes() == original
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash


def test_checkpoint_metadata_keeps_tensors_keys_and_input_unchanged(mapping, tmp_path):
    mapping(prefixes=[{"source": "/old/data", "target": str(tmp_path)}])
    tensor_sentinel = object()
    original = {"motion_keys": ["/old/data/motion.npz"], "tensors": tensor_sentinel,
                "group": ("/old/data/terrain.obj", 30), "/old/data/dict_key": True}
    result = relocate_published_values(original)
    assert result["motion_keys"] == [str(tmp_path / "motion.npz")]
    assert result["group"] == (str(tmp_path / "terrain.obj"), 30)
    assert result["tensors"] is tensor_sentinel
    assert "/old/data/dict_key" in result
    assert original["motion_keys"] == ["/old/data/motion.npz"]


@pytest.mark.parametrize("payload", [
    {"schema": "wrong"},
    {"schema": SCHEMA, "prefixes": {}},
    {"schema": SCHEMA, "files": [{"source": "relative", "target": "/new"}]},
    {"schema": SCHEMA, "prefixes": [{"source": "/old", "target": "relative"}]},
    {"schema": SCHEMA, "files": [{"source": "/old"}]},
    {"schema": SCHEMA, "prefixes": [{"source": "/old", "target": "/a"},
                                    {"source": "/old", "target": "/b"}]},
])
def test_invalid_installed_mapping_is_an_error(mapping, payload):
    path = mapping()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_published_path("/old/motion.npz")


def test_mapping_updates_are_visible_and_never_inferred_from_path_exists(mapping, tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    # Declared mappings apply even when an old path still happens to exist.
    mapping(prefixes=[{"source": str(source), "target": str(tmp_path / "new_a")}])
    assert resolve_published_path(source / "motion.npz") == tmp_path / "new_a/motion.npz"
    mapping(prefixes=[{"source": str(source), "target": str(tmp_path / "new_b")}])
    assert resolve_published_path(source / "motion.npz") == tmp_path / "new_b/motion.npz"


def test_tool_loads_without_site_packages(mapping):
    import somaforge_core.published_data as module
    mapping(prefixes=[{"source": "/old/data", "target": "/restored/data"}])
    result = subprocess.run([sys.executable, "-S", "-c", """
import runpy, sys
namespace = runpy.run_path(sys.argv[1])
assert str(namespace['resolve_published_path']('/old/data/motion.npz')) == '/restored/data/motion.npz'
assert not {'numpy', 'torch', 'newton', 'isaaclab'} & sys.modules.keys()
""", module.__file__], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_rule_cache_reads_unchanged_mapping_json_once(mapping, tmp_path, monkeypatch):
    from somaforge_core import published_data as module
    path = mapping(files=[{"source": "/old/motion.npz", "target": str(tmp_path / "new/motion.npz")}])
    original_load = module.json.load
    reads = []
    def counted_load(stream, *args, **kwargs):
        reads.append(Path(stream.name))
        return original_load(stream, *args, **kwargs)
    monkeypatch.setattr(module.json, "load", counted_load)
    for _ in range(25):
        assert resolve_published_path("/old/motion.npz") == tmp_path / "new/motion.npz"
        assert relocate_published_values({"path": "/old/motion.npz"}) == {"path": str(tmp_path / "new/motion.npz")}
    assert reads == [path.resolve()]
    # A caller cannot mutate the cached exact-file table for later readers.
    files, _ = module._load_rules()
    with pytest.raises(TypeError):
        files[Path("/old/motion.npz")] = tmp_path / "wrong.npz"
    assert resolve_published_path("/old/motion.npz") == tmp_path / "new/motion.npz"


def test_rule_cache_sees_atomic_replacement_with_same_size_and_mtime(mapping, tmp_path):
    path = mapping(prefixes=[{"source": "/old/data", "target": str(tmp_path / "new_a")}])
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new_a/a.npz"
    previous = path.stat()
    replacement = path.with_name("replacement.json")
    replacement.write_text(json.dumps({"schema": SCHEMA, "prefixes": [
        {"source": "/old/data", "target": str(tmp_path / "new_b")}]}))
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new_b/a.npz"


def test_rule_cache_sees_in_place_update_with_changed_file_stamp(mapping, tmp_path):
    path = mapping(prefixes=[{"source": "/old/data", "target": str(tmp_path / "new_a")}])
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new_a/a.npz"
    previous = path.stat()
    path.write_text(json.dumps({"schema": SCHEMA, "prefixes": [
        {"source": "/old/data", "target": str(tmp_path / "new_b")}]}))
    # Explicitly change the stamp rather than relying on the filesystem's
    # clock granularity; same-size/same-mtime atomic replacement is tested above.
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new_b/a.npz"


def test_rule_cache_follows_root_switch_and_reuses_previous_root(mapping, tmp_path, monkeypatch):
    from somaforge_core import published_data as module
    first = mapping(prefixes=[{"source": "/old/data", "target": str(tmp_path / "first_target")}])
    root_a = Path(os.environ["SOMAFORGE_ROOT"])
    root_b = tmp_path / "another_checkout"
    second = root_b / "runtime/current/manifests/published_data_paths.json"
    second.parent.mkdir(parents=True)
    second.write_text(json.dumps({"schema": SCHEMA, "prefixes": [
        {"source": "/old/data", "target": str(tmp_path / "second_target")}]}))
    original_load = module.json.load
    reads = []
    def counted_load(stream, *args, **kwargs):
        reads.append(Path(stream.name))
        return original_load(stream, *args, **kwargs)
    monkeypatch.setattr(module.json, "load", counted_load)
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "first_target/a.npz"
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root_b))
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "second_target/a.npz"
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root_a))
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "first_target/a.npz"
    assert reads == [first.resolve(), second.resolve()]


def test_rule_cache_sees_file_appearance_absence_and_invalid_update(mapping, tmp_path):
    assert resolve_published_path("/old/data/a.npz") == Path("/old/data/a.npz")
    path = mapping(prefixes=[{"source": "/old/data", "target": str(tmp_path / "new")}])
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new/a.npz"
    retained = path.with_name("retained_mapping.json")
    path.rename(retained)
    assert resolve_published_path("/old/data/a.npz") == Path("/old/data/a.npz")
    retained.rename(path)
    assert resolve_published_path("/old/data/a.npz") == tmp_path / "new/a.npz"
    path.write_text(json.dumps({"schema": "invalid replacement"}))
    with pytest.raises(ValueError, match="Unsupported published-data"):
        resolve_published_path("/old/data/a.npz")
