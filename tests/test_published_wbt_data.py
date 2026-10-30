"""Published WBT metadata moves without rewriting evidence or sampler tensors."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.managers.command.tests.test_wbt_recent30_probe import command, append
from holosoma.utils.eval_utils import _experiment_config_from_serialized
from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest, _resolve_manifest_path
from holosoma.utils.path import resolve_data_file_path
from somaforge_core import robot_assets


ROOT = Path(__file__).resolve().parents[1]
OLD_CHECKOUT = Path("/unmounted/source_somaforge")
OLD_RUNTIME = Path("/unmounted/source_somaforge_data/current")


def test_nested_release_installation_paths_are_not_relocated_twice(tmp_path, monkeypatch):
    from somaforge_core.published_data import resolve_published_path, relocate_published_values
    old = tmp_path / 'original_data'
    root = old / 'release_validation/checkout'
    installed = root / 'runtime/current'
    mapping = installed / 'manifests/published_data_paths.json'
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps(dict(schema='somaforge_published_data_paths_v1', files=[],
        prefixes=[dict(source=str(old), target=str(installed)),
                  dict(source=str(tmp_path/'original_checkout'), target=str(root))])))
    monkeypatch.setenv('SOMAFORGE_ROOT', str(root))
    expected = installed / 'motions/reference.npz'
    assert resolve_published_path(old/'motions/reference.npz') == expected
    assert resolve_published_path(expected) == expected
    assert relocate_published_values(relocate_published_values(str(old/'motions/reference.npz'))) == str(expected)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / "new_checkout"
    mapping = root / "runtime/current/manifests/published_data_paths.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps({"schema": "somaforge_published_data_paths_v1", "files": [],
        "prefixes": [{"source": str(OLD_CHECKOUT), "target": str(root)},
                     {"source": str(OLD_RUNTIME), "target": str(root / "runtime/current")}]}))
    # This unit fixture tests metadata relocation against the real canonical
    # asset contract; bootstrap relocation of asset bytes has separate coverage.
    monkeypatch.setattr(robot_assets, "somaforge_root", lambda: ROOT)
    monkeypatch.setenv("SOMAFORGE_ROOT", str(root))
    return root


def test_data_path_aliases_preserve_package_and_unmapped_behavior(publication):
    assert resolve_data_file_path(str(OLD_RUNTIME / "motions/a.npz")) == str(publication / "runtime/current/motions/a.npz")
    assert resolve_data_file_path(str(OLD_CHECKOUT / "runtime/current/motions/a.npz")) == str(publication / "runtime/current/motions/a.npz")
    assert resolve_data_file_path("/unmapped/input.npz") == "/unmapped/input.npz"
    assert resolve_data_file_path("s3://bucket/input.npz") == "s3://bucket/input.npz"
    assert resolve_data_file_path("holosoma/data/robots/g1/g1_29dof_spherehand.urdf") == str(ROOT / robot_assets.G1_SPHEREHAND_URDF_RELATIVE)


@pytest.mark.parametrize("extension", [".json", ".yaml"])
def test_canonical_manifest_relocates_source_and_terrain_paths_without_rewriting_hashes(publication, extension):
    folder = publication / "runtime/current/motions/test"
    folder.mkdir(parents=True)
    for name in ("motion.npz", "source.npz", "terrain.obj"):
        (folder / name).write_bytes(name.encode())
    asset = robot_assets.canonical_g1_asset_metadata()
    digest = lambda name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
    payload = {"schema": "somaforge_motion_terrain_manifest_v1", "robot_asset_id": "robot.g1.spherehand",
        "robot_asset_sha256": asset["urdf_sha256"], "robot_asset_bundle_sha256": asset["asset_bundle_sha256"],
        "robot_asset_usd_bundle_sha256": asset["usd_bundle_sha256"], "kinematics_backend": "newton_direct_fk",
        "motion_files": [{"motion_id": "one", "motion_file": "motion.npz", "terrain_id": 0,
            "source_file": str(OLD_RUNTIME / "motions/test/source.npz"), "source_sha256": digest("source.npz"),
            "motion_sha256": digest("motion.npz"), "kinematics_backend": "newton_direct_fk"}],
        "terrains": [{"terrain_id": 0, "terrain_file": str(OLD_CHECKOUT / "runtime/current/motions/test/terrain.obj"),
            "terrain_sha256": digest("terrain.obj")} ]}
    path = folder / ("manifest" + extension)
    # JSON is also a valid YAML mapping, so both readers exercise the same bytes.
    path.write_text(json.dumps(payload))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_motion_terrain_manifest(str(OLD_RUNTIME / "motions/test" / path.name))
    assert result["motion_files"][0]["source_file"] == str(folder / "source.npz")
    assert result["terrains"][0]["terrain_file"] == str(folder / "terrain.obj")
    assert result["motion_files"][0]["source_sha256"] == payload["motion_files"][0]["source_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_checkpoint_config_relocation_preserves_learning_rates_and_original_metadata(publication):
    serialized = ExperimentConfig().to_serializable_dict()
    serialized["training"]["checkpoint"] = str(OLD_RUNTIME / "models/teacher.pt")
    serialized["logger"]["base_dir"] = str(OLD_CHECKOUT / "runtime/current/models/new_run")
    serialized["terrain"]["terrain_term"]["motion_matched_manifest"] = str(OLD_RUNTIME / "motions/test/manifest.json")
    serialized["algo"]["config"]["actor_learning_rate"] = .00037
    serialized["algo"]["config"]["critic_learning_rate"] = .00019
    original = copy.deepcopy(serialized)
    restored = _experiment_config_from_serialized(serialized)
    assert restored.training.checkpoint == str(publication / "runtime/current/models/teacher.pt")
    assert restored.logger.base_dir == str(publication / "runtime/current/models/new_run")
    assert restored.terrain.terrain_term.motion_matched_manifest == str(publication / "runtime/current/motions/test/manifest.json")
    assert restored.algo.config.actor_learning_rate == .00037
    assert restored.algo.config.critic_learning_rate == .00019
    assert serialized == original


def test_sampler_restores_motions_and_json_group_identities_on_new_machine(publication):
    source = command((str(OLD_RUNTIME / "motions/a.npz"), str(OLD_CHECKOUT / "runtime/current/motions/b.npz")), group=True)
    # get_checkpoint_state normally writes real source-machine paths. Create
    # its immutable evidence before invoking restoration on another machine.
    append(source, 0, [0, 1, 0])
    append(source, 1, [1, 1])
    append(source, 0, [0, 1, 0], group=True)
    append(source, 1, [1, 1], group=True)
    source._normal_group_sampling_weights.copy_(torch.tensor([.8, .2]))
    source._refresh_normal_motion_weights_from_group_weights()
    state = source.get_checkpoint_state()
    state["motion_keys"] = [str(OLD_RUNTIME / "motions/a.npz"), str(OLD_CHECKOUT / "runtime/current/motions/b.npz")]
    state["group_probe"]["keys"] = [json.dumps([key]) for key in state["motion_keys"]]
    original = copy.deepcopy(state)
    target = command((str(publication / "runtime/current/motions/b.npz"),
                      str(publication / "runtime/current/motions/a.npz")), group=True)
    target.load_checkpoint_state(state)
    assert target._sampler_checkpoint_matched_motion_count == 2
    assert target._probe_success_recent30.tolist() == pytest.approx([1., 1 / 3])
    assert target._group_probe_success_recent30.tolist() == pytest.approx([1., 1 / 3])
    assert target._normal_group_sampling_weights.tolist() == pytest.approx([.2, .8])
    assert torch.equal(target._probe_recent30_values, original["tensors"]["_probe_recent30_values"].flip(0))
    assert torch.equal(target._group_probe_recent30_values, original["group_probe"]["tensors"]["_group_probe_recent30_values"].flip(0))
    assert state["motion_keys"] == original["motion_keys"]
    assert state["group_probe"]["keys"] == original["group_probe"]["keys"]
    for key, tensor in state["tensors"].items():
        assert torch.equal(tensor, original["tensors"][key])


def test_php_inventory_uses_logical_runtime_paths_and_excludes_archives(tmp_path):
    spec = importlib.util.spec_from_file_location("php_data_release_inventory", ROOT / "scripts/php_data_release_inventory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "checkout"
    runtime = root / "runtime/current"
    source = runtime / "models/teacher.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unchanged checkpoint")
    entry = module._entry(root, source)
    assert entry["destination"] == "runtime/current/models/teacher.pt"
    assert entry["source"] == str(source)
    assert entry["size"] == len(b"unchanged checkpoint")
    archived = runtime / "archives/duplicate.pt"
    archived.parent.mkdir()
    archived.write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="Archive duplicates"):
        module._entry(root, archived)
    temporary = root / "tmp/model.pt"
    temporary.parent.mkdir()
    temporary.write_bytes(b"diagnostic")
    with pytest.raises(ValueError, match="Temporary diagnostic"):
        module._entry(root, temporary)


def test_baseline29_relative_terrain_keeps_original_logical_checkout_base(publication):
    original = ROOT / "runtime/current/manifests/omniretarget_baseline_29.json"
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    terrain = json.loads(original.read_text())["terrains"][0]
    relative = terrain["terrain_file"]
    assert relative.startswith("../../../configs/")
    base = publication / "runtime/current/manifests"
    # The published manifest is on a different physical runtime volume, while
    # its original relative path was authored through repo/runtime/current.
    expected = publication / "configs/motion_matched/terrain_obj_cache/climb_00/multi_boxes_z_scale_1.0.obj"
    assert _resolve_manifest_path(relative, base) == str(expected)
    tracked = ROOT / expected.relative_to(publication)
    assert hashlib.sha256(tracked.read_bytes()).hexdigest() == terrain["terrain_sha256"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_motionedit_asset_version_plan_and_segment_readers_relocate_metadata_only(publication):
    from motion_edit.contact.plans import read_contact_edit_plan
    from motion_edit.storage.io import read_motion_asset, read_motion_version, read_canonical_segments

    folder = publication / "runtime/current/motion_edit"
    folder.mkdir(parents=True)
    motion = str(OLD_RUNTIME / "motions/a.npz")
    contact = str(OLD_RUNTIME / "motion_edit/contacts/a.json")
    metadata = {"raw_contact": [True, False], "force": [0., 123.], "sha256": "a" * 64}
    rows = {
        "asset.json": {"motion_asset_id": "a", "motion_path": motion, "contact_layer": contact,
                       "metadata": metadata},
        "version.json": {"motion_version_id": "a_v1", "motion_path": motion, "contact_layer": contact,
                         "metadata": metadata},
        "plan.json": {"plan_id": "plan", "source_motion_id": "a", "source_motion_path": motion,
                      "source_contact_layer": contact, "metadata": metadata},
        "segments.jsonl": {"motion_id": "a", "segment_id": "a_0", "start_frame": 0, "end_frame": 3,
                           "motion_path": motion, "metadata": metadata},
    }
    for name, row in rows.items():
        (folder / name).write_text(json.dumps(row) + "\n")
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in rows}
    old = OLD_RUNTIME / "motion_edit"
    asset = read_motion_asset("a", old / "asset.json")
    version = read_motion_version("a_v1", old / "version.json")
    plan = read_contact_edit_plan(old / "plan.json")
    segment = read_canonical_segments("a_v1", old / "segments.jsonl")[0]
    expected_motion = str(publication / "runtime/current/motions/a.npz")
    assert asset.motion_path == version.motion_path == segment.motion_path == expected_motion
    assert plan.source_motion_path == expected_motion
    assert asset.contact_layer == version.contact_layer == plan.source_contact_layer == str(folder / "contacts/a.json")
    for record in (asset, version, plan, segment):
        assert record.metadata == metadata
    assert {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in rows} == hashes


def test_query_cli_relocates_only_declared_input_paths_and_keeps_shared_binding(publication):
    from scripts import serve_newton_contact_queries as module
    from somaforge_core.native_model_binding import inspect_model, native_binding

    cfg = module.Config(checkpoint=OLD_RUNTIME / "models/teacher.pt", binding=OLD_RUNTIME / "bindings/a.json",
                        inspection_output=Path("new_inspection.json"), query_output=Path("new_query.json"),
                        probe_motion=OLD_RUNTIME / "motions/a.npz")
    mapped = module._relocate_query_inputs(cfg)
    assert mapped.checkpoint == publication / "runtime/current/models/teacher.pt"
    assert mapped.binding == publication / "runtime/current/bindings/a.json"
    assert mapped.probe_motion == publication / "runtime/current/motions/a.npz"
    assert mapped.inspection_output == cfg.inspection_output == Path("new_inspection.json")
    assert mapped.query_output == cfg.query_output == Path("new_query.json")
    assert cfg.checkpoint == OLD_RUNTIME / "models/teacher.pt"
    assert module.inspect_model is inspect_model and module.native_binding is native_binding
    unchanged = module.Config(checkpoint=Path("relative/teacher.pt"), binding=Path("relative/binding.json"),
                              inspection_output=Path("out.json"))
    assert module._relocate_query_inputs(unchanged) is unchanged
