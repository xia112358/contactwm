"""Relocated certificates and cache identities still verify original bytes."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from generator.support_path_acceptance import SCHEMA as CERTIFICATE_SCHEMA, validate_support_paths
from somaforge_core.contact_dataset import contact_dataset_fingerprint, validate_contact_cache
from somaforge_core.demonstration_events import SCHEMA as EVENT_SCHEMA
from somaforge_core.published_data import SCHEMA as MAPPING_SCHEMA


def evidence(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def install_copy_mapping(tmp_path, monkeypatch, original):
    restored = tmp_path / "different_disk/restored"
    shutil.copytree(original, restored)
    checkout = tmp_path / "different_checkout"
    monkeypatch.setenv("SOMAFORGE_ROOT", str(checkout))
    mapping = checkout / "runtime/current/manifests/published_data_paths.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps({"schema": MAPPING_SCHEMA, "prefixes": [
        {"source": str(original), "target": str(restored)}]}))
    return restored


def test_relocated_certificate_keeps_source_hashes_and_rejects_changed_evidence(tmp_path, monkeypatch):
    original = tmp_path / "source_workstation"
    original.mkdir()
    motion = original / "motion.npz"
    labels = original / "labels.npz"
    motion.write_bytes(b"motion identity fixture")
    labels.write_bytes(b"recorded label identity fixture")
    intervals = [{"start_frame": 0, "end_frame": 10}, {"start_frame": 10, "end_frame": 20}]
    clock = original / "clock.json"
    clock.write_text(json.dumps({"intervals": intervals}))
    events = [{"schema": EVENT_SCHEMA, "source_audit": str(motion), **row} for row in intervals]
    event_file = original / "events.jsonl"
    event_file.write_text("".join(json.dumps(row) + "\n" for row in events))
    certificate = original / "certificate.json"
    certificate.write_text(json.dumps({"schema": CERTIFICATE_SCHEMA, "frame_count": 21,
        "evidence": {"motion": evidence(motion), "labels": evidence(labels),
                     "events": evidence(event_file), "action_clock": evidence(clock)},
        "segments": events}))
    manifest = original / "manifest.json"
    manifest.write_text(json.dumps({"demonstration_event_evidence": [{**evidence(certificate), "motion_id": 0}],
        "motion_files": [{"motion_file": str(motion), "newton_contact_file": str(labels),
                          "event_segments_file": str(event_file),
                          "event_segments_sha256": evidence(event_file)["sha256"]}]}))
    samples = [{"motion_id": 0, "current_frame": 0, "target_frame": 10},
               {"motion_id": 0, "current_frame": 10, "target_frame": 20}]
    before = {path.name: path.read_bytes() for path in original.iterdir()}
    baseline = validate_support_paths(manifest, samples)
    restored = install_copy_mapping(tmp_path, monkeypatch, original)
    report = validate_support_paths(manifest, samples)
    assert report["status"] == baseline["status"]
    assert report["event_count"] == 2
    assert report["certificates"] == [str(restored / "certificate.json")]
    assert {path.name: path.read_bytes() for path in restored.iterdir()} == before
    assert {path.name: path.read_bytes() for path in original.iterdir()} == before
    # Changing the relocated source cannot be hidden by an intact old copy.
    (restored / "labels.npz").write_bytes(b"changed label evidence")
    with pytest.raises(ValueError, match="Stale demonstration evidence"):
        validate_support_paths(manifest, samples)


def test_relocated_dataset_preserves_cache_fingerprint_and_geometry_checks(tmp_path, monkeypatch):
    # These files test content identity, not fabricated Newton admission.
    original = tmp_path / "source_dataset"
    original.mkdir()
    motion = original / "motion.npz"
    np.savez(motion, fps=np.array(50.), joint_pos=np.zeros((2, 36)))
    labels = original / "labels.npz"
    labels.write_bytes(b"fixed label content")
    terrain = original / "terrain.obj"
    terrain.write_text("v 0 0 0\n")
    catalog = original / "surfaces.jsonl"
    catalog.write_text('{"surface_id":"ground"}\n')
    plan = original / "plan.json"
    plan.write_text(json.dumps({"metadata": {"target_surface_catalog": str(catalog)}}))
    manifest = original / "manifest.json"
    manifest.write_text(json.dumps({"motion_files": [{"motion_id": 0, "terrain_id": 0,
        "motion_file": str(motion), "newton_contact_file": str(labels), "edit_plan_file": str(plan)}],
        "terrains": [{"terrain_id": 0, "terrain_file": str(terrain)}]}))
    fingerprint = contact_dataset_fingerprint(manifest)
    cache = {"contact_dataset_fingerprint": fingerprint}
    before = {path.name: path.read_bytes() for path in original.iterdir()}
    restored = install_copy_mapping(tmp_path, monkeypatch, original)
    assert contact_dataset_fingerprint(manifest) == fingerprint
    assert contact_dataset_fingerprint(restored / "manifest.json") == fingerprint
    validate_contact_cache(cache, manifest)
    assert {path.name: path.read_bytes() for path in restored.iterdir()} == before
    assert {path.name: path.read_bytes() for path in original.iterdir()} == before
    (restored / "surfaces.jsonl").write_text('{"surface_id":"changed"}\n')
    with pytest.raises(ValueError, match="Cache contact dataset/policy mismatch"):
        validate_contact_cache(cache, manifest)


def test_trainer_relocates_parsed_path_fields_before_data_access(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from generator import train_full1000_position as trainer

    original = tmp_path / "original_inputs"
    original.mkdir()
    restored = install_copy_mapping(tmp_path, monkeypatch, original)
    names = ("manifest", "data_cache", "q_cache", "query_checkpoint",
             "query_source_manifest", "query_source_model", "static_audit_pool")
    inputs = {name: original / name for name in names}
    output = tmp_path / "new_experiment"
    cfg = trainer.Config(**inputs, output=output, gpu_pipeline=False)
    monkeypatch.setattr(trainer, "parse_isaaclab_launcher_args", lambda description: SimpleNamespace())
    monkeypatch.setattr(trainer.tyro, "cli", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["train"])
    with pytest.raises(ValueError, match="Endpoint training requires complete solids"):
        trainer.main()
    assert {name: getattr(cfg, name) for name in names} == {name: restored / name for name in names}
    assert cfg.output == output
    assert not output.exists()
