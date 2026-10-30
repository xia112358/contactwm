"""Relocation fixes saved input identities without relaxing input checks."""
import json
import shutil

import torch

from generator.predictor_runtime import evidence
from somaforge_core import published_data


def relocated_copy(tmp_path, monkeypatch, source):
    target = tmp_path / "other_disk/inputs"
    shutil.copytree(source, target)
    mapping = tmp_path / "installed_mapping.json"
    mapping.write_text(json.dumps({"schema": published_data.SCHEMA, "prefixes": [
        {"source": str(source), "target": str(target)}]}))
    # Keep the real canonical asset root; only this test's install metadata is
    # redirected. Asset identity validation itself is never mocked.
    monkeypatch.setattr(published_data, "_mapping_path", lambda: mapping)
    return target


def test_checkpoint_and_parent_corpus_compare_actual_relocated_file_evidence(tmp_path, monkeypatch):
    source = tmp_path / "original"
    source.mkdir()
    predictor = source / "predictor.pt"
    predictor.write_bytes(b"fixed predictor content for evidence hashing")
    parent = source / "parent.pt"
    torch.save({"records": [], "predictor": evidence(predictor)}, parent)
    corpus = source / "extended.pt"
    torch.save({"records": [], "predictor": evidence(predictor),
                "parent_corpus": evidence(parent)}, corpus)
    checkpoint = source / "checkpoint.pt"
    torch.save({"corpus": evidence(parent), "predictor": evidence(predictor),
                "model": {"weight": torch.arange(3.)}, "optimizer": {"step": torch.tensor(4)},
                "scene_fingerprints": {"0": "immutable-native-fingerprint"},
                "config": {"corpus": corpus}}, checkpoint)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    target = relocated_copy(tmp_path, monkeypatch, source)
    saved = published_data.load_published_tensor_input(checkpoint)
    data = published_data.load_published_tensor_input(corpus)
    assert saved["predictor"] == evidence(target / "predictor.pt") == data["predictor"]
    assert saved["corpus"] == data["parent_corpus"] == evidence(target / "parent.pt")
    assert saved["config"]["corpus"] == target / "extended.pt"
    assert saved["scene_fingerprints"] == {"0": "immutable-native-fingerprint"}
    torch.testing.assert_close(saved["model"]["weight"], torch.arange(3.))
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    (target / "parent.pt").write_bytes(b"changed parent corpus")
    assert saved["corpus"] != evidence(target / "parent.pt")


def test_loader_does_not_traverse_model_optimizer_or_record_payloads(monkeypatch):
    weight = torch.arange(4.)
    model = {"weight": weight}
    optimizer = {"state": {0: {"exp_avg": weight}}}
    records = [{"q": weight, "source_record": "/unchanged/record/metadata"}]
    payload = {"model": model, "optimizer": optimizer, "records": records,
               "scene_fingerprints": {"0": "same"}, "predictor": {"path": "/old/predictor.pt", "sha256": "same-hash"}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    loaded = published_data.load_published_tensor_input("/unused/input.pt")
    assert loaded["model"] is model
    assert loaded["optimizer"] is optimizer
    assert loaded["records"] is records
    assert loaded["model"]["weight"] is weight
