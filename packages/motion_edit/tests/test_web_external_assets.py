"""Data relocation must preserve URLs without exposing undeclared directories."""
import json

import pytest
from fastapi.testclient import TestClient

from motion_edit.web import server


@pytest.fixture
def external_assets(tmp_path, monkeypatch):
    checkout, data = tmp_path / "code", tmp_path / "data"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "runtime").mkdir()
    data.mkdir()
    (checkout / "runtime/current").symlink_to(data, target_is_directory=True)
    (checkout / "configs/assets_manifest.json").write_text(json.dumps({
        "schema": "somaforge_asset_manifest_v1", "root": "..",
        "external_roots": ["runtime/current"], "assets": [],
    }))
    (data / "terrain.obj").write_text('v 0 0 0\n')
    (checkout / "source.txt").write_text("source fixture")
    secret = tmp_path / "secret.txt"
    secret.write_text("not public")
    (data / "escape.txt").symlink_to(secret)
    (checkout / "unregistered").symlink_to(data, target_is_directory=True)
    monkeypatch.setattr(server, "somaforge_root", lambda: checkout)
    return checkout, data, secret


def test_external_files_use_declared_logical_url_and_remain_readable(external_assets, monkeypatch):
    checkout, data, secret = external_assets
    assert server._repo_url(data / "terrain.obj") == "/repo/runtime/current/terrain.obj"
    # URL conversion alone must not expose checkout files. Assets are selected
    # by the validated review loader before the server allows their download.
    assert server._repo_url(checkout / "source.txt") == "/repo/source.txt"
    monkeypatch.setattr(server, "load_review", lambda *args, **kwargs: {
        "read_only": True, "asset_url": kwargs["url"](data / "terrain.obj")})
    with TestClient(server.create_app(review_path="selected-report.json")) as client:
        response = client.get(server._repo_url(data / "terrain.obj"))
        assert response.status_code == 200
        assert response.text == "v 0 0 0\n"
        assert client.get("/repo/source.txt").status_code == 404


def test_external_static_mount_rejects_symlink_and_directory_escapes(external_assets):
    checkout, data, secret = external_assets
    with pytest.raises(ValueError, match="outside declared roots"):
        server._repo_url(secret)
    with TestClient(server.create_app()) as client:
        for path in ("/repo/runtime/current/escape.txt", "/repo/unregistered/terrain.obj",
                     "/repo/runtime/current/%2e%2e/secret.txt"):
            assert client.get(path).status_code == 404
