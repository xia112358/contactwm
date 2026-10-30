"""Playback remains available after trajectory editing/generation retirement."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from motion_edit.cli import build_parser
from motion_edit.web import server
from motion_edit.web.motion_data import load_contact_force_payload, load_motion_sequence


class WebMotionDataTests(unittest.TestCase):
    def test_loads_named_motion_and_eight_part_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.npz"
            qpos = np.zeros((3, 9), dtype=np.float32)
            force = np.zeros((3, 8, 3), dtype=np.float32)
            force[1, 0, 2] = 100.0
            raw_points = np.zeros((3, 2, 3), dtype=np.float32)
            raw_forces = np.zeros((3, 2, 3), dtype=np.float32)
            raw_points[1, 0] = [0.3, 0.4, 0.5]
            raw_forces[1, 0] = [4.0, 5.0, 6.0]
            np.savez(
                path,
                fps=np.asarray(50),
                joint_pos=qpos,
                joint_names=np.asarray(["joint_a", "joint_b"]),
                contact_force_part_order=np.asarray(["LHEE", "LTOE", "RHEE", "RTOE", "LH", "RH", "LK", "RK"]),
                contact_force_part_w=force,
                contact_force_part_mask=np.linalg.norm(force, axis=-1) > 0,
                contact_force_part_position_w=np.zeros_like(force),
                contact_force_part_position_valid=np.linalg.norm(force, axis=-1) > 0,
                raw_contact_count=np.asarray([0, 1, 0]),
                raw_contact_point0_w=raw_points,
                raw_contact_force_w=raw_forces,
            )
            loaded, fps, names = load_motion_sequence(path)
            contacts = load_contact_force_payload(path, frame_count=3)
        self.assertEqual(loaded.shape, (3, 9))
        self.assertEqual(fps, 50)
        self.assertEqual(names, ("joint_a", "joint_b"))
        self.assertEqual(len(contacts["part_order"]), 8)
        self.assertEqual(contacts["masks"], [])
        self.assertTrue(contacts["recorded_force_masks"][1][0])
        self.assertEqual(contacts["mask_status"], "recorded_force_mask_unverified")
        self.assertTrue(contacts["position_valid"][1][0])
        self.assertEqual(contacts["sample_points"][1], [[0.30000001192092896, 0.4000000059604645, 0.5]])
        self.assertEqual(contacts["sample_forces"][1], [[4.0, 5.0, 6.0]])


@pytest.mark.parametrize('command', ['generate-ref', 'edit-clip', 'edit-splice', 'generate-contact-jitter'])
def test_retired_commands_are_not_available(command):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command])
    assert error.value.code == 2


def test_review_cli_and_read_only_legacy_alias():
    parsed = build_parser().parse_args(['review', '--input', 'cases.json', '--no-browser'])
    assert parsed.input == 'cases.json' and parsed.no_browser
    alias = build_parser().parse_args(['contact-editor', '--review', 'motion.npz'])
    assert alias.input == 'motion.npz'
    with pytest.raises(SystemExit):
        build_parser().parse_args(['review', '--input', 'a.npz', '--motion-id', 'a'])


def test_playback_service_has_no_editing_or_generation_routes():
    app = server.create_app()
    with TestClient(app) as client:
        assert client.get('/api/health').json() == {'status': 'ok', 'mode': 'read_only'}
        assert client.get('/api/session').status_code == 404
        for action in ('generate', 'move', 'move-handle', 'save', 'validate', 'undo', 'redo', 'reset'):
            assert client.post('/api/session/' + action, json={}).status_code == 404
        assert client.get('/api/session/generation').status_code == 404
        assert client.put('/api/session/settings', json={}).status_code == 404
        assert client.get('/repo/.git/config').status_code == 404
        assert client.get('/repo/pyproject.toml').status_code == 404


def test_registered_motion_loads_read_only_without_edit_layer(monkeypatch):
    motion = server.PlaybackMotion('version-1', 'asset-1', 'version-1', 'motion.npz', 'augmented')
    asset = SimpleNamespace(motion_id='original', motion_asset_id='asset-1', terrain_mesh=None)
    monkeypatch.setattr(server, '_resolve_motion', lambda _: motion)
    monkeypatch.setattr(server, 'read_motion_asset', lambda _: asset)
    monkeypatch.setattr(server, 'upsert_recent_motion', lambda _: None)
    monkeypatch.setattr(server, '_registered_annotations', lambda *args: None)
    monkeypatch.setattr(server, 'load_review', lambda *a, **kw: {
        'read_only': True, 'capabilities': {'playback': True, 'edit_contacts': False, 'generate': False},
        'review': {'kind': 'motion'},
    })
    with TestClient(server.create_app()) as client:
        response = client.post('/api/session/load', json={'motion_id': 'version-1'})
        assert response.status_code == 200
        payload = response.json()
        assert payload['motion_id'] == 'version-1' and payload['provenance'] == 'augmented'
        assert payload['read_only'] is True
        assert not payload['capabilities']['generate']
        assert client.get('/api/session').json() == payload


def test_selected_asset_is_served_but_checkout_files_are_not(monkeypatch, tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    mesh = root / 'terrain.obj'
    mesh.write_text('v 0 0 0\n')
    secret = root / 'private.json'
    secret.write_text('{"private": true}')
    manifest = SimpleNamespace(root=root, logical_path=lambda path: Path(path).relative_to(root))
    monkeypatch.setattr(server.AssetManifest, 'load', lambda *a: manifest)
    monkeypatch.setattr(server, 'load_review', lambda *a, **kw: {
        'read_only': True, 'terrain_obj_url': kw['url'](mesh),
    })
    with TestClient(server.create_app(review_path='report.json')) as client:
        assert client.get('/repo/terrain.obj').text == 'v 0 0 0\n'
        assert client.get('/repo/private.json').status_code == 404


@pytest.mark.parametrize('case', ['matching', 'wrong_motion', 'wrong_clock', 'generated', 'hash_bound'])
def test_registered_force_sidecar_requires_exact_source_and_is_never_inherited(monkeypatch, tmp_path, case):
    import hashlib
    qpos = np.zeros((3, 9), dtype=np.float32)
    qpos[:, 3] = 1.
    names = np.asarray(['joint_a', 'joint_b'])
    source = tmp_path / 'motion.npz'
    np.savez(source, joint_pos=qpos, joint_names=names, fps=np.asarray(50.))
    force_qpos = qpos.copy()
    if case == 'wrong_motion':
        force_qpos[1, 0] = 0.1
    force = np.zeros((3, 1, 3), dtype=np.float32)
    force[1, 0, 2] = 123.
    sidecar = tmp_path / 'force.npz'
    fields = dict(joint_pos=force_qpos, joint_names=names,
                  fps=np.asarray(25. if case == 'wrong_clock' else 50.),
                  contact_force_part_order=np.asarray(['LH']), contact_force_part_w=force,
                  contact_force_part_mask=np.asarray([[False], [True], [False]]))
    if case == 'hash_bound':
        fields.pop('joint_pos'); fields.pop('joint_names')
        fields['source_sha256'] = np.asarray(hashlib.sha256(source.read_bytes()).hexdigest())
    np.savez(sidecar, **fields)
    motion = server.PlaybackMotion('motion-1', 'asset-1', 'version-1' if case == 'generated' else None,
                                   str(source), 'augmented' if case == 'generated' else 'source')
    asset = SimpleNamespace(motion_id='motion-1', motion_asset_id='asset-1', terrain_mesh=None,
                            source_contact_layer=None, surface_catalog_path=None, contact_force_npz=str(sidecar))
    monkeypatch.setattr(server, '_resolve_motion', lambda _: motion)
    monkeypatch.setattr(server, 'read_motion_asset', lambda _: asset)
    monkeypatch.setattr(server, 'upsert_recent_motion', lambda _: None)
    monkeypatch.setattr(server, 'load_review', lambda *args, **kwargs: {
        'qpos': qpos.tolist(), 'read_only': True, 'contact_force': {'forces': [], 'masks': []},
        'capabilities': {'playback': True, 'edit_contacts': False, 'generate': False},
        'review': {'kind': 'motion', 'notice': ''},
    })
    with TestClient(server.create_app()) as client:
        response = client.post('/api/session/load', json={'motion_id': 'motion-1'})
        assert response.status_code == 200, response.text
        contacts = response.json()['contact_force']
        if case in {'matching', 'hash_bound'}:
            assert contacts['forces'][1][0][2] == 123.
            assert contacts['masks'] == []  # Measured force cannot certify current contact.
            assert contacts['mask_status'] == 'recorded_force_mask_unverified'
        else:
            assert contacts['forces'] == []


@pytest.mark.parametrize('generated', [False, True])
def test_catalog_surfaces_and_historical_source_graph_remain_read_only(tmp_path, generated):
    import json
    source = tmp_path / 'motion.npz'
    np.savez(source, joint_pos=np.zeros((3, 9)), joint_names=np.asarray(['a', 'b']))
    layer = tmp_path / 'layer'
    for name in ('events', 'anchors', 'patches', 'transitions'):
        (layer / name).mkdir(parents=True)
        (layer / name / 'motion-1.jsonl').write_text('')
    (layer / 'transitions/motion-1.jsonl').write_text(json.dumps({
        'motion_id': 'motion-1', 'transition_id': 'step-1', 'start_frame': 0, 'end_frame': 2,
    }) + '\n')
    surfaces = tmp_path / 'surfaces.jsonl'
    surfaces.write_text(json.dumps({
        'motion_id': 'motion-1', 'surface_id': 'ground', 'object_id': None, 'surface_type': 'plane',
        'origin': [0, 0, 0], 'normal': [0, 0, 1], 'tangent_u': [1, 0, 0], 'tangent_v': [0, 1, 0],
    }) + '\n')
    asset = SimpleNamespace(motion_id='motion-1', motion_asset_id='asset-1', source_contact_layer=str(layer),
                            surface_catalog_path=str(surfaces), contact_force_npz=None)
    motion = server.PlaybackMotion('motion-1', 'asset-1', 'version-1' if generated else None, str(source), 'source')
    payload = {'qpos': [[0] * 9] * 3, 'graph': {'anchors': [], 'transitions': []},
               'review': {'notice': ''}, 'contact_force': {'masks': []}}
    server._registered_annotations(payload, motion, asset)
    assert payload['surfaces'][0]['surface_id'] == 'ground'
    assert payload['contact_force']['masks'] == []
    if generated:
        assert payload['graph']['transitions'] == []
    else:
        assert payload['graph']['transitions'][0]['transition_id'] == 'step-1'
        assert payload['review']['annotation_status'] == 'historical_catalog_annotations'
