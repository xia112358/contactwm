"""The maintained trainer imports without executing scratch training scripts."""
import subprocess
import sys
from unittest.mock import Mock


def test_training_helpers_do_not_import_scratch_trainers():
    subprocess.run([sys.executable, '-c', '''
import sys
from generator.training_data import prepare, take, box_obbs, split_name
from generator.scene_workers import Workers
from generator.research.isolated_gradients import audit
assert 'train_conditioned_pose_baseline' not in sys.modules
assert 'train_first_touch_geometry' not in sys.modules
assert 'newton_scene_router_audit' not in sys.modules
assert not any(n.startswith('climb00_pipeline') for n in sys.modules)
'''], check=True)


def test_box_chart_rotation_does_not_expand_terrain():
    import numpy as np
    from generator.training_data import box_obbs
    # A 45-degree chart of a square must keep the true edge lengths.
    polygon = np.array([[[0., 1.], [1., 0.], [0., -1.], [-1., 0.]]])
    center, rotation, half, ground = box_obbs(dict(
        box_edge_start=polygon, box_origin=np.array([[0., 0., 1.]]),
        box_basis=np.eye(3)[None], box_height=np.array([1.])))
    np.testing.assert_allclose(half, [[2**-.5, 2**-.5, .5]])
    np.testing.assert_allclose(center, [[0., 0., .5]])
    np.testing.assert_allclose(rotation[0].T @ rotation[0], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(ground, [0.])


def test_cached_side_cannot_be_used_as_box_top():
    import numpy as np
    import pytest
    from generator.training_data import box_obbs
    basis = np.array([[[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]]])
    with pytest.raises(ValueError, match='horizontal and upward'):
        box_obbs(dict(box_edge_start=np.array([[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]]),
            box_origin=np.array([[0., 0., .35]]), box_basis=basis, box_height=np.array([.35])))


def test_new_trainer_defaults_are_independent_and_build_without_old_weights():
    from dataclasses import asdict
    from generator.train_full1000_position import Config
    from generator.predictor_architecture import build_position_predictor
    from generator.structured_position_predictor import STRUCTURED_SCHEMA
    cfg = Config(architecture=STRUCTURED_SCHEMA, execution_observation_gradients=False,
                 width=24, layers=1, location_width=8)
    assert cfg.architecture == STRUCTURED_SCHEMA
    assert cfg.execution_observation_gradients is False
    assert cfg.execution_plan_gradients is False
    model = build_position_predictor(asdict(cfg))
    assert model.schema == STRUCTURED_SCHEMA
    assert set(model.training_parameter_groups()) == {'planner', 'executor'}


def test_shared_observation_training_flag_fails_before_output_or_workers(monkeypatch):
    import pytest
    from generator import train_full1000_position as module
    launcher = Mock()
    monkeypatch.setattr(module, 'parse_isaaclab_launcher_args', launcher)
    monkeypatch.setattr(sys, 'argv', ['train', '--architecture', 'full1000_position_predictor_v2', '--execution-observation-gradients'])
    with pytest.raises(ValueError, match='forbids shared'):
        module.main()
    launcher.assert_called_once()


def test_gradient_isolation_migration_requires_full_resume_before_workers(monkeypatch):
    import pytest
    from generator import train_full1000_position as module
    launcher = Mock()
    monkeypatch.setattr(module, 'parse_isaaclab_launcher_args', launcher)
    monkeypatch.setattr(sys, 'argv', ['train', '--resume-gradient-isolation'])
    with pytest.raises(ValueError, match='requires a full v1 resume'):
        module.main()
    launcher.assert_called_once()


def test_v1_scratch_defaults_roundtrip_with_planner_owned_encoder_and_new_interval():
    import io
    from dataclasses import asdict
    import torch
    from generator.train_full1000_position import Config
    from generator.predictor_architecture import build_position_predictor, load_position_predictor, LEGACY_SCHEMA
    from generator.predictor_initialization import initialize_predictor, state_fingerprint
    from somaforge_core.robot_assets import encode_robot_asset_json
    cfg = Config(width=24, layers=1, location_width=8)
    assert cfg.architecture == LEGACY_SCHEMA and cfg.shared_shape_target_interval
    assert not cfg.execution_observation_gradients and not cfg.execution_plan_gradients
    assert cfg.optimizer_gradient_ownership == 'planner_supervised_v1'
    model = build_position_predictor(asdict(cfg), schema=cfg.architecture).eval()
    assert set(model.training_parameter_groups()) == {'planner', 'executor'}
    before = state_fingerprint(model)
    _, loaded, _, provenance = initialize_predictor(model, seed=cfg.seed)
    assert not loaded and not provenance['predictor_checkpoints_loaded']
    assert provenance['optimizer_resumed'] is False
    assert before == state_fingerprint(model)
    stream = io.BytesIO()
    torch.save(dict(schema=cfg.architecture, architecture_contract=model.architecture_contract(),
        config=asdict(cfg), model=model.state_dict(), robot_asset_json=encode_robot_asset_json()), stream)
    stream.seek(0)
    restored = load_position_predictor(torch.load(stream, weights_only=False)).eval()
    assert state_fingerprint(restored) == before
    assert restored.architecture_contract() == model.architecture_contract()


def test_historical_baseline_recipe_and_manifests_remain_consistent():
    import json
    from dataclasses import asdict
    from generator.train_full1000_position import Config, ROOT

    pointer = json.loads((ROOT / 'baselines/current.json').read_text())
    baseline = json.loads((ROOT / pointer['record']).read_text())
    recorded = json.loads((ROOT / pointer['training_config']).read_text())
    defaults = asdict(Config())
    # The designated historical baseline is immutable; new scratch defaults
    # must not rewrite it to claim it used the new gradient-isolation policy.
    assert recorded['architecture'] == baseline['architecture']['schema']
    for key in ('width', 'layers', 'location_width', 'execution_observation_gradients',
                'execution_plan_gradients'):
        assert recorded[key] == baseline['architecture'][key]
    assert recorded['execution_observation_gradients'] is True
    assert recorded['keep_patch_weight'] == baseline['training_course']['keep_patch_weight']
    assert recorded['steps'] == baseline['training_course']['planned_total_intervals']
    assert recorded['acceptance_penetration_m'] == baseline['acceptance']['shallow_penetration_m']
    assert baseline['id'] == pointer['id']
    assert baseline['status'] == 'current_training_and_analysis_baseline'
    assert baseline['checkpoint']['step'] == 1000
    assert baseline['training_result']['status'] == 'incomplete'
    pipeline = json.loads((ROOT / 'configs/training_pipeline_manifest.json').read_text())
    predictor = pipeline['stages']['predictor']
    assert predictor['baseline_asset_id'] == pointer['id']
    assert predictor['baseline_record'] == pointer['record']
    assert predictor['architecture_schema'] == defaults['architecture']
    assets = json.loads((ROOT / 'configs/assets_manifest.json').read_text())['assets']
    active = [asset for asset in assets if asset['status'] == 'current_training_and_analysis_baseline']
    assert len(active) == 1 and active[0]['id'] == pointer['id']
    for archived in json.loads((ROOT / pointer['archives']).read_text())['records']:
        if 'record' in archived:
            assert json.loads((ROOT / archived['record']).read_text())['status'] == 'archived'


def test_formal_rollout_entry_uses_versioned_position_reader():
    from generator import evaluate_position_rollout as module
    from generator.predictor_architecture import load_position_predictor, POSITION_SCHEMAS
    assert module.load_position_predictor is load_position_predictor
    assert module.POSITION_SCHEMAS == POSITION_SCHEMAS
    assert module.ROOT.joinpath('packages/generator/generator').is_dir()
