"""Information identity, branch isolation and versioned serialization regressions."""
import copy
import io

import pytest
import torch

from generator.structured_position_predictor import StructuredPositionPredictor, JointPoseHead, STRUCTURED_SCHEMA
from generator.predictor_architecture import build_position_predictor, load_position_predictor
from generator.predictor_initialization import initialize_training_joint_bias, state_fingerprint
from somaforge_core.robot_assets import encode_robot_asset_json
from somaforge_core.heightmap import HEIGHTMAP_ROWS, HEIGHTMAP_COLS
from somaforge_core.g1_kinematics import _quaternion_multiply_wxyz


@pytest.fixture
def setup():
    torch.manual_seed(47)
    model = StructuredPositionPredictor(24, 1, 8, region_plan=True, unified_contact=True, event_roles=True).eval()
    q = torch.zeros(2, 36); q[:, 2] = .8; q[:, 3] = 1.
    q[1, 7:] = .03
    inputs = dict(current_q=q, current_contact=torch.zeros(2, 6, dtype=torch.bool),
                  current_anchor=torch.zeros(2, 6, 3),
                  heightmap=torch.full((2, HEIGHTMAP_ROWS, HEIGHTMAP_COLS), -.8))
    roles = torch.tensor([[1, 2, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]])
    regions = torch.zeros(2, 6, 4, dtype=torch.bool); regions[:, :2, 0] = True
    teacher = dict(teacher_role=roles, teacher_contact=roles != 0,
                   teacher_cell=torch.full((2, 6), 123, dtype=torch.long),
                   teacher_regions=regions, teacher_mask=torch.ones(2, dtype=torch.bool))
    return model, inputs, teacher


def checkpoint(model):
    contract = model.architecture_contract()
    config = {key: contract[key] for key in ('width', 'layers', 'location_width', 'body_geometry',
        'part_geometry', 'region_plan', 'unified_contact', 'event_roles')}
    config['architecture'] = STRUCTURED_SCHEMA
    return dict(schema=STRUCTURED_SCHEMA, architecture_contract=contract, config=config,
                model=model.state_dict(), robot_asset_json=encode_robot_asset_json())


def test_encoders_and_parameter_groups_have_no_shared_storage(setup):
    model, _, _ = setup
    groups = model.training_parameter_groups()
    assert set(groups) == {'planner', 'executor'}
    assert {id(p) for p in groups['planner']}.isdisjoint(id(p) for p in groups['executor'])
    assert len(groups['planner']) + len(groups['executor']) == len(list(model.parameters()))
    planner = list(model.planner_observation.parameters())
    executor = list(model.executor_observation.parameters())
    assert {p.data_ptr() for p in planner}.isdisjoint(p.data_ptr() for p in executor)
    assert any(not torch.equal(a, b) for a, b in zip(planner, executor))


def test_execution_and_planning_gradients_are_isolated(setup):
    model, inputs, teacher = setup
    before = state_fingerprint(model)
    prediction = model(**inputs, **teacher)
    groups = model.training_parameter_groups()
    parameters = groups['planner'] + groups['executor']; cut = len(groups['planner'])
    losses = [prediction.qpos.square().sum(),
              prediction.role_logits.square().sum() + prediction.location_logits.square().sum()
              + prediction.region_logits.square().sum()]
    for loss, forbidden, allowed in [(losses[0], slice(0, cut), slice(cut, None)),
                                     (losses[1], slice(cut, None), slice(0, cut))]:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        assert all(g is None for g in gradients[forbidden])
        assert any(g is not None and g.abs().sum() > 0 for g in gradients[allowed])
    assert before == state_fingerprint(model)
    assert all(p.grad is None for p in parameters)


def test_joint_readout_does_not_broadcast_one_pooled_feature():
    head = JointPoseHead(24)
    features = torch.randn(2, 29, 24, requires_grad=True)
    first = head(features)
    changed = features.detach().clone(); changed[:, 3] += 1.
    second = head(changed)
    others = [j for j in range(29) if j != 3]
    torch.testing.assert_close(first[:, others], second[:, others], rtol=0, atol=0)
    assert not torch.equal(first[:, 3], second[:, 3])
    gradient, = torch.autograd.grad(first[:, 3].sum(), features)
    assert torch.count_nonzero(gradient[:, others]) == 0
    assert gradient[:, 3].abs().sum() > 0


def test_current_observation_and_future_intent_keep_distinct_token_positions(setup):
    model, inputs, teacher = setup
    seen = {}
    hook = model.execution_read.register_forward_pre_hook(
        lambda module, args: seen.update(query=args[0].detach(), memory=args[1].detach()))
    prediction = model(**inputs, **teacher)
    hook.remove()
    assert seen['query'].shape == (2, 30, 24)
    # root1 + pose6 + actual contact6 + joints29 + physical geometry6 + intent6
    assert seen['memory'].shape == (2, 54, 24)
    assert not torch.equal(seen['memory'][:, 7:13], seen['memory'][:, -6:])
    assert torch.equal(prediction.conditioned_role, teacher['teacher_role'])
    assert torch.equal(prediction.planned_regions, teacher['teacher_regions'])
    changed = dict(teacher, teacher_role=teacher['teacher_role'].flip(0))
    other = model(**inputs, **changed)
    torch.testing.assert_close(prediction.role_logits, other.role_logits, rtol=0, atol=0)
    assert not torch.equal(prediction.qpos, other.qpos)


def test_inactive_cells_and_inactive_current_anchors_do_not_change_execution(setup):
    model, inputs, teacher = setup
    first = model(**inputs, **teacher)
    changed = dict(teacher); changed['teacher_cell'] = teacher['teacher_cell'].clone()
    changed['teacher_cell'][:, 2:] += 100
    altered = dict(inputs, current_anchor=torch.full_like(inputs['current_anchor'], 100.))
    second = model(**altered, **changed)
    torch.testing.assert_close(first.qpos, second.qpos, rtol=0, atol=0)


def test_versioned_save_reload_and_contract_rejection(setup):
    model, inputs, teacher = setup
    buffer = io.BytesIO(); torch.save(checkpoint(model), buffer); buffer.seek(0)
    loaded = load_position_predictor(torch.load(buffer, weights_only=False)).eval()
    first, second = model(**inputs, **teacher), loaded(**inputs, **teacher)
    for field in first.__dataclass_fields__:
        a, b = getattr(first, field), getattr(second, field)
        if a is not None: torch.testing.assert_close(a, b, rtol=0, atol=0)
    for mutate in ('contract', 'schema', 'config', 'weights', 'missing_contract'):
        bad = copy.deepcopy(checkpoint(model))
        if mutate == 'contract': bad['architecture_contract']['joints'].reverse()
        if mutate == 'schema': bad['schema'] = 'full1000_position_predictor_v1'
        if mutate == 'config': bad['config']['architecture'] = 'full1000_position_predictor_v1'
        if mutate == 'weights': bad['model'].pop('root_pose_head.weight')
        if mutate == 'missing_contract': bad.pop('architecture_contract')
        with pytest.raises((ValueError, RuntimeError)):
            load_position_predictor(bad)
    with pytest.raises(ValueError, match='explicit v2'):
        build_position_predictor(dict(width=24, layers=1))


def test_joint_mean_bias_keeps_root_and_other_parameters_unchanged(setup):
    model, inputs, _ = setup
    before = {k: v.clone() for k, v in model.state_dict().items()}
    q = inputs['current_q'].clone(); q[:, 7:] = 2.
    initialize_training_joint_bias(model, q, torch.tensor([0]))
    for key, value in model.state_dict().items():
        expected = torch.full((29,), 2.) if key == 'joint_pose_head.bias' else before[key]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_translation_and_yaw_equivariance(setup):
    model, inputs, teacher = setup
    first = model(**inputs, **teacher)
    angle = .61
    quat = torch.tensor([torch.cos(torch.tensor(angle/2)), 0., 0., torch.sin(torch.tensor(angle/2))])
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    shift = torch.tensor([2., -1., 0.])
    def transform(q):
        result = q.clone(); result[:, :3] = q[:, :3] @ rotation.T + shift
        result[:, 3:7] = _quaternion_multiply_wxyz(quat.expand(len(q), -1), q[:, 3:7])
        return result
    moved = dict(inputs, current_q=transform(inputs['current_q']),
                 current_anchor=inputs['current_anchor'] @ rotation.T + shift)
    second = model(**moved, **teacher)
    torch.testing.assert_close(second.qpos, transform(first.qpos), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(second.role_logits, first.role_logits, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('flags', [{}, {'unified_contact': True}, {'part_geometry': True}, {'body_geometry': True}])
def test_supported_observation_variants_keep_finite_output(setup, flags):
    _, inputs, _ = setup
    model = StructuredPositionPredictor(24, 1, 8, **flags).eval()
    prediction = model(**inputs)
    assert prediction.qpos.shape == (2, 36)
    assert torch.isfinite(prediction.qpos).all()


@pytest.mark.parametrize('flag', ['execution_observation_gradients', 'execution_plan_gradients'])
def test_shared_gradient_flags_cannot_enter_new_model(flag):
    with pytest.raises(ValueError, match='cannot share'):
        StructuredPositionPredictor(24, 1, **{flag: True})


def test_batch_members_do_not_mix_in_forward_or_input_gradients(setup):
    model, inputs, teacher = setup
    observed = dict(inputs, current_q=inputs['current_q'].clone().requires_grad_(True))
    result = model(**observed, **teacher)
    gradient, = torch.autograd.grad(result.qpos[0].square().sum(), observed['current_q'])
    assert torch.count_nonzero(gradient[1]) == 0
    single = model(**{k: v[:1] for k, v in inputs.items()}, **{k: v[:1] for k, v in teacher.items()})
    torch.testing.assert_close(result.qpos[:1], single.qpos, atol=1e-7, rtol=1e-5)
    reversed_result = model(**{k: v.flip(0) for k, v in inputs.items()},
                            **{k: v.flip(0) for k, v in teacher.items()})
    torch.testing.assert_close(result.qpos.flip(0), reversed_result.qpos, atol=1e-7, rtol=1e-5)


def test_new_model_keeps_actual_teacher_validation(setup):
    model, inputs, teacher = setup
    missing = dict(teacher); missing.pop('teacher_regions')
    with pytest.raises(ValueError, match='actual demonstrated regions'):
        model(**inputs, **missing)
    inconsistent = dict(teacher, teacher_contact=torch.zeros_like(teacher['teacher_contact']))
    with pytest.raises(ValueError, match='role/contact mismatch'):
        model(**inputs, **inconsistent)
    inconsistent = dict(teacher, teacher_regions=torch.zeros_like(teacher['teacher_regions']))
    with pytest.raises(ValueError, match='regions/contact mismatch'):
        model(**inputs, **inconsistent)


def test_explicit_legacy_reader_preserves_old_forward(setup):
    from generator.full1000_position_predictor import Full1000PositionPredictor
    _, inputs, _ = setup
    config = dict(width=24, layers=1, location_width=8, region_plan=True, unified_contact=True,
                  event_roles=True, execution_observation_gradients=True)
    legacy = Full1000PositionPredictor(**config).eval()
    old = dict(schema='full1000_position_predictor_v1', config=config,
               model=legacy.state_dict(), robot_asset_json=encode_robot_asset_json())
    loaded = load_position_predictor(old).eval()
    assert isinstance(loaded, Full1000PositionPredictor)
    expected, actual = legacy(**inputs), loaded(**inputs)
    for key in expected.__dataclass_fields__:
        a, b = getattr(expected, key), getattr(actual, key)
        if a is not None: torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_versioned_gradient_audit_never_constructs_optimizer_or_imports_archive(setup, monkeypatch):
    """Mock witness objective checks audit plumbing, not physical Newton validation."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from generator.research import isolated_gradients as module
    model, inputs, _ = setup
    payload = checkpoint(model); payload['step'] = 0
    reports = {}
    monkeypatch.setattr(torch, 'load', lambda *a, **k: payload)
    monkeypatch.setattr(torch, 'save', lambda *a, **k: None)
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes',
        lambda p: b'fixture' if p == Path('fixtures/audit_fixture.pt') else read_bytes(p))
    monkeypatch.setattr(Path, 'write_text', lambda p, text: reports.update({p.name: json.loads(text)}))
    monkeypatch.setattr(module, 'write_witness_manifest', lambda cfg: None)
    def forbidden_optimizer(*args, **kwargs):
        pytest.fail('Read-only audit cannot construct an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden_optimizer)
    cfg = SimpleNamespace(audit_checkpoint=Path('fixtures/audit_fixture.pt'), output=Path('fixtures/audit_fixture'),
                          static_audit_samples=2, static_audit_pool=None)
    def objective(prediction, observation, ids, labels):
        plan = prediction.role_logits.square().mean((1, 2))
        execution = prediction.qpos.square().mean(-1)
        metrics = dict(audit_planner=plan, unified_region_loss=execution,
                       audit_imitation=execution, audit_layout=execution,
                       region_gradient_valid=torch.ones(2, dtype=torch.bool))
        queried = SimpleNamespace(q=prediction.qpos, world_frame={}, pair={'dist': torch.zeros(0)})
        return plan + execution, metrics, (queried, {})
    module.audit(model, cfg, inputs, torch.arange(2), lambda *a, **k: {}, objective)
    report = reports['isolation_audit.json']
    assert report['optimizer_constructed'] is False and report['parameter_updates'] == 0
    assert report['weights_bitwise_unchanged']
    components = report['records'][0]['components']
    assert components['planner']['gradient_norms']['executor'] == 0.
    assert components['contact_execution']['gradient_norms']['planner'] == 0.
