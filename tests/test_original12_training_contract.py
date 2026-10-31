"""Exact original task physics and CLI wiring, without importing CUDA or a simulator."""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'packages/supermimic')]
import original12
from supermimic.contact_response import contract, metadata, validate_identity
from supermimic.original12_check import validate_state


def test_actual_make_sim_wires_newton_collisions_at_every_two_ms_substep():
    tree = ast.parse((ROOT/'packages/supermimic/supermimic/lowstep.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'make_sim')
    cfg = SimpleNamespace(physics=SimpleNamespace(solver_cfg=SimpleNamespace(use_mujoco_contacts=True), collision_decimation=10))
    namespace = dict(contact_response='newton_contacts_v2', response_contract=contract,
                     original_make_sim=lambda *args: cfg)
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), '<actual make_sim>', 'exec'), namespace)
    result = namespace['make_sim']('cuda:0', SimpleNamespace())
    assert result.physics.solver_cfg.use_mujoco_contacts is False
    assert result.physics.collision_decimation == 1
    assert result.physics.num_substeps == 10
    assert result.physics.solver_cfg.enable_multiccd is True
    assert contract('newton_contacts_v2')['control_dt'] / result.physics.num_substeps == .002
    with pytest.raises(ValueError, match='physics changed'):
        validate_identity({'contact_response_contract': metadata('manifold_v2')},
                          {'contact_response_contract': metadata('newton_contacts_v2')})


def test_no_reset_noise_reaches_the_actual_environment_constructor():
    tree = ast.parse((ROOT/'packages/supermimic/supermimic/train.py').read_text())
    parser = argparse.ArgumentParser()
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--no_reset_noise')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=copy.deepcopy(node))], type_ignores=[])), '<actual parser>', 'exec'), {'parser': parser})
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'TrackingSim')
    for flags, expected in [([], True), (['--no_reset_noise'], False)]:
        args = parser.parse_args(flags);args.num_envs=4096;args.gamma=.99
        captured = []
        namespace = dict(args=args, sim_cfg=None, motion=None, mdp=None,
                         TrackingSim=lambda *a, **kw: captured.append(kw))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=copy.deepcopy(call))], type_ignores=[])), '<actual constructor>', 'exec'), namespace)
        assert captured[0]['reset_noise'] is expected


def test_checkpoint_contract_cannot_substitute_old_physics_or_incomplete_weights():
    recipe = json.loads((ROOT/'configs/experiments/original12_newton2ms.json').read_text())
    state = dict.fromkeys(('pi','wm','critic','opt_pi','opt_wm','opt_v','reset_sampler_state'))
    state.update(iteration=1500, args=dict(no_reset_noise=True,contact_response='newton_contacts_v2'),
                 task_manifest_sha256=recipe['task_manifest_sha256'],contact_response_contract=metadata('newton_contacts_v2'))
    assert validate_state(state,recipe)==1500
    with pytest.raises(ValueError):
        validate_state(dict(state, contact_response_contract=metadata('manifold_v2')),recipe)
    with pytest.raises(ValueError,match='Incomplete'):
        validate_state({k:v for k,v in state.items() if k!='opt_wm'},recipe)


def test_portable_recipe_contains_complete_recorded_flags_and_published_paths():
    args = SimpleNamespace(resources=None, output=None, command='tracking',resume_state='initialization',startup_check=False)
    result = original12.plan(args,environ={})
    argv = result['launch'].argv
    assert argv[argv.index('--contact_response')+1]=='newton_contacts_v2'
    assert '--no_reset_noise' in argv
    assert argv[argv.index('--num_envs')+1]=='4096'
    assert argv[argv.index('--resume')+1]==str(ROOT/'runtime/current/models/original12/initialization.pt')
    assert not any('/perception/' in x or '/home/zhouxia/workspace/' in x for x in argv)
    assert 'CUDA_VISIBLE_DEVICES' not in result['launch'].environment
    args.command='prepare';args.skip_environment=False;args.assets=None
    preparation=original12.plan(args,environ={})
    assert any('setup_workflow_environment.py' in item for item in preparation['commands'][0])
    assert all('original12-training' in command for command in preparation['commands'][1:])


def test_all_workflow_parsers_accept_original_physics_without_native_imports():
    for module in ('train', 'evaluate', 'contact_collect', 'contact_initialize', 'contact_train'):
        tree = ast.parse((ROOT/'packages/supermimic/supermimic'/f'{module}.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--contact_response')
        choices = next(k.value for k in node.keywords if k.arg == 'choices')
        assert 'newton_contacts_v2' in ast.literal_eval(choices), module
