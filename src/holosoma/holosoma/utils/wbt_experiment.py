"""Compose native manager configurations for independently invoked WBT experiments.

No training orchestration, checkpoint selection or simulator initialization lives here.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

ROUTED_ALGO = "holosoma.agents.ppo.routed_distill_ppo.RoutedDistillPPO"
SEGMENT_DISTILL_ALGO = "holosoma.php_contact.distill.SegmentDistillPPO"
REFERENCE_SCHEMA = "somaforge_wbt_observed_contact_reference_v1"
PARTS = ("left_foot", "right_foot", "left_hand", "right_hand", "left_knee", "right_knee")
STUDENT_FUNCTIONS = frozenset({
    "holosoma.managers.observation.terms.wbt:actions",
    "holosoma.managers.observation.terms.wbt:base_ang_vel",
    "holosoma.managers.observation.terms.wbt:base_lin_vel",
    "holosoma.managers.observation.terms.wbt:dof_pos",
    "holosoma.managers.observation.terms.wbt:dof_vel",
    "holosoma.managers.observation.terms.wbt:projected_gravity",
    "holosoma.managers.observation.terms.wbt:PHPDepthImage",
    "holosoma.managers.observation.terms.wbt_experiment:velocity_command",
    "holosoma.managers.observation.terms.wbt_experiment:actual_contacts",
})


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contact_reference(path, *, collision_profile=None):
    """Reject legacy force-threshold labels and ambiguous temporal alignment."""
    import json
    from somaforge_core.contact_face_selection import FACE_POLICY
    from somaforge_core.robot_assets import validate_g1_asset_metadata

    with np.load(path, allow_pickle=False) as data:
        required = {"observed_contact_active", "observed_contact_force_w", "observed_contact_provenance_json"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: missing native contact reference fields {sorted(required-set(data.files))}")
        meta = json.loads(str(data["observed_contact_provenance_json"].item()))
        active = data["observed_contact_active"].copy()
        force = data["observed_contact_force_w"].copy()
    validate_g1_asset_metadata(meta.get("robot_asset"), context=str(path))
    if collision_profile is not None:
        from somaforge_core.collision_profiles import require_profile
        require_profile(meta.get("robot_asset"), collision_profile, context=str(path))
    if meta.get("schema") != REFERENCE_SCHEMA or meta.get("face_selection") != FACE_POLICY:
        raise ValueError(f"{path}: incompatible native contact reference semantics")
    if tuple(meta.get("parts", ())) != PARTS or meta.get("sampling") != "last_native_solve_before_policy_observation":
        raise ValueError(f"{path}: missing contact ordering or solver sampling contract")
    if meta.get("activation") != "CONSTRAINT_and_dist_lt_includemargin_with_allocated_efc":
        raise ValueError(f"{path}: contact activation is not native constraint truth")
    if active.ndim != 2 or active.shape[1] != len(PARTS) or force.shape != (*active.shape, 3):
        raise ValueError(f"{path}: invalid contact reference shape")
    if not np.isfinite(force).all() or not np.isin(active, [0, 1]).all():
        raise ValueError(f"{path}: nonfinite force or invalid contact states")
    return active.astype(bool), force.astype(np.float32), meta


def assert_student_observations(config):
    """Check the entire actor input graph, including depth and history groups."""
    for key in config.algo.config.module_dict.actor.input_dim:
        group = config.observation.groups[key]
        for name, term in group.terms.items():
            segment_commands = {
                'holosoma.php_contact.segments:future_contacts',
                'holosoma.php_contact.segments:goal',
                'holosoma.php_contact.terms:height_scan',
            } if config.algo._target_ == SEGMENT_DISTILL_ALGO else set()
            if term.func not in STUDENT_FUNCTIONS | segment_commands:
                raise ValueError(f"Student actor input {key}/{name} is not an approved measured observation: {term.func}")


def prepare_routed_distillation(config):
    """Import each explicitly provided teacher's native observation layout.

    The caller supplies a combined manifest. Motions are routed by canonical
    file identity, never inferred from filenames, task labels or checkpoint rank.
    """
    if config.algo._target_ not in (ROUTED_ALGO, SEGMENT_DISTILL_ALGO):
        return config
    import torch
    from holosoma.config_types.observation import ObsGroupCfg, ObsTermCfg
    from holosoma.utils.eval_utils import _experiment_config_from_serialized
    from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
    from somaforge_core.published_data import load_published_json, resolve_published_path
    from somaforge_core.robot_assets import validate_g1_asset_metadata

    distill = config.algo.config.distill
    if distill.teacher_spec_file:
        from holosoma.config_types.algo import TeacherActorConfig
        spec_path = resolve_published_path(distill.teacher_spec_file)
        payload = load_published_json(spec_path)
        rows = payload["teachers"] if isinstance(payload, dict) else payload
        def resolve(value):
            return str(resolve_published_path(value, base_dir=spec_path.parent))
        specs = tuple(TeacherActorConfig(checkpoint=resolve(row["checkpoint"]),
            manifest=resolve(row["manifest"])) for row in rows)
        distill = dataclasses.replace(distill, teachers=specs)
    if not distill.teachers:
        raise ValueError("Supply algo.config.distill.teachers explicitly; no checkpoint is selected automatically")
    assert_student_observations(config)
    groups = {key: value for key, value in config.observation.groups.items() if not key.startswith("expert_")}
    teachers, obs_keys = [], []
    covered = set()
    for index, spec in enumerate(distill.teachers):
        checkpoint = resolve_published_path(spec.checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        validate_g1_asset_metadata(loaded.get("robot_asset"), context=str(checkpoint))
        from somaforge_core.collision_profiles import require_profile
        require_profile(loaded.get("robot_asset"),getattr(config.robot.asset, "collision_profile", "legacy_v1"),
                        context=f"routed teacher {checkpoint}")
        expert = _experiment_config_from_serialized(loaded["experiment_config"])
        if dataclasses.asdict(expert.robot.control) != dataclasses.asdict(config.robot.control):
            raise ValueError(f"{checkpoint}: teacher/student action control differs")
        manifest = load_motion_terrain_manifest(spec.manifest)
        for row in manifest["motion_files"]:
            identity = str(Path(row["motion_file"]).resolve())
            if identity in covered:
                raise ValueError(f"A motion belongs to multiple experts: {identity}")
            covered.add(identity)
        keys = []
        for key in expert.algo.config.module_dict.actor.input_dim:
            target = f"expert_{index}_{key}"
            source = expert.observation.groups[key]
            terms = {name: dataclasses.replace(term, params={**term.params, "add_noise": False})
                     if "add_noise" in term.params else term for name, term in source.terms.items()}
            if config.algo._target_ == SEGMENT_DISTILL_ALGO:
                terms = {name: dataclasses.replace(term,
                    func='holosoma.php_contact.segments:teacher_observation',
                    params={'function': term.func, 'parameters': term.params}) for name, term in terms.items()}
            groups[target] = dataclasses.replace(source, enable_noise=False, terms=terms)
            keys.append(target)
        teachers.append(dataclasses.replace(spec, checkpoint=str(checkpoint), observation_keys=tuple(keys)))
        obs_keys.extend(keys)
    own_manifest = config.command.setup_terms["motion_command"].params["motion_config"].motion_manifest
    wanted = load_motion_terrain_manifest(own_manifest)
    identities = {str(Path(row["motion_file"]).resolve()) for row in wanted["motion_files"]}
    if not identities.issubset(covered):
        raise ValueError(f"Student manifest has {len(identities-covered)} motions with no teacher")
    groups["expert_route"] = ObsGroupCfg(terms={"route": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt_experiment:teacher_route",
        params={"manifests": tuple(spec.manifest for spec in teachers)},
    )})
    return dataclasses.replace(config,
        observation=dataclasses.replace(config.observation, groups=groups),
        algo=dataclasses.replace(config.algo, config=dataclasses.replace(config.algo.config,
            distill=dataclasses.replace(distill, teachers=tuple(teachers),
                teacher_obs_keys=tuple([*obs_keys, "expert_route"])))),
    )


def prepare_wbt_experiment(config):
    """Validate required native labels before creating a GPU environment."""
    config = prepare_routed_distillation(config)
    functions = {term.func for group in config.observation.groups.values() for term in group.terms.values()} if config.observation else set()
    if config.reward:
        functions.update(term.func for term in config.reward.terms.values())
    requires_reference = any(func in {
        "holosoma.managers.observation.terms.wbt_experiment:sparse_reference",
        "holosoma.managers.observation.terms.wbt_experiment:reference_force",
        "holosoma.managers.reward.terms.wbt_experiment:contact_match",
        "holosoma.managers.reward.terms.wbt_experiment:force_tracking",
    } for func in functions)
    if requires_reference:
        from holosoma.utils.motion_terrain_manifest import load_motion_terrain_manifest
        filename = config.command.setup_terms["motion_command"].params["motion_config"].motion_manifest
        for row in load_motion_terrain_manifest(filename)["motion_files"]:
            load_contact_reference(row["motion_file"], collision_profile=getattr(config.robot.asset, "collision_profile", "legacy_v1"))
    return config
