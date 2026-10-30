"""Explicit v1/v2 architecture selection for scratch training and checkpoints."""
from somaforge_core.robot_assets import decode_robot_asset_json
from generator.structured_position_predictor import StructuredPositionPredictor, STRUCTURED_SCHEMA

LEGACY_SCHEMA = 'full1000_position_predictor_v1'
POSITION_SCHEMAS = (LEGACY_SCHEMA, STRUCTURED_SCHEMA)
ARCHITECTURE_FIELDS = ('width', 'layers', 'location_width', 'body_geometry', 'part_geometry',
                       'region_plan', 'unified_contact', 'event_roles')


def build_position_predictor(config, *, schema=STRUCTURED_SCHEMA):
    """Legacy construction must be requested explicitly; never infer from tensor shapes."""
    if schema not in POSITION_SCHEMAS:
        raise ValueError(f'Unsupported position architecture: {schema}')
    if schema == STRUCTURED_SCHEMA:
        if config.get('architecture') != STRUCTURED_SCHEMA:
            raise ValueError('Structured training/checkpoint requires explicit v2 architecture')
        missing = set(ARCHITECTURE_FIELDS) - config.keys()
        if missing:
            raise ValueError(f'Structured architecture configuration missing: {sorted(missing)}')
        cls = StructuredPositionPredictor
    else:
        if config.get('architecture', LEGACY_SCHEMA) != LEGACY_SCHEMA:
            raise ValueError('Legacy schema/config architecture mismatch')
        from generator.full1000_position_predictor import Full1000PositionPredictor
        cls = Full1000PositionPredictor
    defaults = dict(width=192, layers=3, location_width=32, body_geometry=False,
        part_geometry=False, region_plan=False, unified_contact=False, event_roles=False)
    kwargs = {key: config.get(key, value) for key, value in defaults.items()}
    kwargs.update(execution_plan_gradients=config.get('execution_plan_gradients', False),
                  execution_observation_gradients=config.get('execution_observation_gradients', False),
                  role_validity=config.get('role_validity', False),
                  world_surface_scan=config.get('world_surface_scan', False),
                  depth_observation=config.get('depth_observation', False))
    if config.get("limb_contact_modes", False):
        if schema != LEGACY_SCHEMA:
            raise ValueError("Limb contact modes currently require v1")
        kwargs["limb_contact_modes"] = True
    model = cls(**kwargs)
    from somaforge_core.world_mesh_scan import LEGACY_DEPTH_PROFILE, depth_camera_profile
    model.depth_scan_profile = config.get('depth_scan_profile', LEGACY_DEPTH_PROFILE)
    depth_camera_profile(model.depth_scan_profile)
    return model


def load_position_predictor(checkpoint, *, device='cpu'):
    decode_robot_asset_json(checkpoint['robot_asset_json'], context='versioned position checkpoint')
    model = build_position_predictor(checkpoint['config'], schema=checkpoint['schema']).to(device)
    if getattr(model, "limb_contact_modes", False):
        from generator.limb_contact import SCHEMA
        if checkpoint.get("architecture_contract", {}).get("limb_contact_modes") != SCHEMA:
            raise ValueError("Checkpoint lacks its four-limb contact contract")
    if model.role_validity:
        from generator.role_validity import ROLE_VALIDITY_SCHEMA
        if checkpoint.get('architecture_contract', {}).get('role_validity') != ROLE_VALIDITY_SCHEMA:
            raise ValueError('Checkpoint lacks its required role-validity contract')
    if model.world_surface_scan:
        from somaforge_core.world_mesh_scan import SCAN_SCHEMA
        if checkpoint.get('architecture_contract', {}).get('interaction_candidates') != SCAN_SCHEMA:
            raise ValueError('Checkpoint lacks its required world mesh scan contract')
        if checkpoint['architecture_contract'].get('interaction_coordinates') != model.architecture_contract()['interaction_coordinates']:
            raise ValueError('Checkpoint world mesh coordinate contract mismatch')
    if model.depth_observation:
        if checkpoint.get('architecture_contract', {}).get('depth_camera') != model.architecture_contract().get('depth_camera'):
            raise ValueError('Checkpoint depth camera calibration contract mismatch')
        if (checkpoint.get('architecture_contract', {}).get('environment_observation')
                != model.architecture_contract()['environment_observation']):
            raise ValueError('Checkpoint lacks its required 2D depth observation contract')
    if checkpoint['schema'] == STRUCTURED_SCHEMA:
        if checkpoint.get('architecture_contract') != model.architecture_contract():
            raise ValueError('Checkpoint architecture contract/order mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    return model
