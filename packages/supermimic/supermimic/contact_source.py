"""Enforce original PHP source identity before creating new contact slices."""


def require_original_contact_source(manifest):
    component = str(manifest.get('source_component', ''))
    tasks = manifest.get('tasks', [])
    composed = (component.startswith(('php_all_five_groups', 'motions-'))
                or any(str(t.get('source_motion_id', '')).startswith('motions-') for t in tasks))
    if composed or manifest.get('composed_trajectories') is True:
        raise ValueError('Composed PHP trajectories are historical data and must not be re-sliced. '
                         'Use original PHP database clips from data/php_originals_v1/catalogue.json, '
                         'then collect matching native contacts before segmentation.')
    if manifest.get('source_unit') != 'original_database_clip':
        raise ValueError('New contact slicing requires source_unit=original_database_clip '
                         'and original source-frame provenance; do not relabel an old composed dataset.')
