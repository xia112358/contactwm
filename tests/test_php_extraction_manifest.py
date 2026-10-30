"""PHP source metadata must not change the exact policy collection task identity."""
import copy

import pytest

from somaforge_core.trajectory_refinement import reference_signature, sha256_file


def manifests(tmp_path):
    motion = tmp_path/'motion.npz'
    terrain = tmp_path/'terrain.obj'
    motion.write_bytes(b'canonical motion')
    terrain.write_text('v 0 0 0\n')
    source = dict(source_kind='php_release_direct_newton_fk', php_component='motions-low-climb-76',
        motion_files=[dict(motion_file=str(motion), motion_sha256=sha256_file(motion), terrain_id=7)],
        terrains=[dict(terrain_file=str(terrain), terrain_sha256=sha256_file(terrain), terrain_id=7)])
    normalized = copy.deepcopy(source)
    normalized.pop('source_kind'); normalized.pop('php_component')
    normalized['purpose'] = 'kinematic references'
    normalized['motion_files'][0]['terrain_id'] = 0
    normalized['terrains'][0]['terrain_id'] = 0
    return source, normalized


def test_php_description_and_normalized_terrain_ids_preserve_task_identity(tmp_path):
    source, normalized = manifests(tmp_path)
    assert reference_signature(source, tmp_path) == reference_signature(normalized, tmp_path)


def test_changed_sampling_weight_or_mesh_binding_is_not_equivalent(tmp_path):
    source, normalized = manifests(tmp_path)
    normalized['motion_files'][0]['weight'] = 2.
    assert reference_signature(source, tmp_path) != reference_signature(normalized, tmp_path)
    normalized['motion_files'][0]['weight'] = 1.
    other = tmp_path/'other.obj'; other.write_text('v 1 0 0\n')
    normalized['terrains'][0].update(terrain_file=str(other), terrain_sha256=sha256_file(other))
    assert reference_signature(source, tmp_path) != reference_signature(normalized, tmp_path)


def test_modified_file_cannot_be_certified_using_a_stale_declared_hash(tmp_path):
    source, _ = manifests(tmp_path)
    (tmp_path/'motion.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        reference_signature(source, tmp_path)
