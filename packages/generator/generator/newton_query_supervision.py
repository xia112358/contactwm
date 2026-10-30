"""Loss-only scene identity and local-to-native frame, not predictor features."""
import json
import numpy as np
import torch
from somaforge_core.g1_kinematics import _quaternion_matrix_wxyz
from somaforge_core.robot_assets import decode_robot_asset_json
from somaforge_core.contact_face_selection import upward_face_mask
from somaforge_core.published_data import load_published_json, resolve_published_path


def validate_native_box_scene(center, rotation, half, ground, native_scene):
    """Compare cached geometry with recorded actual Newton terrain faces.

    The numerical tolerance checks a rigid coordinate transform, not contact
    activation. No distance threshold or contact labels are introduced here.
    """
    catalog = native_scene['surface_catalog']
    faces = [f for f, keep in zip(catalog, upward_face_mask(
        [f['normal_w'] for f in catalog]), strict=True) if keep]
    if len(faces) != 2:
        raise ValueError('Predictor scene requires exactly two actual upward Newton faces')
    faces.sort(key=lambda f: f['plane_offset'])
    floor_face, top_face = faces
    if top_face['plane_offset'] <= floor_face['plane_offset']:
        raise ValueError('Actual Newton top must be above ground')
    triangles = [t for group in native_scene['shape_surface'].values()
        for f in group if f['surface'] == top_face['surface']
        for t in f['triangles_w']]
    if not triangles:
        raise ValueError('Missing actual Newton top triangles; geometry cannot be verified')
    actual = np.unique(np.asarray(triangles, dtype=np.float64).reshape(-1, 3), axis=0)
    center, rotation, half, ground = [v.detach().double().cpu().numpy()
        for v in (center, rotation, half, ground)]
    signs = np.array([[-1., -1., 1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., 1.]])
    predicted = center[:, None] + np.einsum('bij,bpj->bpi', rotation, half[:, None]*signs)
    distance = np.linalg.norm(predicted[:, :, None]-actual[None, None], axis=-1)
    error = max(float(np.abs(ground-floor_face['plane_offset']).max()),
        float(np.abs(predicted[..., 2]-top_face['plane_offset']).max()),
        float(distance.min(2).max()), float(distance.min(1).max()))
    if (not np.isfinite(error) or error > 1e-4
            or not np.allclose(rotation[:, :, 2], [0., 0., 1.], atol=1e-6, rtol=0)):
        raise ValueError(f'Cached box differs from actual Newton terrain; rebuild predictor cache (error={error})')
    return error


def prepare_query_metadata(manifest, inputs, target, samples, *, scene=None):
    manifest = resolve_published_path(manifest)
    entries = load_published_json(manifest)['motion_files']
    def resolve(value):
        return resolve_published_path(value, manifest.parent)
    grouped = {}
    for i, sample in enumerate(samples):grouped.setdefault(sample.motion_id, []).append(i)
    device = inputs['current_q'].device
    origins = inputs['current_q'].new_empty((len(samples),3))
    bases = inputs['current_q'].new_empty((len(samples),3,3))
    fingerprints = torch.empty((len(samples),32),dtype=torch.uint8,device=device)
    max_error = 0.
    scene_error = 0.
    for motion, indices in grouped.items():
        entry = entries[motion]
        with np.load(resolve(entry['newton_contact_file']),allow_pickle=False) as z:
            semantics = json.loads(z['contact_semantics_json'].item())
            fp = semantics['scene']['model_fingerprint']
        with np.load(resolve(entry['motion_file']),allow_pickle=False) as z:
            decode_robot_asset_json(z['robot_asset_json'],context='Newton query training metadata')
            world = torch.as_tensor(z['joint_pos'],device=device,dtype=inputs['current_q'].dtype)
        if world.ndim != 2 or world.shape[1] != 36:
            raise ValueError('Native query metadata requires canonical qpos motion')
        ids = torch.tensor(indices,device=device)
        current = world[[samples[i].current_frame for i in indices]]
        end = world[[samples[i].target_frame for i in indices]]
        local = inputs['current_q'][ids]
        basis = _quaternion_matrix_wxyz(current[:,3:7]) @ _quaternion_matrix_wxyz(local[:,3:7]).transpose(-1,-2)
        origin = current[:,:3]-torch.einsum('bij,bj->bi',basis,local[:,:3])
        tq = target['q'][ids]
        # A scene transform cannot repair different joint data or a mismatched
        # cache. Check both endpoints, not just a constructed start alignment.
        error = max(float((current[:,7:]-local[:,7:]).abs().max()),
            float((end[:,7:]-tq[:,7:]).abs().max()),
            float((torch.einsum('bij,bj->bi',basis,tq[:,:3])+origin-end[:,:3]).abs().max()),
            float((basis@_quaternion_matrix_wxyz(tq[:,3:7])-_quaternion_matrix_wxyz(end[:,3:7])).abs().max()))
        if error > 1e-4:raise ValueError(f'Local/native q cache mismatch for {entry["motion_id"]}: {error}')
        if not torch.allclose(basis[:,2],basis.new_tensor([0.,0.,1.]).expand(len(ids),3),atol=1e-5,rtol=0):
            raise ValueError('Training frame is not ground-preserving yaw')
        origins[ids] = origin; bases[ids] = basis
        if scene is not None:
            scene_error = max(scene_error, validate_native_box_scene(
                origin+torch.einsum('bij,bj->bi', basis, scene['box_center'][ids]),
                basis@scene['box_rotation'][ids], scene['box_half_extents'][ids],
                origin[:, 2]+scene['ground_height'][ids], semantics['scene']))
        fingerprints[ids] = torch.tensor(list(bytes.fromhex(fp)),dtype=torch.uint8,device=device)
        max_error = max(max_error,error)
    target.update(newton_world_origin=origins,newton_world_basis=bases,newton_model_fingerprint=fingerprints)
    return dict(query_metadata='loss_only_native_fingerprint_and_frame_v1',native_frame_max_error=max_error,
        native_scene_verified=scene is not None, native_scene_max_error_m=scene_error if scene is not None else None)
