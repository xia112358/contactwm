from __future__ import annotations

from pathlib import Path

from motion_edit.contact.graph import ContactGraph
from motion_edit.contact.io import (
    read_contact_anchors,
    read_contact_events,
    read_contact_patches,
    read_contact_transitions,
    write_contact_jsonl,
)
from motion_edit.contact.patches import patches_from_anchors


def write_contact_layer(root: str | Path, graph: ContactGraph) -> Path:
    layer_root = Path(root).expanduser()
    write_contact_jsonl(layer_root / "events" / f"{graph.motion_id}.jsonl", graph.events)
    write_contact_jsonl(layer_root / "anchors" / f"{graph.motion_id}.jsonl", graph.anchors)
    write_contact_jsonl(layer_root / "patches" / f"{graph.motion_id}.jsonl", graph.patches)
    write_contact_jsonl(layer_root / "transitions" / f"{graph.motion_id}.jsonl", graph.transitions)
    return layer_root


def read_contact_graph(root: str | Path, motion_id: str) -> ContactGraph:
    layer_root = Path(root).expanduser()
    anchors = read_contact_anchors(layer_root / "anchors" / f"{motion_id}.jsonl")
    patch_path = layer_root / "patches" / f"{motion_id}.jsonl"
    patches = read_contact_patches(patch_path) if patch_path.exists() else patches_from_anchors(anchors)
    return ContactGraph(
        motion_id=motion_id,
        events=read_contact_events(layer_root / "events" / f"{motion_id}.jsonl"),
        anchors=anchors,
        patches=patches,
        transitions=read_contact_transitions(layer_root / "transitions" / f"{motion_id}.jsonl"),
    )


def verified_source_patches(graph, *, source_motion_path, min_force_norm=0.0):
    """Use patches from read_verified_contact_graph, never rebind legacy raw data."""
    if min_force_norm != 0:
        raise ValueError('Verified activation patches cannot be filtered by contact force')
    anchors={a.anchor_id:a for a in graph.anchors}
    if len(graph.patches)!=len(anchors):
        raise ValueError('Verified source requires exactly one patch per anchor')
    samples=0
    for patch in graph.patches:
        anchor=anchors.get(patch.anchor_id)
        if anchor is None or not anchor.metadata.get('newton_shape_label'):
            raise ValueError('Source patch lacks verified Newton shape/face ownership')
        if patch.metadata.get('source_target_contract')!='newton_robot_geometry_point_trajectory':
            raise ValueError('Source patch is not a verified robot geometry-point trajectory')
        if patch.source_target_frames!=list(range(anchor.start_frame,anchor.end_frame)):
            raise ValueError('Verified source patch has incomplete time coverage')
        if patch.source_points_local_by_frame is None or patch.source_target_points_w is None:
            raise ValueError('Verified source patch has no frame-resolved rigid binding')
        samples+=sum(len(points) for points in patch.source_target_points_w)
    return list(graph.patches),dict(contract='verified_source_patches_no_raw_rebinding',
        source_motion_path=str(Path(source_motion_path).resolve()),patch_count=len(graph.patches),
        contact_point_samples=samples,force_filter_used=False,legacy_raw_contacts_used=False)


def read_verified_contact_graph(root, motion_id, *, motion_path, labels_path=None):
    """Generation must not consume unverified legacy anchor timelines.

    The ordinary reader remains available for inspection/visualization. Task
    labels are derived by the same loader as training, never force thresholds.
    """
    import json
    import numpy as np
    from somaforge_core.newton_contact_data import load_contact_labels
    from somaforge_core.newton_contacts import PARTS
    from motion_edit.contact.io import read_contact_surfaces
    path = Path(labels_path) if labels_path else Path(root)/'contact_labels'/f'{motion_id}.npz'
    labels = load_contact_labels(motion_path, path)
    graph = read_contact_graph(root, motion_id)
    patch_by_anchor = {p.anchor_id:p for p in graph.patches}
    catalog = {s.surface_id:s for s in read_contact_surfaces(Path(root)/'surfaces'/f'{motion_id}.jsonl')}
    with np.load(path, allow_pickle=False) as z:
        semantics = json.loads(z['contact_semantics_json'].item())
        binding = semantics['scene']
        pairs = labels['contact_pairs']
    faces = binding.get('surface_catalog', [])
    if not faces:
        raise ValueError('Augmentation contact labels need explicit Newton face mapping')
    for anchor in graph.anchors:
        body = anchor.body
        side = body.split('_')[0]
        kind = 'foot' if any(x in body for x in ('heel', 'toe', 'foot', 'ankle')) else 'hand' if 'hand' in body else 'knee' if 'knee' in body else None
        if kind is None or f'{side}_{kind}' not in PARTS:
            raise ValueError(f'Unmapped augmentation contact part: {body}')
        part = PARTS.index(f'{side}_{kind}')
        if anchor.surface_id not in catalog:
            raise ValueError(f'Missing augmentation surface: {anchor.surface_id}')
        surface = catalog[anchor.surface_id]
        if surface.surface_type != 'plane' and not surface.metadata.get('polygon_world'):
            raise ValueError('Augmentation requires real face polygon, not only rectangular bounds')
        normal = np.asarray(surface.normal)
        ids = [f['surface'] for f in faces if np.allclose(f['normal_w'], normal, atol=1e-6, rtol=0)
               and abs(f['plane_offset']-np.dot(normal,surface.origin)) < 1e-6]
        if len(ids) != 1:
            raise ValueError(f'Ambiguous/missing Newton face for {anchor.surface_id}')
        a,b = int(anchor.start_frame),int(anchor.end_frame)
        mask = labels['contact_part_mask']; actual_surface = labels['contact_surface']
        if not 0 <= a < b <= len(mask):
            raise ValueError('Augmentation anchor interval is invalid')
        shape_label = anchor.metadata.get('newton_shape_label')
        if shape_label is not None:
            from somaforge_core.contact_labels import DEFAULT_POLICY, ContactLabelPolicy
            if DEFAULT_POLICY != ContactLabelPolicy():
                raise ValueError('Shape-level aggregation/filtering must be migrated together with part labels')
            if anchor.metadata.get('contact_label_contract') != labels['contact_label_contract']:
                raise ValueError('Augmentation anchor contact policy mismatch')
            shapes = semantics['provenance']['shape_labels']
            if pairs is None or shape_label not in shapes:
                raise ValueError('Augmentation shape identity not present in verified observation')
            shape_id = shapes.index(shape_label)
            if not all(any(p['robot_shape']==shape_id and p['surface']==ids[0] and p['part']==part
                           for p in pairs[f]) for f in range(a,b)):
                raise ValueError(f'Anchor {anchor.anchor_id} has unobserved shape/face contact frames')
            patch = patch_by_anchor.get(anchor.anchor_id)
            if patch is None or patch.source_target_frames != list(range(a,b)):
                raise ValueError('Verified augmentation patch must retain every source frame')
            if patch.newton_shape_labels is None or any(s != shape_label for s in patch.newton_shape_labels):
                raise ValueError('Augmentation patch shape identity differs from its anchor')
            for f, points in zip(patch.source_target_frames, patch.source_target_points_w or []):
                actual = sorted(tuple(p['position_w']) for p in pairs[f]
                                if p['robot_shape']==shape_id and p['surface']==ids[0] and p['part']==part)
                if sorted(tuple(p) for p in points) != actual:
                    raise ValueError('Augmentation source patch points differ from actual geometry observations')
            if patch.source_target_points_w is None or len(patch.source_target_points_w) != b-a:
                raise ValueError('Missing augmentation source point trajectory')
            continue
        if not mask[a:b,part].all() or not (actual_surface[a:b,part] == ids[0]).all():
            raise ValueError(f'Anchor {anchor.anchor_id} disagrees with shared task contact labels; rebuild source layer')
    return graph
