"""Task eligibility over verified Newton pairs; raw solver facts stay untouched."""
import numpy as np

FACE_POLICY = 'primary_horizontal_upward_faces_v1'


def upward_face_mask(normals):
    """Shared face eligibility, not a contact activation test."""
    normals = np.asarray(normals, float)
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all():
        raise ValueError('Invalid face normals')
    return np.all(np.isclose(normals, [0, 0, 1], atol=1e-6, rtol=0), axis=-1)


def effective_contact_contract(fps):
    from .contact_labels import label_contract
    return dict(schema='somaforge_effective_contact_labels_v2',
                face_selection=FACE_POLICY, debounce=label_contract(fps))


def ground_top_catalog(records):
    """Map observed single-box geometry to the same ground=0/top=1 roles.

    Eligibility comes from the shared upward-normal rule, never from treating
    every non-ground surface as a top. Ambiguous catalogs fail closed.
    """
    if not records:raise ValueError('Missing geometry surface catalog')
    ids=[str(r['surface_id']) for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate geometry surface IDs')
    keep=upward_face_mask([r['normal'] for r in records])
    upward=[r for r,k in zip(records,keep) if k]
    ground=[r for r in upward if r['surface_id']=='terrain_ground_z0']
    top=[r for r in upward if r['surface_id']!='terrain_ground_z0']
    if len(ground)!=1 or len(top)!=1:raise ValueError('Expected exactly one upward ground and one upward top')
    for r in (ground[0],top[0]):
        origin=np.asarray(r['origin'],float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Invalid geometry surface origin')
    if top[0]['origin'][2]<=ground[0]['origin'][2]:raise ValueError('Top must be above ground')
    return {0:ground[0],1:top[0]}


def select_contact_pairs(pairs, catalog, *, part_count=6):
    """Keep ground/top primary faces, never promote an ambiguous side to top.

    Normal tolerance classifies geometry, not whether a contact is active.
    Inputs must come from the verified Newton query/loader.
    """
    normals = {int(s['surface']): np.asarray(s['normal_w'], float) for s in catalog}
    if not normals or len(normals) != len(catalog):
        raise ValueError('Missing/duplicate actual surface catalog')
    if any(n.shape != (3,) or not np.isfinite(n).all() or
           not np.isclose(np.linalg.norm(n), 1., atol=1e-6) for n in normals.values()):
        raise ValueError('Invalid actual surface normals')
    upward = {s for s, keep in zip(normals, upward_face_mask(list(normals.values()))) if keep}
    if not isinstance(part_count, int) or part_count < 1:
        raise ValueError('Invalid contact part count')
    mask = np.zeros((len(pairs), part_count), bool)
    surface = np.full(mask.shape, -1, np.int64)
    position = np.zeros((*mask.shape, 3), np.float32)
    kept, excluded = [], []
    for f, row in enumerate(pairs):
        good, bad = [], []
        score = np.full(part_count, np.inf)
        for p in row:
            sid, part = int(p['surface']), int(p['part'])
            point = np.asarray(p['position_w'], float)
            if sid not in normals or not 0 <= part < part_count or not p['allocated']:
                raise ValueError('Unknown surface/part or unallocated Newton contact')
            depth = float(p['dist']) - float(p['includemargin'])
            if point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(depth) or depth >= 0:
                raise ValueError('Invalid activated Newton pair')
            if sid not in upward:
                bad.append(p)
                continue
            good.append(p)
            mask[f, part] = True
            # Same representative rule as Newton reduction, AFTER eligibility.
            if depth < score[part]:
                score[part] = depth
                surface[f, part] = sid
                position[f, part] = point
        kept.append(good)
        excluded.append(bad)
    return dict(contact_part_mask=mask, contact_surface=surface,
                contact_position_w=position, contact_pairs=kept,
                abnormal_contact_pairs=excluded)


def process_face_contacts(pairs, catalog, *, fps):
    from .contact_labels import process_contact_sequence
    selected = select_contact_pairs(pairs, catalog)
    result = process_contact_sequence(selected['contact_part_mask'], selected['contact_surface'],
                                      selected['contact_position_w'], fps=fps)
    result.update(contact_pairs=selected['contact_pairs'],
                  abnormal_contact_pairs=selected['abnormal_contact_pairs'])
    # Separate from the generic debounce contract: unfiltered callers cannot
    # accidentally certify themselves as having applied the face selection.
    result['contact_label_contract'] = effective_contact_contract(fps)
    return result
