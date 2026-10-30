"""Complete realized collision solids for optimization, never contact truth.

Newton's triangle contacts are immutable evidence of constraint activation.
Their local penetration normal can reverse after a body crosses a triangle.
This separate geometry reader queries the SAME enabled shapes as complete
solids. Disconnected static components remain separate; unsupported geometry
fails explicitly rather than being replaced by an enclosing hull.
"""
from __future__ import annotations

import numpy as np

SOLID_GEOMETRY_SCHEMA = 'realized_newton_solid_geometry_v1'
SOLID_WITNESS_SCHEMA = 'realized_solid_distance_witnesses_v1'


def export_solid_geometry(model, provenance):
    """Read the initialized policy model without changing any physics field."""
    import newton
    from newton._src.geometry.broad_phase_common import test_world_and_group_pair
    from somaforge_core.robot_assets import validate_g1_asset_metadata

    validate_g1_asset_metadata(provenance['robot_asset'], context='solid distance export')
    if not provenance.get('model_fingerprint'):
        raise ValueError('Solid geometry requires the actual initialized model fingerprint')
    bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    worlds, groups = model.shape_world.numpy(), model.shape_collision_group.numpy()
    kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
    transforms, body_world = model.shape_transform.numpy(), model.body_world.numpy()
    kind_names = {int(getattr(newton.GeoType, name)): name.lower() for name in
                  ('MESH', 'CONVEX_MESH', 'SPHERE', 'CYLINDER', 'CAPSULE', 'BOX', 'ELLIPSOID')}
    records = []
    for shape, body in enumerate(bodies):
        if not int(flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        # All query worlds contain the same robot; export its world-zero
        # template and its actual global/static terrain only once.
        if body >= 0 and int(body_world[body]) != 0:
            continue
        if int(worlds[shape]) not in (-1, 0):
            continue
        if int(kinds[shape]) not in kind_names:
            raise ValueError(f'Unsupported realized solid shape {shape}, type {kinds[shape]}')
        record = dict(shape=int(shape), body=None if body < 0 else str(model.body_label[body]).rsplit('/', 1)[-1],
                      kind=kind_names[int(kinds[shape])], scale=scales[shape].tolist(),
                      transform=transforms[shape].tolist())
        source = model.shape_source[shape]
        if record['kind'] in ('mesh', 'convex_mesh'):
            if source is None or not hasattr(source, 'vertices'):
                raise ValueError(f'Missing realized solid mesh at shape {shape}')
            record.update(vertices=np.asarray(source.vertices).tolist(),
                          faces=np.asarray(source.indices).reshape(-1, 3).tolist())
        records.append(record)
    if not records or not any(r['body'] is None for r in records):
        raise ValueError('Solid distance requires actual robot and static terrain geometry')
    excluded = {tuple(sorted(map(int, pair))) for pair in model.shape_collision_filter_pairs}
    allowed = []
    for i, a in enumerate(records):
        for j in range(i+1, len(records)):
            b = records[j]
            if a['body'] == b['body']:
                continue
            x, y = a['shape'], b['shape']
            if tuple(sorted((x, y))) in excluded:
                continue
            if bool(test_world_and_group_pair(int(worlds[x]), int(worlds[y]), int(groups[x]), int(groups[y]))):
                allowed.append([i, j])
    return dict(schema=SOLID_GEOMETRY_SCHEMA, model_fingerprint=provenance['model_fingerprint'],
                robot_asset=provenance['robot_asset'], shapes=records, allowed_pairs=allowed,
                meaning='optimization distances only; raw Newton contacts and simulation configuration unchanged')


class SolidDistanceScene:
    """Float64 GJK/EPA on complete native shapes and enabled shape pairs."""

    def __init__(self, data, link_names, *, backend="python"):
        if backend not in ("python", "native_batch"):
            raise ValueError(f"Unknown solid distance backend: {backend}")
        self.backend = backend
        if backend == "native_batch":
            from somaforge_core.coal_batch_loader import load_coal_batch
            load_coal_batch()
        import coal
        import trimesh
        from scipy.spatial.transform import Rotation
        if data.get('schema') != SOLID_GEOMETRY_SCHEMA or not data.get('model_fingerprint'):
            raise ValueError('Missing realized solid geometry contract')
        from somaforge_core.robot_assets import validate_g1_asset_metadata
        validate_g1_asset_metadata(data['robot_asset'], context='solid distance scene')
        self.fingerprint = data['model_fingerprint']
        self.coal = coal
        self.link_names = tuple(link_names)
        self.records = []
        expanded = []
        for shape in data['shapes']:
            ids = []
            scale = np.asarray(shape['scale'], dtype=float)
            transform = np.asarray(shape['transform'], dtype=float)
            if scale.shape != (3,) or transform.shape != (7,) or not np.isfinite(np.r_[scale, transform]).all():
                raise ValueError('Invalid realized solid shape transform')
            name, kind = shape['body'], shape['kind']
            if name is not None and name not in self.link_names:
                raise ValueError(f'Realized solid body is absent from canonical FK: {name}')
            base = dict(shape=shape['shape'], link=-1 if name is None else self.link_names.index(name),
                        body=name, rotation=Rotation.from_quat(transform[3:]).as_matrix(), position=transform[:3])
            if kind in ('mesh', 'convex_mesh'):
                # Newton can export triangle-soup indices. Weld only exactly
                # coincident vertices; never round, fill holes or take a hull.
                vertices, inverse = np.unique(np.asarray(shape['vertices']), axis=0, return_inverse=True)
                mesh = trimesh.Trimesh(vertices=vertices*scale,
                                       faces=inverse[np.asarray(shape['faces'])], process=False)
                pieces = mesh.split(only_watertight=False, repair=False) if name is None else [mesh]
                if not len(pieces):
                    raise ValueError('Empty realized solid mesh')
                for component, piece in enumerate(pieces):
                    if not piece.is_watertight or not piece.is_winding_consistent or not piece.is_convex:
                        raise ValueError(f'Realized shape {shape["shape"]}/{component} is not a verified closed convex solid')
                    points, faces = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
                    for point in piece.vertices:
                        points.append(point)
                    for face in piece.faces:
                        faces.append(coal.Triangle(*map(int, face)))
                    ids.append(len(self.records))
                    self.records.append(dict(base, component=component, geometry=coal.Convex(points, faces),
                                             low=piece.bounds[0], high=piece.bounds[1]))
            else:
                if kind == 'sphere':
                    geometry, half = coal.Sphere(float(scale[0])), np.full(3, scale[0])
                elif kind == 'cylinder':
                    if scale[2] != 0:
                        raise ValueError('A native barrel cylinder requires its actual curved solid geometry')
                    geometry, half = coal.Cylinder(float(scale[0]), 2*float(scale[1])), np.array([scale[0], scale[0], scale[1]])
                elif kind == 'capsule':
                    geometry, half = coal.Capsule(float(scale[0]), 2*float(scale[1])), np.array([scale[0], scale[0], scale[0]+scale[1]])
                elif kind == 'box':
                    geometry, half = coal.Box(2*scale), scale
                elif kind == 'ellipsoid':
                    geometry, half = coal.Ellipsoid(scale), scale
                else:
                    raise ValueError(f'Unsupported realized solid geometry: {kind}')
                if (half <= 0).any():
                    raise ValueError('Nonpositive realized solid dimensions')
                ids.append(len(self.records))
                self.records.append(dict(base, component=0, geometry=geometry, low=-half, high=half))
            expanded.append(ids)
        pairs = [(a, b) for i, j in data['allowed_pairs'] for a in expanded[i] for b in expanded[j]]
        self.allowed = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.links = np.asarray([r['link'] for r in self.records], dtype=np.int64)
        self.local_rotation = np.stack([r['rotation'] for r in self.records])
        self.local_position = np.stack([r['position'] for r in self.records])
        lows, highs = np.stack([r['low'] for r in self.records]), np.stack([r['high'] for r in self.records])
        self.local_center, self.local_half = (lows+highs)/2, (highs-lows)/2
        self.request = coal.DistanceRequest()
        self.request.enable_signed_distance = True
        # Keep Coal's supported default precision. Over-refining EPA on curved
        # native shapes can invalidate its polytope, returning either DBL_MAX
        # penetration or a finite but incorrect depth and recovery normal.

    @staticmethod
    def _invalid_result(sample, first, second, positions, rotations, **details):
        import json
        evidence = dict(sample=sample, shapes=[first['shape'], second['shape']],
                        components=[first['component'], second['component']],
                        positions=positions.tolist(), rotations=rotations.tolist(), **details)
        return ValueError('Inconsistent solid distance/witnesses: '+json.dumps(evidence))

    def query(self, position, rotation):
        """Return every penetrating enabled pair, including full containment.

        AABBs reject provably disjoint pairs and bound result validity. They
        never supply a signed distance, normal or contact label. No native
        triangle candidate list is required, so full containment is still found.
        """
        position, rotation = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
        batch = len(position)
        if position.shape != (batch, len(self.link_names), 3) or rotation.shape != (batch, len(self.link_names), 3, 3):
            raise ValueError('Solid query body pose/mapping mismatch')
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError('Nonfinite solid query pose')
        if (not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1.e-8, rtol=0)
                or not (np.linalg.det(rotation) > 0).all()):
            raise ValueError('Solid query requires proper rigid rotation matrices')
        moving = self.links >= 0
        p = np.broadcast_to(self.local_position, (batch, *self.local_position.shape)).copy()
        r = np.broadcast_to(self.local_rotation, (batch, *self.local_rotation.shape)).copy()
        link = self.links[moving]
        p[:, moving] = position[:, link]+np.einsum('bsij,sj->bsi', rotation[:, link], self.local_position[moving])
        r[:, moving] = rotation[:, link] @ self.local_rotation[moving]
        center = p+np.einsum('bsij,sj->bsi', r, self.local_center)
        half = np.einsum('bsij,sj->bsi', np.abs(r), self.local_half)
        low, high = center-half, center+half
        a, b = self.allowed.T
        candidates = ((low[:, a] <= high[:, b]) & (low[:, b] <= high[:, a])).all(-1)
        if self.backend == "native_batch":
            from somaforge_core.solid_distance_batch import run_candidates
            return run_candidates(self, p, r, low, high, candidates)
        fields = {key: [] for key in ('sample', 'shape0', 'shape1', 'component0', 'component1',
                                     'body_link0', 'body_link1', 'full_kind', 'dist',
                                     'point0_w', 'point1_w', 'normal_w')}
        tolerance = max(self.request.gjk_tolerance, self.request.epa_tolerance)
        for sample in range(batch):
            transforms = {}
            selected = np.flatnonzero(candidates[sample])
            first_ids, second_ids = self.allowed[selected].T
            # Same validity bounds for every candidate, evaluated as arrays.
            # Keep the Coal call/order and all per-result checks unchanged.
            world_lows = np.minimum(low[sample, first_ids], low[sample, second_ids])
            world_highs = np.maximum(high[sample, first_ids], high[sample, second_ids])
            distance_bounds = np.linalg.norm(world_highs-world_lows, axis=-1)
            exit_bounds = np.minimum(high[sample, first_ids]-low[sample, second_ids],
                                     high[sample, second_ids]-low[sample, first_ids]).min(axis=-1)
            for offset, pair in enumerate(selected):
                i, j = self.allowed[pair]
                first, second = self.records[i], self.records[j]
                for index in (i, j):
                    if index not in transforms:
                        transforms[index] = self.coal.Transform3s(r[sample, index], p[sample, index])
                result = self.coal.DistanceResult()
                gap = float(self.coal.distance(first['geometry'], transforms[i], second['geometry'], transforms[j], self.request, result))
                # Translating along any world axis to separate the AABBs also
                # separates the solids. Their minimum exit distance therefore
                # cannot exceed the cheapest such translation, including when
                # one solid completely contains the other. This is a validity
                # bound, never a distance substitute or a contact threshold.
                exit_bound = float(exit_bounds[offset])
                world_low, world_high = world_lows[offset], world_highs[offset]
                distance_bound = float(distance_bounds[offset])
                if (not np.isfinite(gap) or gap > distance_bound+tolerance
                        or (gap < 0 and -gap > exit_bound+tolerance)):
                    raise self._invalid_result(sample, first, second, p[sample, [i, j]],
                        r[sample, [i, j]], reason='invalid_distance_or_geometric_exit_bound',
                        gap=gap, exit_bound=exit_bound, distance_bound=distance_bound,
                        tolerance=tolerance)
                if gap >= 0:
                    continue
                normal = np.asarray(result.normal).copy()
                p0, p1 = np.asarray(result.getNearestPoint1()).copy(), np.asarray(result.getNearestPoint2()).copy()
                # Reject unusable coordinates before subtracting or multiplying
                # them. A finite sentinel must not overflow witness validation.
                world_low, world_high = world_low-tolerance, world_high+tolerance
                if (not np.isfinite(np.r_[normal, p0, p1]).all()
                        or not ((np.stack((p0, p1)) >= world_low).all()
                                and (np.stack((p0, p1)) <= world_high).all())
                        or (np.abs(normal) > 1.+1.e-8).any()
                        or not np.isclose(np.linalg.norm(normal), 1., atol=1.e-8, rtol=0)
                        or not np.isclose(normal @ (p1-p0), gap, atol=1.e-8, rtol=1.e-8)):
                    raise self._invalid_result(sample, first, second, p[sample, [i, j]],
                        r[sample, [i, j]], reason='invalid_witness_coordinates_or_normal', gap=gap,
                        normal=normal.tolist(), point0=p0.tolist(), point1=p1.tolist())
                row = dict(sample=sample, shape0=first['shape'], shape1=second['shape'],
                           component0=first['component'], component1=second['component'],
                           body_link0=first['link'], body_link1=second['link'],
                           full_kind=0 if min(first['link'], second['link']) < 0 else 1,
                           dist=gap, point0_w=p0, point1_w=p1, normal_w=normal)
                for key, value in row.items():
                    fields[key].append(value)
        integers = {'sample', 'shape0', 'shape1', 'component0', 'component1', 'body_link0', 'body_link1', 'full_kind'}
        return {key: np.asarray(values, dtype=np.int64 if key in integers else float).reshape(-1, 3) if key in
                ('point0_w', 'point1_w', 'normal_w') else np.asarray(values, dtype=np.int64 if key in integers else float)
                for key, values in fields.items()}
