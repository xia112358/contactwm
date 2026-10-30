"""Frozen native geometry for optimization, never a contact-label backend.

Support intervals of the complete realized convex hull (or sphere) are tested
against each finite terrain triangle. A separating-axis gap is a conservative
lower bound on Euclidean separation, not an exact Newton contact distance.
No bounding boxes, sampled material witnesses, or generated feedback are used.
Native full-trajectory validation is mandatory before accepting the artifact.
"""
import json
from pathlib import Path

import numpy as np

SCHEMA = 'native_geometry_separation_optimizer_v1'


def export_geometry(model, solver, provenance):
    """Called only inside the dedicated, initialized policy scene."""
    import newton
    from scipy.spatial.transform import Rotation
    from newton._src.geometry.broad_phase_common import test_world_and_group_pair
    from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART
    from somaforge_core.newton_contacts import PARTS
    from somaforge_core.contact_face_selection import upward_face_mask

    if solver.use_mujoco_cpu or solver.mjw_model.opt.run_collision_detection:
        raise ValueError('Requires the current Newton collision / MJWarp solver')
    if solver.newton_shape_to_mjc_geom is None:
        solver._create_inverse_shape_mapping()
    mapping = solver.newton_shape_to_mjc_geom.numpy()
    margin = solver.mjw_model.geom_margin.numpy()
    gap = solver.mjw_model.geom_gap.numpy()
    if margin.shape[0] != 1 or gap.shape != margin.shape:
        raise ValueError('Expected one realized native world')
    flags = model.shape_flags.numpy()
    worlds = model.shape_world.numpy()
    groups = model.shape_collision_group.numpy()
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    excluded = {tuple(sorted(map(int, p))) for p in model.shape_collision_filter_pairs}
    shapes, terrain = [], []
    for i, body in enumerate(bodies):
        if not flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if mapping[i] < 0:
            raise ValueError(f'Unmapped colliding native shape {i}')
        part = None
        if body >= 0:
            name = model.body_label[body].rsplit('/', 1)[-1]
            found = [p for p, label in enumerate(PARTS) if name in CONTACT_BODY_NAMES_BY_PART[label]]
            if not found:
                continue
            if len(found) != 1:
                raise ValueError('Ambiguous native part')
            part = found[0]
        pose = transforms[i]
        rot = Rotation.from_quat(pose[3:])
        source = model.shape_source[i]
        if part is None:
            if source is None or not hasattr(source, 'vertices'):
                raise ValueError('Static terrain must be actual triangle geometry')
            vertices = rot.apply(np.asarray(source.vertices) * scales[i]) + pose[:3]
            triangles = vertices[np.asarray(source.indices).reshape(-1, 3)]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=-1)
            valid = lengths > 1e-10
            normals[valid] /= lengths[valid, None]
            # Same primary upward-horizontal face class as the task selector.
            keep = valid & upward_face_mask(normals)
            terrain.extend(dict(shape=i, vertices=t.tolist()) for t in triangles[keep])
            continue
        radius = 0.
        if types[i] == int(newton.GeoType.SPHERE):
            vertices = np.asarray([pose[:3]])
            radius = float(scales[i, 0])
        elif types[i] == int(newton.GeoType.CONVEX_MESH):
            if source is None or not hasattr(source, 'vertices'):
                raise ValueError('Missing realized convex mesh vertices')
            vertices = rot.apply(np.asarray(source.vertices) * scales[i]) + pose[:3]
        else:
            raise ValueError(f'Unsupported native task shape {i}, type {types[i]}; no approximation')
        shapes.append(dict(shape=i, body=name, part=part, vertices=vertices.tolist(), radius=radius))
    if not shapes or not terrain or set(s['part'] for s in shapes) != set(range(6)):
        raise ValueError('Incomplete native geometry')
    thresholds, allowed = [], []
    for s in shapes:
        row, enabled = [], []
        for t in terrain:
            a, b = s['shape'], t['shape']
            ga, gb = int(mapping[a]), int(mapping[b])
            # Exact rule in installed Newton contact_params, using live fields.
            row.append(float(margin[0, ga]+margin[0, gb]-gap[0, ga]-gap[0, gb]))
            enabled.append(tuple(sorted((a, b))) not in excluded and bool(test_world_and_group_pair(
                int(worlds[a]), int(worlds[b]), int(groups[a]), int(groups[b]))))
        thresholds.append(row)
        allowed.append(enabled)
    return dict(schema=SCHEMA, model_fingerprint=provenance['model_fingerprint'],
        robot_asset=provenance['robot_asset'], shapes=shapes, triangles=terrain,
        includemargin=thresholds, allowed=allowed,
        meaning='conservative optimization separation only; Newton final audit required')


class GeometryRelease:
    def __init__(self, path, fingerprint, link_names):
        from somaforge_core.robot_assets import validate_g1_asset_metadata
        data = json.loads(Path(path).read_text())
        if data['schema'] != SCHEMA or data['model_fingerprint'] != fingerprint:
            raise ValueError('Native geometry schema/model mismatch')
        validate_g1_asset_metadata(data['robot_asset'], context='native geometry optimizer')
        shapes = data['shapes']
        self.indices = np.asarray([link_names.index(s['body']) for s in shapes])
        self.parts = np.asarray([s['part'] for s in shapes])
        count = max(len(s['vertices']) for s in shapes)
        self.vertices = np.stack([np.pad(np.asarray(s['vertices']), ((0, count-len(s['vertices'])), (0, 0)), mode='edge') for s in shapes])
        self.radii = np.asarray([s['radius'] for s in shapes])
        triangles = np.asarray([t['vertices'] for t in data['triangles']])
        edges = np.roll(triangles, -1, axis=1)-triangles
        side_axes = np.cross(edges, [0, 0, 1])
        side_axes /= np.linalg.norm(side_axes, axis=-1, keepdims=True)
        self.axes = np.concatenate((np.broadcast_to([0, 0, 1], (len(triangles), 1, 3)), side_axes), axis=1)
        projection = np.einsum('tvc,tac->tva', triangles, self.axes)
        self.low, self.high = projection.min(1), projection.max(1)
        self.margin = np.asarray(data['includemargin'])
        self.allowed = np.asarray(data['allowed'], bool)
        # Roundoff guard, not an alteration of the simulator's contact margin.
        self.guard = np.finfo(np.float32).eps * max(1., np.abs(triangles).max()) * 4

    def gaps(self, world_vertices):
        import torch
        if isinstance(world_vertices, torch.Tensor):
            array = lambda x: torch.as_tensor(x, dtype=world_vertices.dtype, device=world_vertices.device)
            projection = torch.einsum('svc,tac->stva', world_vertices, array(self.axes))
            low = projection.amin(2)-array(self.radii)[:, None, None]
            high = projection.amax(2)+array(self.radii)[:, None, None]
            separation = torch.maximum(low-array(self.high), array(self.low)-high).amax(-1)
            return separation-array(self.margin)-self.guard
        projection = np.einsum('svc,tac->stva', world_vertices, self.axes)
        low = projection.min(2)-self.radii[:, None, None]
        high = projection.max(2)+self.radii[:, None, None]
        separation = np.maximum(low-self.high, self.low-high).max(-1)
        return separation-self.margin-self.guard


class FrameGeometryConstraint:
    def __init__(self, geometry, source_mask, evaluate):
        self.enabled = (geometry.allowed & ~np.asarray(source_mask, bool)[geometry.parts, None]).ravel()
        self.evaluate = evaluate
        self.roundoff = geometry.guard * .5
        self.calls = 0
        self.last = None

    def _get(self, x):
        if self.last is None or not np.array_equal(x, self.last[0]):
            self.calls += 1
            value, jacobian = self.evaluate(x)
            self.last = (np.array(x, copy=True), np.asarray(value, dtype=np.float64).ravel()[self.enabled],
                         np.asarray(jacobian, dtype=np.float64).reshape((-1, len(x)))[self.enabled])
        return self.last[1:]

    def fun(self, x):
        return self._get(x)[0]

    def jac(self, x):
        return self._get(x)[1]

    def feasible(self, x):
        # Optimizer zero can be -1e-16. Consume at most half the explicit
        # numerical guard, so the actual margin still has positive clearance.
        return bool(np.all(self.fun(x) >= -self.roundoff))


class FastFrameGeometryConstraint:
    """Same support inequalities, with lazy support-point derivatives.

    All vertices participate in value evaluation. Once an active supporting
    feature is selected, differentiating that support point gives a valid
    piecewise derivative without differentiating every mesh vertex. This is
    an optimizer-internal derivative, not a frozen witness or feedback input.
    """
    def __init__(self, geometry, source_mask, cfg_size, forward_vertices, projection_jac):
        self.geometry = geometry
        self.cfg_size = cfg_size
        self.shapes = np.flatnonzero(~np.asarray(source_mask, bool)[geometry.parts]
                                    & geometry.allowed.any(axis=1))
        self.enabled = geometry.allowed[self.shapes].ravel()
        self.forward_vertices = forward_vertices
        self.projection_jac = projection_jac
        self.roundoff = geometry.guard * .5
        self.calls = self.jacobian_calls = 0
        self.last_x = self.last_jac = None

    def fun(self, x):
        cfg = np.asarray(x[:self.cfg_size])
        if self.last_x is not None and np.array_equal(cfg, self.last_x):
            return self.values
        self.calls += 1
        self.last_x = cfg.copy()
        self.last_jac = None
        g = self.geometry
        if not len(self.shapes):
            self.values = np.empty(0)
            return self.values
        world = self.forward_vertices(cfg, self.shapes)
        projection = np.einsum('svc,tac->stva', world, g.axes)
        lo = projection.min(2)-g.radii[self.shapes, None, None]
        hi = projection.max(2)+g.radii[self.shapes, None, None]
        candidates = np.concatenate((lo-g.high, g.low-hi), axis=-1)
        winner = candidates.argmax(-1)
        separation = np.take_along_axis(candidates, winner[..., None], axis=-1)[..., 0]
        self.values = (separation-g.margin[self.shapes]-g.guard).ravel()[self.enabled]
        axes = g.axes.shape[1]
        normals = g.axes[np.arange(len(g.axes))[None, :], winner % axes]
        normals = normals * np.where(winner < axes, 1., -1.)[..., None]
        scores = np.einsum('svc,stc->stv', world, normals)
        tied = scores == scores.min(-1, keepdims=True)
        local = np.einsum('stv,svc->stc', tied, g.vertices[self.shapes])/tied.sum(-1)[..., None]
        self.local = local.reshape(-1, 3)[self.enabled]
        self.normals = normals.reshape(-1, 3)[self.enabled]
        self.indices = np.broadcast_to(g.indices[self.shapes, None], separation.shape).ravel()[self.enabled]
        return self.values

    def jac(self, x):
        self.fun(x)
        if self.last_jac is None:
            self.jacobian_calls += 1
            if len(self.values):
                self.last_jac = np.asarray(self.projection_jac(x[:self.cfg_size], self.indices,
                    self.local, self.normals), np.float64)
            else:
                self.last_jac = np.empty((0, self.cfg_size))
        result = np.zeros((len(self.values), len(x)))
        result[:, :self.cfg_size] = self.last_jac
        return result

    def feasible(self, x):
        return bool(np.all(self.fun(x) >= -self.roundoff))
