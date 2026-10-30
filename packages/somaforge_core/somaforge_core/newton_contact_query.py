"""Contact queries against an explicitly supplied, dedicated Newton scene.

The caller owns scene construction from the policy configuration. This module
does not reconstruct a second URDF/mesh/margin/filter configuration.
"""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import numpy as np

from .newton_contacts import SCHEMA, PARTS, reduce_snapshot, snapshot_solver_contacts
from .contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA

_provider = None


def install_contact_query(provider):
    """Install an in-process query for a dedicated evaluation scene only."""
    global _provider
    _provider = provider


def query_contacts(qpos, scene=None, *, endpoint=None):
    """Batch q [N,36], scene=(center, rotation, half, ground), all same frame.

    Returns realized masks independent of the predictor's desired contacts.
    A loopback worker may host the authoritative simulation to keep Kit out of
    tensor training. No future pose/label enters predictor inputs through here.
    """
    q = np.asarray(qpos, np.float32).copy()
    if q.ndim != 2 or q.shape[1] != 36 or not np.isfinite(q).all():
        raise ValueError('Contact query requires finite [N,36] canonical qpos')
    values = None if scene is None else [np.asarray(x, np.float32) for x in scene]
    expected = [(len(q), 3), (len(q), 3, 3), (len(q), 3), (len(q),)]
    if values is not None and (len(values) != 4 or any(x.shape != shape or not np.isfinite(x).all()
                              for x, shape in zip(values, expected))):
        raise ValueError('Contact scene batch/frame mismatch')
    if _provider is not None and endpoint is None:
        result = _provider(q, values)
    else:
        endpoint = endpoint if endpoint is not None else os.environ.get('SOMAFORGE_NEWTON_CONTACT_URL', '')
        parsed = urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise RuntimeError('Set SOMAFORGE_NEWTON_CONTACT_URL to a local authoritative Newton query worker, '
                               'or install_contact_query in a dedicated evaluation scene; no geometric fallback')
        request = Request(endpoint, data=json.dumps(dict(qpos=q.tolist(), scene=None if values is None else [x.tolist() for x in values])).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError(result['error'])
    if result.get('schema') != SCHEMA or tuple(result.get('parts', ())) != PARTS:
        raise ValueError('Invalid Newton query semantics/part mapping')
    output = dict(result)
    for key, shape, dtype in [('active', (len(q), 6), bool), ('unallocated', (len(q), 6), bool),
                             ('position_w', (len(q), 6, 3), np.float32), ('surface', (len(q), 6), np.int64)]:
        value = np.asarray(result[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid Newton query {key}')
        if dtype is bool and not np.isin(value, [0, 1]).all():
            raise ValueError(f'Nonbinary Newton query {key}')
        output[key] = value.astype(dtype)
    if output['unallocated'].any():
        raise RuntimeError('Newton active contact constraints were not allocated; increase query capacity explicitly')
    if (output['surface'][output['active']] < 0).any():
        raise ValueError('Active Newton contact has no surface mapping')
    return output


class DedicatedNewtonSceneQuery:
    """Use the initialized policy model and solver, never an active training env.

    scene is its known box/ground descriptor; different dimensions are rejected,
    not approximated. Rigid changes of frame are allowed and transform q and
    result together. Each terrain geometry needs its own initialized binding.
    """
    def __init__(self, model, solver, pipeline, *, scene, body_env, shape_surface,
                 joint_q_indices, root_q_start, model_provenance, capture_contact_sources=False,
                 audit_device_reader=False):
        if solver.use_mujoco_cpu or solver.mjw_model.opt.run_collision_detection:
            raise ValueError('Query requires dedicated Newton/MJWarp worlds with Newton collision pipeline')
        self.worlds = max(1, int(model.world_count))
        if self.worlds > 1 and scene is not None:
            raise ValueError('Batched query currently requires the native shared terrain frame')
        from .robot_assets import validate_g1_asset_metadata
        validate_g1_asset_metadata(model_provenance['robot_asset'], context='Newton query scene')
        if not model_provenance.get('model_fingerprint'):
            raise ValueError('Query requires a fingerprint of the actual initialized model')
        self.model, self.solver, self.pipeline = model, solver, pipeline
        self.scene = None if scene is None else [np.asarray(x, np.float64) for x in scene]
        self.body_env, self.shape_surface = body_env, shape_surface
        if any(not isinstance(v, list) for v in shape_surface.values()):
            raise ValueError('Query surface bindings require explicit face normals/classes, not mesh-as-top IDs')
        self.indices = np.asarray(joint_q_indices, int)
        self.root = int(root_q_start) if self.worlds == 1 else np.asarray(root_q_start, int)
        if self.indices.shape != ((29,) if self.worlds == 1 else (self.worlds, 29)):
            raise ValueError('Expected canonical 29 joint coordinate indices')
        if self.worlds > 1:
            if self.root.shape != (self.worlds,):
                raise ValueError('Expected one free-root coordinate mapping per world')
            shape_world = model.shape_world.numpy()
            if any(shape_world[shape] != -1 for shape in shape_surface):
                raise ValueError('Batched native queries require actual globally shared terrain shapes')
        self.provenance = model_provenance
        self.state = model.state(); self.contacts = pipeline.contacts()
        self._static_mapping_cache = {}
        self.configured_terrain_includemargins = self._terrain_includemargins()
        self.source_capture = None
        if capture_contact_sources:
            from .newton_contact_sources import PassiveContactSourceCapture
            self.source_capture = PassiveContactSourceCapture(pipeline,self.contacts,model=model)
        self.audit_device_reader = audit_device_reader
        self.tensor_reader = None

    def _read_device(self):
        if self.source_capture is None or self.scene is not None:
            raise ValueError('Device reader requires native terrain and captured actual contact sources')
        if self.tensor_reader is None:
            from .newton_contact_tensors import NewtonContactTensorReader
            self.tensor_reader = NewtonContactTensorReader(self.model, self.solver, self.body_env, self.shape_surface)
        result = self.tensor_reader.read(self.state, self.contacts, self.source_capture)
        self.tensor_reader.validate(result)
        return result

    def query_device(self, qpos):
        """Query one batch of independent stage endpoints on the caller's GPU.

        Returns raw solver tensors, material witnesses and six-part effective
        observations. Raw tensor views expire at the next query. No HTTP or
        NumPy q conversion is involved; CPU export is explicit via audit_current.
        """
        import torch
        if not isinstance(qpos, torch.Tensor) or not qpos.is_cuda or qpos.ndim != 2 or qpos.shape[1] != 36:
            raise ValueError('Device query requires canonical CUDA qpos [N,36]')
        if not 0 < len(qpos) <= self.worlds or self.scene is not None:
            raise ValueError('Device query batch must fit dedicated native worlds')
        result = self._batch_snapshot(qpos.detach(), device_only=True)
        result['contact_part_mask'] = result['contact_part_mask'][:len(qpos)]
        result['contact_surface'] = result['contact_surface'][:len(qpos)]
        result['requested_worlds'] = len(qpos)
        return result

    def audit_current(self, qpos):
        """Export this same device collision pass, only for explicit auditing."""
        import torch
        import warp as wp
        from .newton_contact_sources import (canonical_source_keys, expand_native_sphere_triangle_keys,
            attach_solver_sources, physical_contact_signature)
        expected = qpos[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]]
        actual = wp.to_torch(self.state.joint_q)[self._gpu_coordinate_indices[:len(qpos)]]
        if not torch.equal(expected, actual):
            raise ValueError('Audit q differs from the current device query state')
        raw = snapshot_solver_contacts(self.solver, self.model, state=self.state, contacts=self.contacts,
                                       static_mapping_cache=self._static_mapping_cache)
        before = physical_contact_signature(raw)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        keys = canonical_source_keys(self.source_capture.keys.numpy()[:count],
            self.source_capture.shape_bits, self.source_capture.sub_key_bits)
        if self.source_capture.analytic_sphere_types is not None:
            keys = expand_native_sphere_triangle_keys(keys, self.source_capture.analytic_sphere_types)
        attach_solver_sources(raw, self.solver, self.contacts, keys)
        if before != physical_contact_signature(raw):
            raise AssertionError('Device audit export mutated physical contact evidence')
        raw['same_pass_source_mapping_verified'] = True
        return self._query(qpos.detach().cpu().numpy(), None, _snapshot_override=raw)

    def _terrain_includemargins(self):
        """Read potential task-pair margins from this realized MJWarp model."""
        import newton
        from .contact_schema import CONTACT_BODY_NAMES_BY_PART
        if self.solver.newton_shape_to_mjc_geom is None:
            self.solver._create_inverse_shape_mapping()
        mapping = self.solver.newton_shape_to_mjc_geom.numpy()
        margin = self.solver.mjw_model.geom_margin.numpy()
        gap = self.solver.mjw_model.geom_gap.numpy()
        if margin.shape[0] not in (1, self.worlds) or gap.shape != margin.shape:
            raise ValueError('Invalid realized MJWarp world mapping for contact margins')
        body = self.model.shape_body.numpy()
        flags = self.model.shape_flags.numpy()
        robot_shapes = []
        for shape, body_index in enumerate(body):
            if (int(body_index) not in self.body_env
                    or not int(flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
                continue
            name = str(self.model.body_label[int(body_index)]).rsplit('/', 1)[-1]
            if any(name in names for names in CONTACT_BODY_NAMES_BY_PART.values()):
                robot_shapes.append(shape)
        def realized_values():
            current_margin = self.solver.mjw_model.geom_margin.numpy()
            current_gap = self.solver.mjw_model.geom_gap.numpy()
            values = []
            for robot_shape in robot_shapes:
                for terrain_shape in self.shape_surface:
                    robot_geom = int(mapping[robot_shape])
                    terrain_geom = int(mapping[terrain_shape])
                    if robot_geom < 0 or terrain_geom < 0:
                        continue
                    world = self.body_env[int(body[robot_shape])]
                    row = 0 if current_margin.shape[0] == 1 else world
                    # Newton main / MJWarp contact_params writes the sum of
                    # geom margins to contact.includemargin.  geom_gap is a
                    # separate detection-envelope quantity and is not
                    # subtracted from the activation threshold.
                    values.append(float(
                        current_margin[row, robot_geom]
                        + current_margin[row, terrain_geom]
                    ))
            return values

        values = realized_values()
        positive = sorted({value for value in values if value > 0.0})
        if not positive and hasattr(self.solver, "_update_geom_properties"):
            # Newton main initializes some MJWarp geometry buffers lazily.
            # Invoke its own property synchronization, then re-read the actual
            # solver fields.  This changes no model margin/gap configuration.
            self.solver._update_geom_properties()
            values = realized_values()
            positive = sorted({value for value in values if value > 0.0})
        if not positive:
            raise ValueError(
                'Realized MJWarp task pairs have no positive includemargin: '
                f'pair_values={sorted(set(values))[:16]}, '
                f'model_shape_margin={sorted(set(map(float, self.model.shape_margin.numpy())))[-8:]}, '
                f'model_shape_gap={sorted(set(map(float, self.model.shape_gap.numpy())))[-8:]}, '
                f'mjw_geom_margin={sorted(set(map(float, self.solver.mjw_model.geom_margin.numpy().reshape(-1))))[-8:]}, '
                f'mjw_geom_gap={sorted(set(map(float, self.solver.mjw_model.geom_gap.numpy().reshape(-1))))[-8:]}'
            )
        return positive

    def _batch_snapshot(self, qpos, *, device_only=False):
        """One fixed-q FK/collision/constraint pass for independent stage outputs.

        Write through a GPU tensor view of the actual Newton joint coordinates.
        Tail worlds repeat the last requested pose and are excluded by worldid
        when returning rows. The legacy CPU snapshot remains the audit reader.
        """
        import newton
        import torch
        import warp as wp
        from .newton_contact_sources import capture_constraint_snapshot
        q = torch.as_tensor(qpos, dtype=torch.float32, device=str(self.model.device))
        if len(q) < self.worlds:
            q = torch.cat((q, q[-1:].expand(self.worlds-len(q), -1)))
        if not bool(torch.isfinite(q).all()) or not bool(torch.isclose(
                q[:, 3:7].norm(dim=-1), q.new_ones(self.worlds), atol=1e-4, rtol=0).all()):
            raise ValueError('Batched query requires finite q and unit root quaternions')
        if not hasattr(self, '_gpu_coordinate_indices'):
            root = torch.as_tensor(self.root, device=q.device).reshape(self.worlds)
            self._gpu_coordinate_indices = torch.cat((root[:, None]+torch.arange(7, device=q.device),
                torch.as_tensor(self.indices, device=q.device).reshape(self.worlds, 29)), dim=1)
        # Warp and Torch use the same CUDA stream for this dependent sequence.
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(q.device))):
            coordinates = wp.to_torch(self.state.joint_q)
            coordinates[self._gpu_coordinate_indices] = q[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]]
            self.state.joint_qd.zero_()
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            if device_only:
                if self.source_capture is None:
                    raise ValueError('Device query requires actual source capture')
                self.source_capture.collide_device(self.state, self.contacts)
                self.solver._update_mjc_data(self.solver.mjw_data, self.model, self.state)
                self.solver._convert_contacts_to_mjwarp(self.model, self.state, self.contacts)
                self.solver._mujoco_warp.fwd_position(self.solver.mjw_model, self.solver.mjw_data, factorize=False)
                result = self._read_device()
            else:
                result = capture_constraint_snapshot(self.pipeline, self.state, self.contacts, self.solver,
                    self._constraint_snapshot, self.source_capture)
            if self.audit_device_reader and not torch.equal(coordinates[self._gpu_coordinate_indices],
                    q[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]]):
                raise AssertionError('Fixed-q query changed supplied joint coordinates')
            return result

    def _constraint_snapshot(self):
        import warp as wp
        from .newton_collision_compat import check_device
        check_device(self.model.device)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count > min(self.contacts.rigid_contact_max,len(self.solver.mjw_data.contact.dist)):
            raise RuntimeError('Newton query contact capacity overflow')
        self.solver._update_mjc_data(self.solver.mjw_data,self.model,self.state)
        self.solver._convert_contacts_to_mjwarp(self.model,self.state,self.contacts)
        with wp.ScopedDevice(self.model.device):
            self.solver._mujoco_warp.fwd_position(self.solver.mjw_model,self.solver.mjw_data,factorize=False)
        snapshot = snapshot_solver_contacts(self.solver,self.model,state=self.state,contacts=self.contacts,
                                           static_mapping_cache=self._static_mapping_cache)
        if self.audit_device_reader:
            self.last_device_snapshot = self._read_device()
            count = int(snapshot['count'])
            for key in ('dist', 'includemargin', 'type', 'worldid', 'active', 'constraint_allocated',
                        'efc_address', 'constraint_rows', 'shape0', 'shape1', 'body0', 'body1'):
                if not np.array_equal(snapshot[key][:count], self.last_device_snapshot[key][:count].cpu().numpy()):
                    raise AssertionError(f'Device/CPU same-pass mismatch: {key}')
        return snapshot

    def __call__(self, qpos, scenes):
        try:
            return self._query(qpos,scenes)
        except RuntimeError as error:
            # A failed CUDA context cannot be read back. Preserve the CPU request
            # so a fresh worker can reproduce it without losing the input pose.
            directory=os.environ.get('SOMAFORGE_CONTACT_FAILURE_DUMP')
            if directory:
                import time
                from pathlib import Path
                from .robot_assets import somaforge_root
                output=Path(directory).resolve()
                if not output.is_relative_to((somaforge_root()/'tmp').resolve()):
                    raise ValueError('Contact diagnostics must be inside project tmp') from error
                output.mkdir(parents=True,exist_ok=True)
                payload=dict(error=str(error),qpos=np.asarray(qpos).tolist(),
                    scenes=None if scenes is None else [np.asarray(v).tolist() for v in scenes],
                    sample_index=getattr(self,'_active_query_index',None),provenance=self.provenance)
                (output/f'failed_request_{time.time_ns()}.json').write_text(json.dumps(payload))
            raise

    def _query(self, qpos, scenes, *, _snapshot_override=None):
        import newton
        from scipy.spatial.transform import Rotation
        rows = []
        separation = []
        if _snapshot_override is not None and (scenes is not None or self.scene is not None or len(qpos) > self.worlds):
            raise ValueError('Same-pass audit must fit the native query worlds')
        if self.scene is None:
            if scenes is not None:
                raise ValueError('Native-scene worker cannot assume a box/ground descriptor')
            center, basis, half, ground = np.zeros(3), np.eye(3), np.zeros(3), 0.
            scenes = [np.broadcast_to(x, (len(qpos), *np.asarray(x).shape)) for x in (center,basis,half,np.asarray(ground))]
        else:
            if scenes is None:
                raise ValueError('Frame-bound worker requires an explicit query scene')
            center, basis, half, ground = self.scene
        batch_raw = None
        for sample_index,(q, c, r, h, g) in enumerate(zip(qpos, *scenes)):
            self._active_query_index=sample_index
            if not np.isclose(np.linalg.norm(q[3:7]),1.,atol=1e-4,rtol=0):
                raise ValueError('Contact query refuses a non-unit root quaternion; no silent q normalization')
            if not np.allclose(h, half, atol=1e-6, rtol=0):
                raise ValueError('Terrain dimensions differ from initialized Newton scene; initialize the matching scene')
            transform = basis @ r.T
            translation = center-transform@c
            if (not np.allclose(transform.T@transform, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(transform), 1., atol=1e-5)
                    or not np.allclose(transform[2], [0, 0, 1], atol=1e-5)
                    or not np.isclose(g+translation[2], ground, atol=1e-5)):
                raise ValueError('Query frame does not preserve the initialized ground/box scene')
            world = sample_index % self.worlds
            if _snapshot_override is not None:
                raw = _snapshot_override
            elif self.worlds > 1:
                if world == 0:
                    batch_raw = self._batch_snapshot(qpos[sample_index:sample_index+self.worlds])
                raw = batch_raw
            else:
                joint_q = self.model.joint_q.numpy().copy()
                root = self.root
                joint_q[root:root+3] = transform@q[:3]+translation
                joint_q[root+3:root+7] = (Rotation.from_matrix(transform)*Rotation.from_quat(q[[4,5,6,3]])).as_quat()
                joint_q[self.indices] = q[7:]
                self.state.joint_q.assign(joint_q); self.state.joint_qd.zero_()
                newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
                from .newton_contact_sources import capture_constraint_snapshot
                raw=capture_constraint_snapshot(self.pipeline,self.state,self.contacts,self.solver,
                                                self._constraint_snapshot,self.source_capture)
            from .newton_contacts import full_robot_separation
            full_separation=full_robot_separation(raw,body_env=self.body_env,shape_surface=self.shape_surface,env_id=world)
            witnesses=[full_separation[kind] for kind in ('worst_terrain','worst_self','closest_terrain','closest_self')]
            witnesses+=full_separation['invalid_penetrating_witnesses']
            for witness in witnesses:
                if witness is None:continue
                for side in (0,1):
                    body=witness[f'body{side}']
                    witness[f'body_name{side}']=(str(self.model.body_label[body]).rsplit('/',1)[-1]
                        if body in self.body_env else None)
                    field=f'point{side}_w'
                    if field not in witness:raise ValueError('Missing full-body geometry witness')
                    witness[field]=((np.asarray(witness[field])-translation)@transform).tolist()
                witness['normal_w']=(np.asarray(witness['normal_w'])@transform).tolist()
            separation.append(full_separation)
            try:
                reduced = reduce_snapshot(raw, body_labels=self.model.body_label,
                    body_env=self.body_env, shape_surface=self.shape_surface, include_candidates=True,env_id=world)
            except ValueError:
                # Opt-in forensic capture, never a fallback contact classifier.
                directory = os.environ.get('SOMAFORGE_CONTACT_FAILURE_DUMP')
                if directory:
                    from pathlib import Path
                    import time
                    from .robot_assets import somaforge_root
                    output = Path(directory).resolve()
                    if not output.is_relative_to((somaforge_root()/'tmp').resolve()):
                        raise ValueError('Contact diagnostics must be inside project tmp')
                    output.mkdir(parents=True, exist_ok=True)
                    n = int(self.contacts.rigid_contact_count.numpy()[0])
                    arrays = {f'snapshot_{key}': value for key,value in raw.items()}
                    for key in ('point0','point1','offset0','offset1','margin0','margin1','shape0','shape1','normal','point_id'):
                        arrays['raw_'+key] = getattr(self.contacts,'rigid_contact_'+key).numpy()[:n].copy()
                    arrays.update(qpos=q,body_q=self.state.body_q.numpy(),shape_margin=self.model.shape_margin.numpy(),
                        raw_to_solver=self.solver._contact_tid_to_cid.numpy()[:n],
                        bindings_json=np.asarray(json.dumps(self.shape_surface)))
                    with (output/f'failure_{time.time_ns()}.npz').open('xb') as stream:
                        np.savez_compressed(stream,**arrays)
                raise
            reduced.position_w = (reduced.position_w-translation)@transform
            reduced.position_w[~reduced.active] = 0
            for pair in reduced.pairs + reduced.candidate_pairs:
                pair['position_w'] = ((np.asarray(pair['position_w'])-translation)@transform).tolist()
                pair['solver_position_w'] = ((np.asarray(pair['solver_position_w'])-translation)@transform).tolist()
                if pair.get('terrain_position_w') is not None:
                    pair['terrain_position_w'] = ((np.asarray(pair['terrain_position_w'])-translation)@transform).tolist()
                if 'contact_source' in pair:
                    pair['contact_source'] = dict(pair['contact_source'])
                    if pair['contact_source'].get('source_triangle_w') is not None:
                        pair['contact_source']['source_triangle_w'] = ((np.asarray(
                            pair['contact_source']['source_triangle_w'])-translation)@transform).tolist()
                    if pair['contact_source'].get('support_vertices_w') is not None:
                        pair['contact_source']['support_vertices_w'] = ((np.asarray(
                            pair['contact_source']['support_vertices_w'])-translation)@transform).tolist()
                    if pair['contact_source'].get('source_feature_projection_w') is not None:
                        pair['contact_source']['source_feature_projection_w'] = ((np.asarray(
                            pair['contact_source']['source_feature_projection_w'])-translation)@transform).tolist()
                pair['normal_w'] = (np.asarray(pair['normal_w'])@transform).tolist()
            rows.append(reduced)
            if self.audit_device_reader:
                from .contact_face_selection import select_contact_pairs
                catalog = {int(face['surface']): face for faces in self.shape_surface.values() for face in faces}
                selected = select_contact_pairs([reduced.pairs], list(catalog.values()))
                for key in ('contact_part_mask', 'contact_surface'):
                    if not np.array_equal(selected[key][0], self.last_device_snapshot[key][world].cpu().numpy()):
                        raise AssertionError(f'Device/CPU same-pass effective contact mismatch: {key}, sample {sample_index}')
                for pair in reduced.candidate_pairs:
                    row = pair['solver_row']
                    if int(self.last_device_snapshot['primary_surface'][row]) != pair['surface']:
                        raise AssertionError('Device/CPU same-pass primary surface mismatch')
        catalog = {int(face['surface']): face for faces in self.shape_surface.values() for face in faces}
        from .newton_collision_compat import SCHEMA as collision_numeric_schema
        return dict(schema=SCHEMA, parts=PARTS, provenance=self.provenance,
            collision_numeric_schema=collision_numeric_schema,
            source_key_encoding=(dict(self.source_capture.encoding)
                if self.source_capture is not None else None),
            source_capture_contract=('single_collision_pass_raw_to_solver_verified_v2'
                if self.source_capture is not None else None),
            surface_attribution_schema=(SOURCE_NORMAL_FAN_SCHEMA
                if self.source_capture is not None else 'legacy_nearest_face'),
            configured_terrain_includemargins=self.configured_terrain_includemargins,
            configured_margin_source='realized_mjwarp_contact_params_margin_sum_v2',
            surface_catalog=list(catalog.values()),
            sampling='fixed supplied q; forward constraint evaluation; no integration',
            **{k: np.stack([getattr(x, k) for x in rows]).tolist()
               for k in ('active', 'unallocated', 'position_w', 'surface')},
            pairs=[x.pairs for x in rows],candidate_pairs=[x.candidate_pairs for x in rows],
            full_robot_separation=separation)
