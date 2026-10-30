"""Authenticated loopback control with Torch CUDA IPC for dense query data.

Both processes keep tensor owners alive until the consumer has cloned on its
own GPU stream. Only explicit diagnostics/metadata travel as CPU objects.
"""
from multiprocessing.connection import Client, Listener
import torch
import torch.multiprocessing  # Registers CUDA tensor reductions for Connection.
from .contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA


def _disable_nagle(connection):
    # Tiny lifetime acknowledgements precede the next tensor request. Nagle
    # plus delayed ACK otherwise inserts ~40 ms despite data staying on GPU.
    import socket
    with socket.fromfd(connection.fileno(), socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def native_geometry_metadata(query, link_names):
    """Shared initialized geometry metadata for tensor and HTTP consumers."""
    catalog = {int(f['surface']): f for faces in query.shape_surface.values() for f in faces}
    metadata = dict(link_names=tuple(link_names), surface_catalog=list(catalog.values()),
        configured_margin=min(query.configured_terrain_includemargins), worlds=query.worlds,
        provenance=query.provenance, schema='newton_cuda_ipc_stage_query_v1',
        surface_attribution_schema=SOURCE_NORMAL_FAN_SCHEMA)
    if getattr(query, 'solid_geometry', None) is not None:
        metadata['solid_geometry'] = query.solid_geometry
    return metadata


def serve_tensor_queries(query, port, authkey):
    from .newton_contact_tensors import NewtonContactTensorReader
    reader = NewtonContactTensorReader(query.model, query.solver, query.body_env, query.shape_surface)
    query.tensor_reader = reader
    metadata = native_geometry_metadata(query, reader.link_names)
    import json
    with Listener(('127.0.0.1', port), authkey=authkey) as listener:
        print(json.dumps(dict(ready=True, port=port, transport='cuda_ipc')), flush=True)
        with listener.accept() as connection:
            _disable_nagle(connection)
            connection.send(metadata)
            while True:
                request = connection.recv()
                if request['op'] == 'close':
                    return
                if request['op'] != 'query':
                    raise ValueError('Unknown tensor query operation')
                q = request['q']
                try:
                    result = query.query_device(q)
                    valid = (torch.arange(len(result['dist']), device=q.device) < result['count']) & (result['worldid'] < len(q))
                    fields = ('dist', 'includemargin', 'type', 'worldid', 'active', 'constraint_allocated',
                        'constraint_rows', 'efc_address', 'shape0', 'shape1', 'body0', 'body1', 'source_key',
                        'primary_surface', 'part', 'task_pair', 'eligible', 'upward', 'full_kind',
                        'body_link0', 'body_link1', 'normal_w', 'geometry_point0_w', 'geometry_point1_w')
                    # Compact on GPU. These allocations remain alive until ack.
                    exported = {key: result[key][valid].contiguous() for key in fields}
                    exported.update({key: result[key][:len(q)].clone() for key in
                        ('contact_part_mask', 'contact_surface', 'contact_position_w')})
                    exported['solver_row'] = valid.nonzero().flatten()
                    if request.get('pack_response',False):
                        from .tensor_field_packing import pack_fields
                        response = dict(packed=pack_fields(exported))
                    else:
                        response = dict(tensors=exported)
                    if request.get('audit', False):
                        response['audit'] = query.audit_current(q)
                    connection.send(response)
                    if connection.recv() != 'copied':
                        raise ValueError('Missing CUDA IPC consumer acknowledgement')
                    del exported, result, q, request, response
                except Exception as exc:
                    connection.send(dict(error=f'{type(exc).__name__}: {exc}'))
                    raise


class TensorSceneClient:
    def __init__(self, port, authkey):
        self.connection = Client(('127.0.0.1', port), authkey=authkey)
        _disable_nagle(self.connection)
        self.metadata = self.connection.recv()
        self.packed_response = False
        if self.metadata.get('schema') != 'newton_cuda_ipc_stage_query_v1':
            raise ValueError('Unknown device query transport')
        if self.metadata.get('surface_attribution_schema') != SOURCE_NORMAL_FAN_SCHEMA:
            raise ValueError('Device query lacks current source-witness face attribution')

    def query(self, q, *, audit=False):
        if not q.is_cuda:
            raise ValueError('Tensor query forbids CPU pose transport')
        self.connection.send(dict(op='query', q=q.detach().contiguous(), audit=audit,
                                  pack_response=self.packed_response))
        response = self.connection.recv()
        if 'error' in response:
            raise RuntimeError(response['error'])
        if 'packed' in response:
            from .tensor_field_packing import clone_fields
            tensors=clone_fields(response['packed'])
        else:
            tensors = {key: value.clone() for key, value in response['tensors'].items()}
        audit_result = response.get('audit')
        # Ensure the IPC producer can release its allocations after ack.
        torch.cuda.current_stream(q.device).synchronize()
        del response
        self.connection.send('copied')
        return (tensors, audit_result) if audit else tensors

    def close(self):
        self.connection.send(dict(op='close'))
        self.connection.close()


class TensorSceneRouter:
    def __init__(self, clients):
        self.clients = clients
        self._device_catalogs = {}
        self.link_names = next(iter(clients.values())).metadata['link_names']
        if any(c.metadata['link_names'] != self.link_names for c in clients.values()):
            raise ValueError('Device scenes have different canonical body/link mappings')

    def __call__(self, q, scene_ids):
        batch_fields = ('contact_part_mask', 'contact_surface', 'contact_position_w')
        rows, outputs = [], {}
        configured = q.new_empty(len(q))
        normals, offsets, surfaces = [], [], []
        assigned = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        from .contact_face_selection import upward_face_mask
        for scene_id, client in self.clients.items():
            indices = (scene_ids == scene_id).nonzero().flatten()
            if not len(indices):
                continue
            catalog_key = (scene_id, q.device, q.dtype)
            if catalog_key not in self._device_catalogs:
                catalog = client.metadata['surface_catalog']
                upward = [face for face, valid in zip(catalog, upward_face_mask([f['normal_w'] for f in catalog])) if valid]
                self._device_catalogs[catalog_key] = (q.new_tensor([f['normal_w'] for f in upward]),
                    q.new_tensor([f['plane_offset'] for f in upward]),
                    scene_ids.new_tensor([f['surface'] for f in upward]))
            normal, offset, surface = self._device_catalogs[catalog_key]
            normals.append((indices, normal)); offsets.append((indices, offset)); surfaces.append((indices, surface))
            configured[indices] = client.metadata['configured_margin']
            assigned[indices] = True
            for chunk in indices.split(client.metadata['worlds']):
                result = client.query(q[chunk])
                result['sample'] = chunk[result['worldid']]
                for key in batch_fields:
                    if key not in outputs:
                        outputs[key] = result[key].new_empty((len(q), *result[key].shape[1:]))
                    outputs[key][chunk] = result.pop(key)
                rows.append(result)
        if not bool(assigned.all()):
            raise ValueError('Missing authoritative tensor scene route')
        # Static native catalogs are uploaded once per scene/device. No
        # contact/pose arrays are downloaded for routing or face association.
        face_count = max(len(value) for _, value in surfaces)
        face_normal = q.new_zeros(len(q), face_count, 3)
        face_offset = q.new_zeros(len(q), face_count)
        face_surface = scene_ids.new_full((len(q), face_count), -1)
        for (idx, normal), (_, offset), (_, surface) in zip(normals, offsets, surfaces):
            face_normal[idx, :len(normal)] = normal
            face_offset[idx, :len(offset)] = offset
            face_surface[idx, :len(surface)] = surface
        return dict(schema='newton_device_witness_batch_v1', surface_attribution_schema=SOURCE_NORMAL_FAN_SCHEMA,
            pairs={key: torch.cat([row[key] for row in rows]) for key in rows[0]},
            **outputs, configured_margin=configured, link_names=self.link_names,
            face_normal=face_normal, face_offset=face_offset, face_surface=face_surface)
