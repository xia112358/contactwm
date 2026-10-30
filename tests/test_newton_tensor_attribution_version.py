from types import SimpleNamespace

import pytest
import somaforge_core.newton_tensor_transport as transport
from somaforge_core.contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA


@pytest.mark.parametrize('version', [None, 'newton_source_triangle_normal_fan_v1', SOURCE_NORMAL_FAN_SCHEMA])
def test_device_handshake_requires_current_face_attribution(monkeypatch, version):
    metadata = dict(schema='newton_cuda_ipc_stage_query_v1', surface_attribution_schema=version)
    monkeypatch.setattr(transport, 'Client', lambda *a, **kw: SimpleNamespace(recv=lambda: metadata))
    monkeypatch.setattr(transport, '_disable_nagle', lambda connection: None)
    if version == SOURCE_NORMAL_FAN_SCHEMA:
        assert transport.TensorSceneClient(1, b'key').metadata == metadata
    else:
        with pytest.raises(ValueError, match='current source-witness face attribution'):
            transport.TensorSceneClient(1, b'key')
