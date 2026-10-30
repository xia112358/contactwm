import pytest
import torch
from somaforge_core.tensor_field_packing import pack_fields,clone_fields


def test_packing_keeps_every_field_dtype_shape_and_value_and_owns_storage():
    original=dict(active=torch.tensor([True,False]),dist=torch.tensor([-.01,.02]),
        position=torch.arange(12,dtype=torch.float32).reshape(2,6)[:,::2],
        address=torch.arange(6,dtype=torch.int32).reshape(2,3),
        kind=torch.tensor([1,2],dtype=torch.int64),empty=torch.empty(0,4),
        scalar=torch.tensor(3.,dtype=torch.float64))
    packed=pack_fields(original);decoded=clone_fields(packed)
    assert len(packed['buffers'])==5
    assert set(original)==set(decoded)
    for name,value in original.items():
        assert decoded[name].dtype==value.dtype and decoded[name].shape==value.shape
        assert torch.equal(value,decoded[name])
    for buffer in packed['buffers']:buffer.zero_()
    for name,value in original.items():assert torch.equal(value,decoded[name])


def test_empty_fields_roundtrip():
    fields=dict(dist=torch.empty(0),address=torch.empty(0,6,dtype=torch.int32))
    result=clone_fields(pack_fields(fields))
    assert result['address'].shape==(0,6) and result['address'].dtype==torch.int32


def test_packing_rejects_overlapping_extents():
    packed=pack_fields(dict(a=torch.ones(3),b=torch.ones(2)))
    name,group,shape,offset,count=packed['fields'][1]
    packed['fields'][1]=(name,group,shape,0,count)
    with pytest.raises(ValueError,match='extent'):clone_fields(packed)


def test_ipc_client_acknowledges_only_after_owned_copy_synchronizes(monkeypatch):
    from types import SimpleNamespace
    from somaforge_core.newton_tensor_transport import TensorSceneClient
    events=[]
    fields=dict(dist=torch.tensor([-.01,.02]),active=torch.tensor([True,False]))
    packed=pack_fields(fields)
    class Connection:
        def send(self,message):events.append(message if isinstance(message,str) else 'query')
        def recv(self):return dict(packed=packed)
    class Pose:
        is_cuda=True
        device='cpu'
        def detach(self):return self
        def contiguous(self):return self
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:
        SimpleNamespace(synchronize=lambda:events.append('synchronize')))
    client=TensorSceneClient.__new__(TensorSceneClient)
    client.connection=Connection();client.packed_response=True
    result=client.query(Pose())
    assert events==['query','synchronize','copied']
    for buffer in packed['buffers']:buffer.zero_()
    for name,value in fields.items():assert torch.equal(result[name],value)
