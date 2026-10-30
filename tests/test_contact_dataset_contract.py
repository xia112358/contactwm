import json
import numpy as np
import pytest
from somaforge_core.contact_dataset import contact_dataset_fingerprint, validate_contact_cache


def test_cache_tracks_label_content_and_requires_provenance(tmp_path):
    np.savez(tmp_path/'motion.npz',fps=np.array(50.),joint_pos=np.zeros((2,36)))
    (tmp_path/'labels.npz').write_bytes(b'first')
    (tmp_path/'terrain.obj').write_text('v 0 0 0\n')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(motion_files=[dict(motion_id=0,terrain_id=0,
        motion_file='motion.npz',newton_contact_file='labels.npz')],
        terrains=[dict(terrain_id=0,terrain_file='terrain.obj')])))
    cache=dict(contact_dataset_fingerprint=contact_dataset_fingerprint(manifest))
    validate_contact_cache(cache,manifest)
    with pytest.raises(ValueError):
        validate_contact_cache({},manifest)
    (tmp_path/'labels.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):
        validate_contact_cache(cache,manifest)


def test_stream_does_not_double_count_shared_boundary(monkeypatch):
    import torch
    import somaforge_core.newton_contact_query as backend
    from contact_solver.contact_validity import query_q_contact_sequence
    def query(q,scene):
        n=len(q)
        return dict(active=np.ones((n,6),bool),surface=np.zeros((n,6),int),position_w=np.zeros((n,6,3)),
            surface_catalog=[dict(surface=0,normal_w=[0,0,1])],
            pairs=[[dict(part=p,surface=0,allocated=True,dist=0.,includemargin=.02,
                         position_w=[0,0,0]) for p in range(6)] for _ in range(n)])
    monkeypatch.setattr(backend,'query_contacts',query)
    q=torch.zeros((3,36));q[:,0]=torch.arange(3)
    scene=(torch.zeros((3,3)),)
    _,stream=query_q_contact_sequence(q,scene,fps=50)
    calls=[];step=stream.step
    def counted(*args):
        calls.append(1);return step(*args)
    stream.step=counted
    q2=q.clone();q2[:,0]+=2
    result,_=query_q_contact_sequence(q2,scene,fps=50,stream=stream,shared_boundary=True)
    assert len(calls)==2 and result['active'].shape==(3,6)
    with pytest.raises(ValueError):
        query_q_contact_sequence(q,scene,fps=50,stream=stream,shared_boundary=True)
