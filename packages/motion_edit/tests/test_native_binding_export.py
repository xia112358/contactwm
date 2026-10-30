import pytest


def test_native_binding_export_and_relabel_share_only_identical_inspection(tmp_path):
    from serve_newton_contact_queries import write_verified_inspection
    path=tmp_path/'model.json';info=dict(model_fingerprint='same',shape_body=[0,-1])
    write_verified_inspection(path,info);original=path.read_bytes()
    write_verified_inspection(path,info)
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='different'):
        write_verified_inspection(path,dict(model_fingerprint='changed',shape_body=[0,-1]))
    assert path.read_bytes()==original
