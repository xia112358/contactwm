import numpy as np
import pytest
from somaforge_core.contact_record_storage import compact_snapshot,stack_contact_channel


def test_compact_preserves_candidates_and_owns_memory():
    original=dict(count=np.array(2),active=np.array([True,False,True,True]),
                  position_w=np.arange(12).reshape(4,3),efc_address=np.arange(16).reshape(4,4))
    result=compact_snapshot(original)
    assert result['active'].tolist()==[True,False]
    for key in ('active','position_w','efc_address'):
        np.testing.assert_array_equal(result[key],original[key][:2])
        assert result[key].base is None
    result['position_w'][0]=99
    assert original['position_w'][0,0]==0


def test_stack_variable_and_zero_contact_counts():
    a=np.ones((2,3),np.float32);b=np.ones((4,3),np.float32)*2
    result=stack_contact_channel([a,np.empty((0,3),np.float32),b])
    assert result.shape==(3,4,3)
    np.testing.assert_array_equal(result[0,:2],a)
    np.testing.assert_array_equal(result[2],b)
    assert not result[1].any()
    assert stack_contact_channel([np.empty((0,3))]).shape==(1,0,3)


def test_overflow_not_silently_truncated():
    with pytest.raises(ValueError):compact_snapshot(dict(count=5,active=np.zeros(3)))
