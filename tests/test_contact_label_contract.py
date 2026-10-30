import numpy as np
import pytest
from somaforge_core.contact_labels import (
    ContactLabelPolicy, ContactLabelStream, label_contract,
    process_contact_sequence, validate_contract,
)


def observations():
    mask = np.ones((12, 2), bool)
    mask[3, 0] = False
    mask[7:10, 0] = False
    surface = np.where(mask, 1, -1)
    surface[4:6, 1] = 2
    point = np.arange(72).reshape(12, 2, 3).astype(float)
    return mask, surface, point


def test_offline_and_chunked_stream_are_identical():
    a, s, p = observations()
    policy = ContactLabelPolicy(on_frames=2, off_frames=2, surface_frames=3)
    offline = process_contact_sequence(a, s, p, fps=50, policy=policy)
    stream = ContactLabelStream(fps=50, policy=policy)
    rows = []
    for start, end in [(0, 4), (4, 8), (8, 12)]:
        rows.extend(stream.step(a[i], s[i], p[i]) for i in range(start, end))
    for key in rows[0]:
        np.testing.assert_array_equal(offline[key], np.stack([r[key] for r in rows]))
    assert offline['contact_part_mask'][3, 0] and not a[3, 0]
    assert not offline['contact_part_mask'][8, 0]
    assert (offline['contact_surface'][:, 1] == 1).all()


def test_default_preserves_actual_contact_and_normalizes_inactive_fields():
    a,s,p = observations()
    result = process_contact_sequence(a,s,p,fps=50)
    np.testing.assert_array_equal(result['contact_part_mask'], a)
    np.testing.assert_array_equal(result['contact_surface'], s)
    np.testing.assert_array_equal(result['contact_position_w'][a], p[a])
    assert (result['contact_position_w'][~a] == 0).all()


def test_filter_is_causal_and_does_not_mutate_raw_data():
    a,s,p = observations(); originals = [x.copy() for x in (a,s,p)]
    cfg=ContactLabelPolicy(off_frames=3)
    full=process_contact_sequence(a,s,p,fps=50,policy=cfg)
    prefix=process_contact_sequence(a[:8],s[:8],p[:8],fps=50,policy=cfg)
    np.testing.assert_array_equal(full['contact_part_mask'][:8],prefix['contact_part_mask'])
    for x,y in zip((a,s,p),originals):
        np.testing.assert_array_equal(x,y)


def test_wrong_fps_or_filter_contract_is_rejected():
    with pytest.raises(ValueError):
        validate_contract(label_contract(50), 100)
    with pytest.raises(ValueError):
        validate_contract(label_contract(50, ContactLabelPolicy(off_frames=2)), 50)
    with pytest.raises(ValueError):
        ContactLabelPolicy(off_frames=0)


def test_bad_observation_is_not_silently_coerced():
    with pytest.raises(ValueError):
        ContactLabelStream(fps=50).step([.5], [1], [[0,0,0]])
