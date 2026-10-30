"""Unobserved native worlds must remain unknown and mappings must be recoverable."""
import unittest
from types import SimpleNamespace
import numpy as np
from supermimic.contact_native import NativeContacts

class ObservationScopeTest(unittest.TestCase):
    def reader(self):
        r=NativeContacts.__new__(NativeContacts);r.direct=False;r.env=SimpleNamespace(N=3)
        r.binding={'body_env':{10:0,20:1,30:2,31:2},'shape_surface':{'untouched':1}}
        return r

    def test_scoping_and_restoration(self):
        r=self.reader();original=dict(r.binding['body_env']);r.evidence_binding=object()
        r.set_observed_worlds(np.array([2],dtype=np.int64))
        self.assertEqual(r.binding['body_env'],{30:2,31:2})
        self.assertEqual(r._observed_world_mask.tolist(),[False,False,True])
        self.assertFalse(hasattr(r,'evidence_binding'))
        self.assertEqual(r.binding['shape_surface'],{'untouched':1})
        r.set_observed_worlds([0,1,2]);self.assertEqual(r.binding['body_env'],original)

    def test_invalid_scopes(self):
        for ids in [[],[1,1],[-1],[3],[1.5],[[1]]]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):self.reader().set_observed_worlds(ids)

    def test_direct_reader_explicitly_rejected(self):
        r=self.reader();r.direct=True
        with self.assertRaises(ValueError):r.set_observed_worlds([0])

if __name__=='__main__':unittest.main()
