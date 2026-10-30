"""Physics changes must not silently reuse historical checkpoints."""
import unittest
from supermimic.contact_response import contract, metadata, validate_identity

class ContactResponseTest(unittest.TestCase):
    def test_legacy_identity(self):
        validate_identity({}, {})
        validate_identity({}, {'contact_response_contract': metadata('legacy')})

    def test_cross_physics_rejected_both_directions(self):
        firm={'contact_response_contract':metadata('firm_v2')}
        for source,target in [({},firm),(firm,{})]:
            with self.assertRaisesRegex(ValueError,'Contact-response physics changed'):
                validate_identity(source,target)
        validate_identity(firm,firm)

    def test_time_resolution(self):
        spec=contract('firm_v2')
        self.assertEqual(spec['control_dt']/spec['substeps'],.002)
        self.assertGreaterEqual(spec['solref'][0],2*spec['control_dt']/spec['substeps'])
        self.assertEqual(spec['margins'],'unchanged')
        self.assertEqual(spec['geometry'],'unchanged')

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):contract('typo')

    def test_diagnostic_data_rejected(self):
        with self.assertRaisesRegex(ValueError,'Diagnostic'):
            validate_identity({'production_compatible':False}, {})

    def test_manifold_is_a_distinct_physics_contract(self):
        self.assertTrue(contract('manifold_v1')['enable_multiccd'])
        self.assertFalse(contract('firm_v2')['enable_multiccd'])
        with self.assertRaises(ValueError):
            validate_identity({'contact_response_contract':metadata('firm_v2')},
                              {'contact_response_contract':metadata('manifold_v1')})

if __name__=='__main__':unittest.main()
