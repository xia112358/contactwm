"""Physics changes must not silently reuse historical checkpoints."""
import unittest
from supermimic.contact_response import contract, metadata, validate_identity

class ContactResponseTest(unittest.TestCase):
    def test_cg_numerics_are_explicit_and_do_not_relabel_old_newton_runs(self):
        old,new=contract('newton_contacts_v2'),contract('newton_contacts_v3')
        self.assertEqual(new['constraint_solver'],'cg')
        for key in ('substeps','solref','solimp','collision_decimation','geometry','margins','gaps'):
            self.assertEqual(old[key],new[key])
        with self.assertRaises(ValueError):
            validate_identity({'contact_response_contract':metadata('newton_contacts_v2')},
                              {'contact_response_contract':metadata('newton_contacts_v3')})

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

    def test_projection_normal_is_a_new_physics_contract(self):
        old, new = contract('manifold_v2'), contract('manifold_v3')
        self.assertEqual(new['manifold_normal'], 'epa_projection_manifold_normal_v2')
        self.assertEqual(old['manifold_normal'], 'epa_witness_manifold_normal_v1')
        for key in ('substeps', 'solref', 'solimp', 'geometry', 'margins', 'gaps'):
            self.assertEqual(old[key], new[key])
        with self.assertRaises(ValueError):
            validate_identity({'contact_response_contract': metadata('manifold_v2')},
                              {'contact_response_contract': metadata('manifold_v3')})

    def test_newton_contacts_preserve_response_but_change_pipeline(self):
        old, new = contract('manifold_v3'), contract('newton_contacts_v1')
        for key in ('substeps', 'solref', 'solimp', 'geometry', 'margins', 'gaps'):
            self.assertEqual(old[key], new[key])
        self.assertEqual(new['contact_generation'], 'newton_collision_pipeline')
        self.assertEqual(new['manifold_normal'], 'upstream')
        for source, target in [('manifold_v3', 'newton_contacts_v1'),
                               ('newton_contacts_v1', 'manifold_v3')]:
            with self.assertRaises(ValueError):
                validate_identity({'contact_response_contract': metadata(source)},
                                  {'contact_response_contract': metadata(target)})

    def test_per_substep_contacts_are_a_distinct_contract(self):
        old, new = contract('newton_contacts_v1'), contract('newton_contacts_v2')
        self.assertEqual(new['collision_decimation'], 1)
        self.assertEqual(new['control_dt'] / new['substeps'], .002)
        self.assertEqual(old['contact_generation'], new['contact_generation'])
        for source, target in [('newton_contacts_v1', 'newton_contacts_v2'),
                               ('newton_contacts_v2', 'newton_contacts_v1')]:
            with self.assertRaises(ValueError):
                validate_identity({'contact_response_contract': metadata(source)},
                                  {'contact_response_contract': metadata(target)})

if __name__=='__main__':unittest.main()
