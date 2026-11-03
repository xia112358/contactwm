import unittest

from supermimic.contact_source import require_original_contact_source


class OriginalContactSourceTest(unittest.TestCase):
    def test_original_source_allowed(self):
        require_original_contact_source(dict(source_unit='original_database_clip',
            composed_trajectories=False, tasks=[dict(source_motion_id='climb_76/range_000')]))

    def test_composed_sources_cannot_be_relabelled(self):
        examples = [dict(source_component='php_all_five_groups_296'),
                    dict(source_component='motions-low-climb-76'),
                    dict(tasks=[dict(source_motion_id='motions-low-climb-76_0030')]),
                    dict(composed_trajectories=True)]
        for example in examples:
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, 'must not be re-sliced'):
                require_original_contact_source(dict(source_unit='original_database_clip', **example))

    def test_unknown_source_is_not_assumed_original(self):
        with self.assertRaisesRegex(ValueError, 'source-frame provenance'):
            require_original_contact_source({})


if __name__ == '__main__':
    unittest.main()
