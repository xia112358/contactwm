import unittest

from motion_edit.contact.raw_contacts import PART_ALIASES
import numpy as np

from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART


class ContactBodySchemaTests(unittest.TestCase):


    def test_raw_contact_aliases_resolve_sphere_hands_not_wrists(self) -> None:
        assert set(CONTACT_BODY_NAMES_BY_PART["left_hand"]).issubset(PART_ALIASES["lh"])
        assert set(CONTACT_BODY_NAMES_BY_PART["right_hand"]).issubset(PART_ALIASES["rh"])
        assert "left_wrist_yaw_link" not in PART_ALIASES["lh"]
        assert "right_wrist_yaw_link" not in PART_ALIASES["rh"]
