from __future__ import annotations

from pathlib import Path

import numpy as np

from motion_edit.cli import build_parser
from somaforge_core import encode_robot_asset_json
from somaforge_core.contact_schema import (
    CONTACT_FORCE_PART_ORDER,
    encode_contact_force_provenance,
    newton_contact_provenance,
)


def test_force_retarget_cli_is_retired() -> None:
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args(["force-retarget"])
