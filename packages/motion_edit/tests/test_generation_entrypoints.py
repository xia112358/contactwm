import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from motion_edit import cli
from motion_edit.contact.plans import ContactEditPlan


@pytest.mark.parametrize('command', ['generate-augmentations', 'generate-lte-augmentation',
                                     'batch-generate-lte-augmentations'])
def test_retired_commands_are_not_registered(command):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command])
