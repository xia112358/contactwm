import importlib.util
from pathlib import Path
import sys
import pytest

from generator.research.archived_experiments import terrain_scaling as module


def test_height_scaling_keeps_ground_xy_and_topology():
    text='v 1 2 0\nv 3 4 0.70449438\nf 1 2 1\n'
    result=module.terrain_text(text,.975).splitlines()
    assert result[0]=='v 1 2 0.0000000000'
    assert result[1].split()[1:3]==['3','4']
    assert float(result[1].split()[3])==pytest.approx(.70449438*.975)
    assert result[2]=='f 1 2 1'


def test_outside_declared_experiment_rejected():
    with pytest.raises(ValueError): module.terrain_text('v 0 0 1',2.)
