import json
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).absolute().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT/'runtime/current/archives/preserved_inputs_20261010/dropout207_training_v1/corpus.pt').exists(),
                               reason='Requires explicitly built local 207-trajectory training corpus')




def test_preserved_dataset_split_and_asset_identity():
    from somaforge_core.robot_assets import decode_robot_asset_json
    root = ROOT/'runtime/current/archives/preserved_inputs_20261010/dropout207_training_v1'
    payload = torch.load(root/'corpus.pt', weights_only=False, map_location='cpu')
    decode_robot_asset_json(payload['robot_asset_json'], context='preserved regression corpus')
    ownership = {}
    for row in payload['records']:
        # This is a preserved historical experiment, not newly certified data.
        assert row['event_contract'] == payload['event_contract']
        assert ownership.setdefault(row['sequence'],row['split']) == row['split']
    assert len(ownership) == 207
    assert payload['frozen_predictor'] is None and payload['frozen_infiller'] is None
