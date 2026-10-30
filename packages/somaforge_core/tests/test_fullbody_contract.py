import numpy as np
import pytest
from somaforge_core.motion_trajectory import FullBodyTrajectory
from somaforge_core import G1_29DOF_JOINT_ORDER, encode_robot_asset_json


def test_fullbody_contract_requires_canonical_36d_qpos() -> None:
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    trajectory = FullBodyTrajectory(qpos, 30.0, encode_robot_asset_json())
    assert trajectory.joint_names == G1_29DOF_JOINT_ORDER


def test_fullbody_contract_rejects_sparse_state() -> None:
    qpos = np.zeros((2, 63), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        FullBodyTrajectory(qpos, 30.0, encode_robot_asset_json())
