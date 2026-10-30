"""Predictor endpoint tolerances and loss scales, independent of observations.

These values never define Newton contact activation or modify solver margins.
Historical checkpoints keep their recorded acceptance configuration.
"""

PENETRATION_ACCEPTANCE_M = 0.03
LEGACY_PENETRATION_ACCEPTANCE_M = 0.001
ENDPOINT_POSITION_TOLERANCE_M = 0.04
JOINT_EXCESS_RAD = 0.15
JOINT_LIMIT_SCALE_RAD = 0.10
PENETRATION_LOSS_SCALE_M = 0.005

