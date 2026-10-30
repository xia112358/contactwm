import dataclasses

import holosoma.config_values.simulator
from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg


isaacsim = dataclasses.replace(
    holosoma.config_values.simulator.isaaclab3_newton,
    config=dataclasses.replace(
        holosoma.config_values.simulator.isaaclab3_newton.config,
        bridge=BridgeConfig(enabled=False),
        virtual_gantry=VirtualGantryCfg(enabled=False),
    ),
)

isaaclab3_newton = isaacsim

DEFAULTS = {
    "isaacsim": isaacsim,
    "isaaclab3-newton": isaaclab3_newton,
}
