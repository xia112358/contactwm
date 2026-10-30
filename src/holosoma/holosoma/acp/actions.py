"""Existing G1 PD control with continuously sampled 0--20 ms command delay."""
from __future__ import annotations

import torch

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm


class ACPJointPositionAction(JointPositionActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_issued_action = torch.zeros_like(self._processed_actions)
        self.current_issued_action = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions):
        self.previous_issued_action.copy_(self.current_issued_action)
        super().process_actions(actions)
        self.current_issued_action.copy_(self._processed_actions)

    def apply_actions(self):
        delay = self.env.acp_actuation_delay_s[:, None]
        # Average the old/new target over a substep straddling command arrival.
        # 0 ms exactly preserves immediate control; 20 ms uses the previous
        # command for the complete 50-Hz control interval.
        fraction = (((self._substep_idx+1)*self.env.sim_dt-delay)/self.env.sim_dt).clamp(0, 1)
        self._actions_after_delay.copy_(self.previous_issued_action + fraction*(
            self.current_issued_action-self.previous_issued_action))
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.previous_issued_action[ids] = 0
        self.current_issued_action[ids] = 0
