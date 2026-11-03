"""Verify original network identity, input isolation, and differentiable recurrence."""
import sys
from pathlib import Path

import torch
from supermimic import networks, mdp as mdp_module
from supermimic.lowstep import install, checkpoint_metadata, validate_checkpoint

original_wm = networks.WorldModel
original_mdp = mdp_module.MDP
original_normalize = mdp_module.normalize_xi
spec = install('data/lowclimb_boxes/task.json', contact_model='original')
assert networks.WorldModel is original_wm
assert mdp_module.normalize_xi is original_normalize
assert (mdp_module.OBS_DIM, mdp_module.XI_DIM) == (99, 33)
from supermimic.buffer import FIELDS
assert 'contact_next' not in FIELDS
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list

kin = G1Kinematics('cpu')
motion = MotionLib.from_dir(spec['motion_dir'], kin, 'cpu')
default = torch.tensor(default_joint_pos_list())
mdp = mdp_module.MDP(kin, motion, default)
base = original_mdp(kin, motion, default)
frames = motion.clip_start.clone()
state = motion.ref_state(frames)
xi = torch.zeros(len(frames), 33)
xi[:, 0] = 1
actions = torch.zeros(len(frames), 29)
torch.testing.assert_close(mdp.obs(state, actions, xi, frames), base.obs(state, actions, xi, frames))
torch.testing.assert_close(mdp.critic_features(state, frames, xi), base.critic_features(state, frames, xi))
wm = networks.WorldModel(hidden=(64, 64))
assert wm.in_dim == 133 and wm.net[-1].out_features == 70
assert not hasattr(wm, 'contact_head') and not hasattr(wm, 'next_context')
assert wm.inputs(state, actions, xi).shape == (40, 133)
# Exercise multiple imagined steps and their gradients without a contact head.
torch.nn.init.normal_(wm.net[-1].weight, std=.001)
action = actions.clone().requires_grad_(True)
predicted = state
for _ in range(4):
    predicted = wm.step(predicted, action, original_normalize(xi))
sum(x.square().mean() for x in predicted).backward()
assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
metadata = checkpoint_metadata(spec)
validate_checkpoint(metadata, metadata)
try:
    validate_checkpoint({**metadata, 'observation_schema': 'local_heightmap_11x11_force6x3_heading_v1'}, metadata)
except ValueError:
    pass
else:
    raise AssertionError('Must reject force6 checkpoints')
print('PASS: unmodified original WorldModel; WM 133->70, obs 99, xi 33; original critic features; '
      'no terrain/contact/task-ID network input; four-step differentiable recurrence; checkpoint isolation')
