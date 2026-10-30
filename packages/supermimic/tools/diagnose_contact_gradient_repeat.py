"""Compare DMO gradients of repeated attempts without changing either network.

Backward-only force ablation preserves every recorded forward value. This is
not a simulated force ablation and cannot establish which gradient is correct.
"""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

pass
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
cfg = json.loads((args.run / 'config.json').read_text())

from supermimic.lowstep import install
spec = install(cfg['task_manifest'], contact_model='force6')
from supermimic.contact_control import ContactActor, ContactClock, ContactGeometry
from supermimic.contact_dmo import replay_loss
from supermimic.lowstep_observations import HeightForceContext
from supermimic.networks import WorldModel
from supermimic import networks
from supermimic.mdp import normalize_xi
from supermimic.robot import default_joint_pos_list

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
device = 'cuda'
checkpoint = torch.load(cfg['resume'], map_location=device, weights_only=False)
source = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
networks.AMP_DTYPE = torch.bfloat16 if source['args'].get('amp') == 'bf16' else None
actor = ContactActor().to(device)
actor.load_state_dict(checkpoint['actor'])
wm = WorldModel().to(device)
wm.load_state_dict(checkpoint['wm'])
prior = copy.deepcopy(actor).requires_grad_(False)
prior.load_state_dict(torch.load(cfg['prior_checkpoint'], map_location=device, weights_only=False)['actor'])
demo = torch.load(cfg['recording'], map_location=device, weights_only=False)['data']
clock = ContactClock(demo, hold=1, role_aware=True)
geometry = ContactGeometry(demo, device)
geometry.load_state_dict(checkpoint['geometry'])
context = HeightForceContext(spec['tasks'])
default = torch.tensor(default_joint_pos_list(), device=device)
buffer = torch.load(args.run / 'replay.pt', map_location=device, weights_only=False)
rows = (buffer['ptr'] - buffer['size'] + torch.arange(buffer['size'], device=device)) % buffer['T']
data = {k: v[rows] for k, v in buffer['data'].items()}
del buffer
starts = (data['ep_step'] == 0).all(-1).nonzero().flatten().tolist()
blocks = [{k: v[s:e] for k, v in data.items()}
          for s, e in zip(starts, starts[1:] + [len(rows)])]
assert len(blocks) >= 2
blocks = blocks[-2:]
assert all(block['done'].any(0).all() for block in blocks)
params = list(actor.parameters())


def flat(gradients):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(gradients, params)])


def cosine(a, b):
    return float(torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30))


def proposal(gradient):
    # Each proposal starts with identical saved Adam moments and parameters.
    trial = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizer = torch.optim.Adam(trial)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
    offset = 0
    for p in trial:
        p.grad = gradient[offset:offset + p.numel()].reshape_as(p).clone()
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(trial, 1.)
    optimizer.step()
    return torch.cat([(p.detach() - base.detach()).flatten() for p, base in zip(trial, params)])


saved = {}
report = dict(scope='Two real repeated attempts at frozen checkpoint 25; identical actor and initial state. No simulation or training updates. Force ablation changes backward only.',
              checkpoint=cfg['resume'], amp=str(networks.AMP_DTYPE), runs={}, comparisons={})
original_step, original_context = wm.step, wm.next_context


def run(label, block, detach_force=False):
    actions = []

    def step(state, action, xi):
        actions.append(action)
        return original_step(state, action, xi)

    def next_context(*values):
        prediction = original_context(*values)
        return prediction.detach() if detach_force else prediction

    wm.step, wm.next_context = step, next_context
    try:
        parts = {}
        loss, details = replay_loss(actor, wm, clock, geometry, context, normalize_xi, default,
                                   block, prior_actor=prior, completion_mode=cfg['completion_mode'],
                                   component_losses=parts, check=True)
        contact_grads = torch.autograd.grad(parts['contact'], params + actions, retain_graph=True, allow_unused=True)
        contact = flat(contact_grads[:len(params)])
        action_grads = torch.stack([g if g is not None else torch.zeros_like(a)
                                   for g, a in zip(contact_grads[len(params):], actions)])
        total = flat(torch.autograd.grad(loss, params, allow_unused=True))
        delta = proposal(total)
        live = torch.arange(len(actions), device=device)[:, None] < (block['done'].long().argmax(0) + 1)[None]
        action_grads = action_grads.masked_fill(~live[..., None], 0.)
        frames = [t for t in [0, 9, 19, 29, 39, 43, 44, 45, 46] if t < len(actions)]
        result = dict(loss=float(loss.detach()), parts={k: float(v.detach()) for k, v in parts.items()},
                      contact_gradient_norm=float(contact.norm()), total_gradient_norm=float(total.norm()),
                      step_norm=float(delta.norm()), predicted_total_change=float(total @ delta),
                      predicted_contact_change=float(contact @ delta),
                      action_gradient_norm_by_frame={str(t + 1): float(action_grads[t].norm()) for t in frames},
                      action_replay_max_error=details['action_replay_max_error'],
                      forward_state_max_error=details['forward_state_max_error'])
        report['runs'][label] = result
        saved[label] = dict(contact=contact.detach(), total=total.detach(), delta=delta.detach(),
                            actions=action_grads.detach())
        print(label, json.dumps(result), flush=True)
        (args.run / 'gradient-repeat-audit.json').write_text(json.dumps(report, indent=2))
    finally:
        wm.step, wm.next_context = original_step, original_context


for label, weights, detach in [('updated', checkpoint['wm'], False),
                                ('original', source['wm'], False),
                                ('no_force_backward', checkpoint['wm'], True)]:
    wm.load_state_dict(weights)
    for index, block in enumerate(blocks):
        run(f'{label}_{index}', block, detach_force=detach)

wm.load_state_dict(checkpoint['wm'])
run('exact_replay_repeat', blocks[1])
for a, b in [('updated_0', 'updated_1'), ('original_0', 'original_1'),
             ('no_force_backward_0', 'no_force_backward_1'),
             ('updated_1', 'original_1'), ('updated_1', 'no_force_backward_1'),
             ('updated_1', 'exact_replay_repeat')]:
    x, y = saved[a], saved[b]
    report['comparisons'][a + '__' + b] = dict(
        contact_gradient_cosine=cosine(x['contact'], y['contact']),
        contact_gradient_relative_difference=float((x['contact'] - y['contact']).norm() / x['contact'].norm()),
        total_gradient_cosine=cosine(x['total'], y['total']),
        adam_step_cosine=cosine(x['delta'], y['delta']),
        x_step_contact_change_on_y=float(y['contact'] @ x['delta']),
        y_step_contact_change_on_x=float(x['contact'] @ y['delta']))
assert all(torch.equal(v, checkpoint['actor'][k]) for k, v in actor.state_dict().items())
assert all(torch.equal(v, checkpoint['wm'][k]) for k, v in wm.state_dict().items())
report['networks_unchanged'] = True
(args.run / 'gradient-repeat-audit.json').write_text(json.dumps(report, indent=2))
print('COMPARISONS', json.dumps(report['comparisons']), flush=True)
