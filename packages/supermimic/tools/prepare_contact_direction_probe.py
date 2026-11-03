"""Prepare diagnostic alternatives to one saved Adam step; no training update."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

from supermimic.contact_control import ContactActor

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
proposal = torch.load(a.run / 'diagnostic-adam-proposal.pt', map_location='cpu', weights_only=False)
checkpoint = torch.load(proposal['source_checkpoint'], map_location='cpu', weights_only=False)
actor = ContactActor()
actor.load_state_dict(proposal['base_actor'])
parameters = list(actor.parameters())
gradient = proposal['gradient']
delta = proposal['delta']
optimizer = torch.optim.Adam(parameters)
optimizer.load_state_dict(copy.deepcopy(checkpoint['actor_optimizer']))
for group in optimizer.param_groups:
    group['betas'] = (0., group['betas'][1])
offset = 0
for parameter in parameters:
    parameter.grad = gradient[offset:offset + parameter.numel()].reshape_as(parameter).clone()
    offset += parameter.numel()
torch.nn.utils.clip_grad_norm_(parameters, 1.)
optimizer.step()
base = proposal['base_actor']
names = [name for name, _ in actor.named_parameters()]
fresh = torch.cat([(actor.state_dict()[name] - base[name]).flatten() for name in names])
directions = {'adam': delta, 'no_momentum': fresh * (delta.norm() / fresh.norm()),
              'negative_gradient': -gradient * (delta.norm() / gradient.norm())}
variants = {}
statistics = {}
for label, direction in directions.items():
    state = {key: value.clone() for key, value in base.items()}
    offset = 0
    for name in names:
        n = state[name].numel()
        state[name] += direction[offset:offset + n].reshape_as(state[name])
        offset += n
    variants[label] = state
    # Report actual fp32 parameter changes after rounding.
    realized = torch.cat([(state[name] - base[name]).flatten() for name in names])
    statistics[label] = dict(step_norm=float(realized.norm()),
                            cosine_negative_gradient=float(torch.dot(realized, -gradient) / (realized.norm() * gradient.norm())),
                            predicted_total_change=float(torch.dot(gradient, realized)))
plan = []
rng = random.Random(17)
conditions = [('base', 'adam', 0.), ('adam_full', 'adam', 1.), ('adam_quarter', 'adam', .25),
              ('no_momentum', 'no_momentum', 1.), ('negative_gradient', 'negative_gradient', 1.)]
for repeat in range(4):
    ordered = conditions.copy()
    rng.shuffle(ordered)
    for name, variant, scale in ordered:
        plan.append(dict(label=f'{name}_{repeat}', variant=variant, scale=scale))
proposal.update(trial_actors=variants, probes=plan, direction_statistics=statistics)
torch.save(proposal, a.run / 'diagnostic-direction-proposals.pt')
(a.run / 'direction-proposal-plan.json').write_text(json.dumps(dict(
    scope='Controlled diagnostic proposals, not accepted training weights. Equal parameter norms do not imply equal action changes. Four shuffled repeats of five conditions.',
    direction_statistics=statistics, probes=plan), indent=2))
print(json.dumps(statistics, indent=2))
