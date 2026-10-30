"""Summarize successful completion times separately from censored failures."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads((args.run / 'config.json').read_text())
    buf = torch.load(args.run / 'replay.pt', map_location='cpu', weights_only=False)
    rows = (buf['ptr'] - buf['size'] + torch.arange(buf['size'])) % buf['T']
    data = {k: v[rows] for k, v in buf['data'].items()}
    del buf
    starts = (data['ep_step'] == 0).all(-1).nonzero().flatten().tolist()
    complete = [(s, e) for s, e in zip(starts, starts[1:] + [len(rows)])
                if data['done'][s:e].any(0).all()]
    assert complete
    start, end = complete[-1]
    data = {k: v[start:end] for k, v in data.items()}
    stop = data['done'].long().argmax(0)
    success = ~data['term'][stop, torch.arange(len(stop))]
    segments = []
    for g in data['goal_index'][0].unique().tolist():
        mask = data['goal_index'][0] == g
        frames = (stop[mask & success] + 1).float()
        values, counts = torch.unique(frames.long(), return_counts=True)
        budget = cfg['segment_budgets'][g]
        failed = mask & ~success
        segments.append(dict(segment=g+1, attempts=int(mask.sum()), successes=len(frames),
            demo_frames=budget-cfg['segment_grace_steps'], budget=budget,
            quantiles=dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                torch.quantile(frames, torch.tensor([0., .1, .5, .9, 1.])).tolist())) if len(frames) else None,
            histogram=dict(zip(values.tolist(), counts.tolist())),
            timeouts=int((failed & (stop+1 == budget)).sum()),
            early_failures=int((failed & (stop+1 < budget)).sum())))
    report = dict(checkpoint=cfg['resume'], scope='Latest complete frozen batch; successful task completion frames only; failures separate',
                  dt_seconds=.02, segments=segments)
    (args.run / 'duration-distribution.json').write_text(json.dumps(report, indent=2))
    for row in segments:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
