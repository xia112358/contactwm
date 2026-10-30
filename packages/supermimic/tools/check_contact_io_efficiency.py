"""CPU-only checkpoint/replay equivalence and read benchmark; no actor updates."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch

pass
from supermimic.contact_io import load_checkpoint_cpu
from supermimic.contact_dagger import BalancedReplay, _load_replay_groups


def check():
    torch.manual_seed(7)
    actor = torch.nn.Sequential(torch.nn.Linear(11, 13), torch.nn.Tanh(), torch.nn.Linear(13, 3))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    for parameter in actor.parameters():
        optimizer.state[parameter] = dict(step=torch.tensor(4.), exp_avg=torch.randn_like(parameter),
                                          exp_avg_sq=torch.rand_like(parameter))
    sample = torch.randn(9, 11)
    temporary_root = runtime_path('tmp')
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='contact-io-check-', dir=temporary_root) as directory:
        root = Path(directory)
        for modern in (True, False):
            path = root / f'model-{modern}.pt'
            torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict()), path,
                       _use_new_zipfile_serialization=modern)
            saved = load_checkpoint_cpu(path)
            clone = torch.nn.Sequential(torch.nn.Linear(11, 13), torch.nn.Tanh(), torch.nn.Linear(13, 3))
            clone.load_state_dict(saved['actor'])
            torch.testing.assert_close(actor(sample), clone(sample), rtol=0, atol=0)
            restored = torch.optim.Adam(clone.parameters(), lr=1.)
            restored.load_state_dict(saved['optimizer'])
            assert restored.param_groups[0]['lr'] == 1e-5
            for p, q in zip(actor.parameters(), clone.parameters()):
                for key, value in optimizer.state[p].items():
                    assert restored.state[q][key].device == q.device
                    torch.testing.assert_close(value, restored.state[q][key], rtol=0, atol=0)
            # mmap is private: mutating an in-memory optimizer state cannot
            # modify the trusted checkpoint on disk.
            saved['actor']['0.weight'].zero_()
            torch.testing.assert_close(load_checkpoint_cpu(path)['actor']['0.weight'],
                                       actor.state_dict()['0.weight'], rtol=0, atol=0)
        old = {}
        paths = []
        for round_id in range(7):
            data = {i: {key: torch.randn(i+round_id+1, width) for key,width in
                        [('history', 15), ('condition', 7), ('action', 3)]} for i in range(3)}
            path = root / f'rollout_{round_id:04d}.pt'
            torch.save(dict(data=data), path);paths.append(path)
            for i,row in data.items():
                old[i] = {key: torch.cat((old[i][key], value)) for key,value in row.items()} if i in old else row
        new = _load_replay_groups(paths)
        for i,row in old.items():
            for key,value in row.items():torch.testing.assert_close(new[i][key], value, rtol=0, atol=0)
        old_replay, new_replay = BalancedReplay(old), BalancedReplay(new)
        torch.manual_seed(11);sample_old = old_replay.sample(101)
        torch.manual_seed(11);sample_new = new_replay.sample(101)
        for key in sample_old:torch.testing.assert_close(sample_old[key], sample_new[key], rtol=0, atol=0)
        assert _load_replay_groups([]) == {}
    return dict(modern_and_legacy_checkpoint=True, model_forward_exact=True, optimizer_state_exact=True,
                disk_snapshot_immutable=True, replay_order_and_seeded_sampling_exact=True)


def benchmark(paths):
    reports = []
    for path in paths:
        keys = None;timings = {'eager': [], 'mapped': []};expected = None;bytes_used = 0
        for repeat in range(8):
            name = 'eager' if repeat%2 == 0 else 'mapped'
            start = time.perf_counter()
            saved = (torch.load(path, map_location='cpu', weights_only=False) if name == 'eager'
                     else load_checkpoint_cpu(path))
            keys = [key for key in ('actor','pi','wm','geometry') if key in saved]
            # Match initialization: touch only the state dictionaries consumed
            # by models, leaving unused optimizer/replay tensors out.
            states = {key: {name:value.clone() if isinstance(value, torch.Tensor) else value
                            for name,value in saved[key].items()} for key in keys}
            elapsed = time.perf_counter()-start
            if repeat >= 2:timings[name].append(elapsed)
            if expected is None:
                expected = states
                bytes_used = sum(value.numel()*value.element_size() for state in states.values()
                                 for value in state.values() if isinstance(value, torch.Tensor))
            else:
                for key,state in states.items():
                    for name,value in state.items():
                        if isinstance(value,torch.Tensor):torch.testing.assert_close(value,expected[key][name],rtol=0,atol=0)
            del saved,states
        reports.append(dict(path=str(path), checkpoint_mb=path.stat().st_size/2**20,
                            selected_model_mb=bytes_used/2**20, selected_keys=keys,
                            eager_median_ms=statistics.median(timings['eager'])*1000,
                            mapped_median_ms=statistics.median(timings['mapped'])*1000,
                            repeats_per_mode=3, selected_tensors_exact=True))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = dict(checks=check(), benchmarks=benchmark(args.checkpoint), actor_updates_performed=0,
                  scope='CPU warm-cache reads and selected-state copies; excludes simulator and GPU initialization')
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
