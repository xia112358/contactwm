"""Conservative boundary waits: both motion and contact evidence must be quiet."""
from supermimic.paths import runtime_path, runtime_root, workspace_root, module_root
import torch


def boundary_wait_bounds(data, begin, end, minimum=10, context=2):
    s = data['s'][begin:end]
    quiet = ((s[:, 7:10].norm(dim=-1) < .05)
             & (s[:, 10:13].norm(dim=-1) < .15)
             & (s[:, 42:71].square().mean(-1).sqrt() < .2))
    active = data['active'][begin:end].bool()
    known = data['known'][begin:end].bool()
    surface = data['surface'][begin:end]
    # Unknown evidence is never proof of no contact conversion.
    quiet &= known.all(-1)
    stable = (active[1:] == active[:-1]).all(-1)
    stable &= ((surface[1:] == surface[:-1]) | ~active[1:]).all(-1)
    # Check the endpoint of each transition too, including the last frame.
    if 's1' in data:
        t = data['s1'][begin:end]
        quiet &= ((t[:, 7:10].norm(dim=-1) < .05)
                  & (t[:, 10:13].norm(dim=-1) < .15)
                  & (t[:, 42:71].square().mean(-1).sqrt() < .2))
    n = end-begin
    head = 0
    while head < n and bool(quiet[head]) and (head == 0 or bool(stable[head-1])):
        head += 1
    tail = n
    while tail > 0 and bool(quiet[tail-1]) and (tail == n or bool(stable[tail-1])):
        tail -= 1
    if head == n and n >= minimum:
        return begin, begin, dict(head_wait=n, tail_wait=n, entire_wait=True)
    start = begin + max(0, head-context) if head >= minimum else begin
    stop = begin + min(n, tail+context) if n-tail >= minimum else end
    return start, stop, dict(head_wait=head, tail_wait=n-tail, entire_wait=False)
