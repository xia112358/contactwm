"""Match contact episodes by partial order and support dependencies.

Reference frame numbers identify authored actions/blocks only. They never set
observation deadlines. Independent limbs may exchange event order. Observed
episodes retain their original Newton sample clock; poses/labels are untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

SCHEMA = "contact_partial_order_acceptance_v1"


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def audit_contact_order(contract, mask, surface):
    """Maximum consistent matching, with unmatched requirements reported.

    Establish means achieving a contact state, not an implicit release/re-touch.
    Explicit release requirements separate repeated touchdown episodes. Keep
    requirements tie the supporting episode to dependent acquisitions/releases,
    including intervening keep-only actions. No global ordering of independent
    limbs, reference-frame windows, or prescribed action durations is imposed.
    Existing stable-evidence/dropout policy is used solely for sensor evidence.
    """
    mask, surface = np.asarray(mask), np.asarray(surface)
    if (mask.ndim != 2 or mask.shape[1] != 6 or surface.shape != mask.shape
            or not len(mask) or not np.isin(mask, [0, 1]).all()):
        raise ValueError("Expected nonempty binary six-part contact evidence")
    mask = mask.astype(bool)
    if np.any(mask & (surface < 0)):
        raise ValueError("Missing actual contact surface")
    stable = int(contract["policy"]["stable_frames"])
    dropout = int(contract["policy"]["max_dropout_frames"])
    if stable < 1 or dropout < 0:
        raise ValueError("Invalid contact evidence policy")
    actions = contract["actions"]
    if not actions or not any(a["requirements"] for a in actions):
        raise ValueError("Empty action contract")
    blocks = contract["retained_blocks"]
    block_ids = []
    for a in actions:
        matches = [i for i, (lo, hi) in enumerate(blocks) if lo <= a["start"] <= a["end"] < hi]
        if len(matches) != 1:
            raise ValueError("Action must belong to exactly one retained block")
        block_ids.append(matches[0])
    if any(b < a for a, b in zip(block_ids, block_ids[1:])):
        raise ValueError("Unordered action blocks")

    cache = {}
    def episodes(part, face, release=False):
        key = (part, face, release)
        if key not in cache:
            raw = ~mask[:, part] if release else mask[:, part] & (surface[:, part] == face)
            continuous = raw.copy()
            if not release:
                for lo, hi in _runs(~raw):
                    if (lo > 0 and hi < len(raw) and hi-lo <= dropout
                            and not mask[lo:hi, part].any()):
                        continuous[lo:hi] = True
            runs = []
            for lo, hi in _runs(continuous):
                evidence = [(a+lo, b+lo) for a, b in _runs(raw[lo:hi]) if b-a >= stable]
                if evidence:
                    for begin, end in evidence:
                        runs.append((int(lo), int(hi), int(begin), int(end-stable)))
            cache[key] = runs
        return cache[key]

    nodes, checks, action_nodes = [], [], [[] for _ in actions]
    owners = {}
    def node(action, part, face, kind):
        n = len(nodes)
        nodes.append(dict(action=action, part=part, surface=face, kind=kind,
                          episodes=episodes(part, face, kind == "release")))
        return n

    for i, a in enumerate(actions):
        if i == 0 or block_ids[i] != block_ids[i-1]:
            owners = {}
        seen = set()
        for req in a["requirements"]:
            part, face, kind = int(req["part"]), int(req["surface"]), req["kind"]
            if not 0 <= part < 6 or part in seen or kind not in {"keep", "establish", "release"}:
                raise ValueError("Invalid/conflicting action requirements")
            if (kind == "release" and face != -1) or (kind != "release" and face < 0):
                raise ValueError("Invalid required surface")
            seen.add(part)
            row = dict(action=a["id"], action_index=i, kind=kind, part=part, surface=face)
            if kind == "keep":
                if part not in owners:
                    owners[part] = node(i, part, face, "initial_support")
                owner = owners[part]
                if nodes[owner]["surface"] != face or nodes[owner]["kind"] == "release":
                    raise ValueError("Keep requires an established matching surface")
                row["owner"] = owner
            else:
                owners[part] = node(i, part, face, kind)
                row["node"] = owners[part]
                action_nodes[i].append(owners[part])
            checks.append(row)

    for row in checks:
        if row["kind"] != "keep":
            continue
        i = row["action_index"]
        targets = list(action_nodes[i])
        if not targets:
            # A keep-only phase confirms the preceding milestone. It does not
            # silently extend support into a later action that never requested
            # it. A subsequent explicit keep adds its own dependency.
            j = i-1
            while j >= 0 and block_ids[j] == block_ids[i]:
                if action_nodes[j]:
                    targets.extend(action_nodes[j]); break
                j -= 1
        row["targets"] = sorted(set(targets) - {row["owner"]})

    cost, constraints, upper, limits = [], [], [], []
    def variable(weight=0., limit=1):
        index = len(cost); cost.append(weight); limits.append(limit); return index
    def bound(terms, limit):
        constraints.append(terms); upper.append(limit)
    for n in nodes:
        n["variables"] = [variable() for _ in n["episodes"]]
        n["time"] = variable(limit=len(mask)-1)
        if n["variables"]:
            bound({v: 1 for v in n["variables"]}, 1)
        for v, episode in zip(n["variables"], n["episodes"]):
            bound({n["time"]: -1, v: len(mask)}, len(mask)-episode[2])
            bound({n["time"]: 1, v: len(mask)}, len(mask)+episode[3])
    # All selected events on one limb must respect its authored order even
    # when intervening events are unmatched. Independent limbs have no edge.
    for j, later in enumerate(nodes):
        for earlier in nodes[:j]:
            if earlier["part"] != later["part"]:
                continue
            bound({earlier["time"]: 1, later["time"]: -1,
                   **{v: len(mask) for v in earlier["variables"]+later["variables"]}}, 2*len(mask))
    for row in checks:
        if row["kind"] != "keep":
            row["variables"] = nodes[row["node"]]["variables"]
            for v in row["variables"]:
                cost[v] = -1.
            continue
        y = variable(-1.)
        row["variables"] = [y]
        owner = nodes[row["owner"]]
        bound({y: 1, **{v: -1 for v in owner["variables"]}}, 0)
        for target_id in row["targets"]:
            target = nodes[target_id]
            bound({y: 1, **{v: -1 for v in target["variables"]}}, 0)
            bound({owner["time"]: 1, target["time"]: -1, y: len(mask)}, len(mask))
            for u, a in zip(owner["variables"], owner["episodes"]):
                # The same support episode must cover the dependent milestone.
                # Subsequent stability samples belong to the new contact;
                # they do not prescribe extra duration for the old support.
                bound({target["time"]: -1, y: len(mask), u: len(mask)}, 2*len(mask)-a[0])
                # Half-open episodes may meet at a handover with no sample
                # where both supports are absent; overlap is not mandatory.
                bound({target["time"]: 1, y: len(mask), u: len(mask)}, 2*len(mask)+a[1])
    if not cost:
        # Missing observed episodes still yields an explicit failed report.
        solution = np.zeros(0)
    else:
        # Deterministic preference for earlier matches cannot outweigh even
        # one fulfilled requirement (total tie cost is strictly below 0.1).
        for n in nodes:
            cost[n["time"]] = .1/((len(mask)+1)*max(1, len(nodes)))
        rr, cc, vv = [], [], []
        for i, terms in enumerate(constraints):
            for j, value in terms.items():
                rr.append(i); cc.append(j); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(constraints), len(cost))).tocsc()
        result = milp(np.asarray(cost), integrality=np.ones(len(cost)), bounds=Bounds(0, limits),
                      constraints=LinearConstraint(matrix, -np.inf, upper),
                      options={"mip_rel_gap": 0.})
        if not result.success:
            raise RuntimeError(f"Contact order matching failed: {result.message}")
        solution = result.x
    assignments = {}
    for i, n in enumerate(nodes):
        episode = next((e for v, e in zip(n["variables"], n["episodes"]) if solution[v] > .5), None)
        t = int(round(solution[n["time"]]))
        assignments[i] = (dict(interval=list(episode[:2]), matched_frame=t,
                              stable_evidence=[t, t+stable]) if episode else None)
    output = []
    for row in checks:
        passed = any(solution[v] > .5 for v in row["variables"])
        entry = {k: row[k] for k in ("action", "kind", "part", "surface")}
        entry["passed"] = bool(passed)
        if row["kind"] == "keep":
            entry["support_episode"] = assignments[row["owner"]]
            entry["dependent_events"] = [dict(action=actions[nodes[t]["action"]]["id"],
                part=nodes[t]["part"], episode=assignments[t]) for t in row["targets"]]
            entry["reason"] = "support_dependency_satisfied" if passed else "missing_or_interrupted_support_dependency"
        else:
            entry["episode"] = assignments[row["node"]]
            entry["reason"] = "ordered_episode_matched" if passed else "missing_episode_or_same_part_order_conflict"
        output.append(entry)
    failures = [r for r in output if not r["passed"]]
    return dict(schema=SCHEMA, passed=not failures, check_count=len(output),
        failed_checks=len(failures), checks=output, failures=failures,
        matching="maximum consistent partial-order episode matching",
        episode_format="interval and stable_evidence have exclusive ends; matched_frame is an observed sample",
        reference_frame_deadlines=False, independent_parts_may_reorder=True,
        evidence_policy=dict(stable_frames=stable, max_dropout_frames=dropout),
        scope="contact event order and support dependencies; action durations and dynamics unverified")
