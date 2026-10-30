"""Diagnostic, CUDA-graph-compatible audit of every native constraint solve.

Install before the first ``sim.step`` (before its CUDA graph is captured). This
does not change contacts, forces, integration, or policy inputs. Counters include
any native solves performed during graph warmup/capture as well as replay.
"""
from __future__ import annotations

import functools
import importlib
import math

import warp as wp
from mujoco_warp._src.types import ContactType


REASONS = {
    1: 'contact_count_out_of_capacity',
    2: 'contact_world_out_of_range',
    3: 'nonfinite_contact_distance_or_margin',
    4: 'nonfinite_active_contact_geometry',
    5: 'nonorthonormal_active_contact_frame',
    6: 'invalid_active_contact_dimension',
    7: 'active_constraint_not_allocated_in_world',
    8: 'constraint_count_out_of_capacity',
    9: 'nonfinite_live_constraint',
    10: 'nonfinite_native_state_after_solve',
    11: 'native_overflow_flag',
}
COUNTERS = ('native_solves', 'contact_rows', 'active_contact_rows',
            'max_contact_count', 'max_world_constraints', 'live_constraint_rows',
            'error_events')


@wp.func
def _fail(code: int, world: int, row: int, column: int, value: int,
          first: wp.array[int], counters: wp.array[wp.int64]):
    wp.atomic_add(counters, 6, wp.int64(1))
    won = wp.atomic_cas(first, 0, 0, code) == 0
    if won:
        first[1] = world
        first[2] = row
        first[3] = int(counters[0])
        first[4] = column
        first[5] = value
    return won


@wp.kernel
def _begin(nacon: wp.array[int], capacity: int, first: wp.array[int],
           counters: wp.array[wp.int64]):
    wp.atomic_add(counters, 0, wp.int64(1))
    wp.atomic_max(counters, 3, wp.int64(nacon[0]))
    if nacon[0] < 0 or nacon[0] > capacity:
        _fail(1, -1, -1, -1, nacon[0], first, counters)


@wp.kernel
def _contacts(nacon: wp.array[int], world: wp.array[int], kind: wp.array[int],
              dist: wp.array[float], margin: wp.array[float],
              pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
              dim: wp.array[int], address: wp.array2d[int], nefc: wp.array[int],
              efc_capacity: int, cone: int, tolerance: float,
              first: wp.array[int], detail: wp.array[float],
              counters: wp.array[wp.int64]):
    i = wp.tid()
    if i >= nacon[0]:
        return
    wp.atomic_add(counters, 1, wp.int64(1))
    w = world[i]
    if w < 0 or w >= nefc.shape[0]:
        _fail(2, w, i, -1, w, first, counters)
        return
    if not (kind[i] & int(ContactType.CONSTRAINT)):
        return
    if not wp.isfinite(dist[i]) or not wp.isfinite(margin[i]):
        if _fail(3, w, i, -1, 0, first, counters):
            detail[0] = dist[i]
            detail[1] = margin[i]
        return
    if dist[i] >= margin[i]:
        return
    wp.atomic_add(counters, 2, wp.int64(1))
    f = frame[i]
    p = pos[i]
    finite = True
    for j in range(3):
        finite = finite and wp.isfinite(p[j])
        for k in range(3):
            finite = finite and wp.isfinite(f[j, k])
    n = wp.vec3(f[0, 0], f[0, 1], f[0, 2])
    t = wp.vec3(f[1, 0], f[1, 1], f[1, 2])
    u = wp.vec3(f[2, 0], f[2, 1], f[2, 2])
    orthonormal = (wp.abs(wp.dot(n, n)-1.0) <= tolerance and
                   wp.abs(wp.dot(t, t)-1.0) <= tolerance and
                   wp.abs(wp.dot(u, u)-1.0) <= tolerance and
                   wp.abs(wp.dot(n, t)) <= tolerance and
                   wp.abs(wp.dot(n, u)) <= tolerance and
                   wp.abs(wp.dot(t, u)) <= tolerance and
                   wp.abs(wp.dot(wp.cross(n, t), u)-1.0) <= tolerance)
    code = int(0)
    if not finite:
        code = 4
    elif not orthonormal:
        code = 5
    if code != 0:
        if _fail(code, w, i, -1, 0, first, counters):
            detail[0] = dist[i]
            detail[1] = margin[i]
            for j in range(3):
                detail[2+j] = p[j]
                for k in range(3):
                    detail[5+3*j+k] = f[j, k]
    d = dim[i]
    if d != 1 and d != 3 and d != 4 and d != 6:
        _fail(6, w, i, -1, d, first, counters)
        return
    rows = d
    if cone == 0 and d > 1:
        rows = 2*(d-1)
    if rows > address.shape[1]:
        _fail(7, w, i, -1, rows, first, counters)
        return
    for j in range(rows):
        a = address[i, j]
        if a < 0 or a >= nefc[w] or a >= efc_capacity:
            _fail(7, w, i, j, a, first, counters)


@wp.kernel
def _worlds(nefc: wp.array[int], overflow: wp.array[int],
            qp: wp.array2d[float], qv: wp.array2d[float], qa: wp.array2d[float],
            capacity: int, first: wp.array[int], counters: wp.array[wp.int64]):
    w = wp.tid()
    wp.atomic_max(counters, 4, wp.int64(nefc[w]))
    if nefc[w] < 0 or nefc[w] > capacity:
        _fail(8, w, -1, -1, nefc[w], first, counters)
    if overflow[w] != 0:
        _fail(11, w, -1, -1, overflow[w], first, counters)
    for j in range(qp.shape[1]):
        if not wp.isfinite(qp[w, j]):
            _fail(10, w, j, 0, 0, first, counters)
    for j in range(qv.shape[1]):
        if not wp.isfinite(qv[w, j]):
            _fail(10, w, j, 1, 0, first, counters)
        if not wp.isfinite(qa[w, j]):
            _fail(10, w, j, 2, 0, first, counters)


@wp.kernel
def _constraints(nefc: wp.array[int], force: wp.array2d[float],
                 D: wp.array2d[float], pos: wp.array2d[float], aref: wp.array2d[float],
                 first: wp.array[int], detail: wp.array[float],
                 counters: wp.array[wp.int64]):
    w, j = wp.tid()
    if j >= nefc[w]:
        return
    wp.atomic_add(counters, 5, wp.int64(1))
    if (not wp.isfinite(force[w, j]) or not wp.isfinite(D[w, j]) or
            not wp.isfinite(pos[w, j]) or not wp.isfinite(aref[w, j])):
        if _fail(9, w, j, -1, 0, first, counters):
            detail[0] = force[w, j]
            detail[1] = D[w, j]
            detail[2] = pos[w, j]
            detail[3] = aref[w, j]


class ContactSolverAudit:
    """One-process diagnostic hook; instantiate before native graph capture.

    ``check()`` synchronizes a small first-failure buffer and raises on any
    recorded anomaly. ``report()`` returns cumulative JSON-compatible evidence.
    Errors persist, so failures cannot disappear between control-step reads.
    Use a fresh process to remove this diagnostic from already captured graphs.
    """
    def __init__(self, manager, *, frame_tolerance=1e-4):
        self.manager = manager
        self.solver = manager._solver
        self.data = d = self.solver.mjw_data
        self.device = d.qpos.device
        self.cone = int(self.solver.mjw_model.opt.cone)
        if self.cone not in (0, 1):
            raise ValueError('Unsupported native friction cone')
        if not 0 < frame_tolerance <= 1e-3:
            raise ValueError('Expected a strict positive contact frame tolerance')
        self.frame_tolerance = float(frame_tolerance)
        self.first = wp.zeros(6, dtype=int, device=self.device)
        self.detail = wp.zeros(14, device=self.device)
        self.counters = wp.zeros(len(COUNTERS), dtype=wp.int64, device=self.device)
        forward = importlib.import_module('mujoco_warp._src.forward')
        original = forward.solver.solve
        if getattr(original, '_supermimic_contact_audit', False):
            raise RuntimeError('Only one ContactSolverAudit may own a process')

        @functools.wraps(original)
        def solve(model, data):
            result = original(model, data)
            if data is self.data:
                self.launch()
            return result

        solve._supermimic_contact_audit = True
        forward.solver.solve = solve

    def launch(self):
        """Insert audit kernels on the current native stream (capture-safe)."""
        d = self.data
        c = d.contact
        wp.launch(_begin, dim=1, inputs=[d.nacon, c.dist.shape[0], self.first, self.counters], device=self.device)
        wp.launch(_contacts, dim=c.dist.shape[0], inputs=[d.nacon, c.worldid, c.type,
            c.dist, c.includemargin, c.pos, c.frame, c.dim, c.efc_address, d.nefc,
            d.efc.force.shape[1], self.cone, self.frame_tolerance,
            self.first, self.detail, self.counters], device=self.device)
        wp.launch(_worlds, dim=d.qpos.shape[0], inputs=[d.nefc, d.overflow,
            d.qpos, d.qvel, d.qacc, d.efc.force.shape[1], self.first, self.counters], device=self.device)
        wp.launch(_constraints, dim=d.efc.force.shape, inputs=[d.nefc,
            d.efc.force, d.efc.D, d.efc.pos, d.efc.aref,
            self.first, self.detail, self.counters], device=self.device)

    def report(self):
        first = self.first.numpy().tolist()
        counts = dict(zip(COUNTERS, map(int, self.counters.numpy().tolist())))
        cmax = int(self.data.contact.dist.shape[0])
        emax = int(self.data.efc.force.shape[1])
        failure = None
        if first[0]:
            # Preserve the kind/sign of invalid values without emitting JSON's
            # non-standard NaN/Infinity tokens. A failure report must itself be
            # writable with allow_nan=False, including the failure we detect.
            detail = [value if math.isfinite(value) else
                      'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'
                      for value in self.detail.numpy().tolist()]
            failure = dict(reason=REASONS[first[0]], reason_code=first[0],
                           world=first[1], row=first[2], native_solve=first[3],
                           column=first[4], value=first[5], detail=detail,
                           nonfinite_detail_encoding='NaN, +Infinity, -Infinity are strings')
        return dict(schema='native_solver_substep_audit_v1',
                    scope='all worlds, every native solve including graph warmup/replay',
                    stage='after constraint solve, before integration',
                    frame_tolerance=self.frame_tolerance, first_failure=failure,
                    contact_capacity=cmax, world_constraint_capacity=emax,
                    peak_contact_capacity_fraction=counts['max_contact_count']/cmax,
                    peak_constraint_capacity_fraction=counts['max_world_constraints']/emax,
                    **counts)

    def check(self):
        """Return normally only if every audited native solve remains valid."""
        if int(self.counters.numpy()[0]) == 0:
            raise RuntimeError('Native contact audit has not executed; install it before CUDA graph capture')
        if int(self.first.numpy()[0]):
            raise RuntimeError(f'Native contact audit failed: {self.report()["first_failure"]}')
