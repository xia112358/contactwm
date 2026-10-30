"""GPU-resident first-failure recorder for MJWarp, including captured CUDA graphs.

Diagnostic evidence only: normal training checkpoints keep their existing format.
"""
from __future__ import annotations

import importlib
import warp as wp
from mujoco_warp._src.types import vec5


@wp.kernel
def _stash(qp: wp.array2d[float], qv: wp.array2d[float], warm: wp.array2d[float],
           ctrl: wp.array2d[float], slot: int, nq: int, nv: int, nu: int,
           ring: wp.array3d[float]):
    w, j = wp.tid()
    if j < nq:
        ring[slot, w, j] = qp[w, j]
    if j < nv:
        ring[slot, w, nq+j] = qv[w, j]
        ring[slot, w, nq+nv+j] = warm[w, j]
    if j < nu:
        ring[slot, w, nq+2*nv+j] = ctrl[w, j]


@wp.kernel
def _detect(qp: wp.array2d[float], qv: wp.array2d[float], qa: wp.array2d[float],
            smooth: wp.array2d[float], nq: int, nv: int, slot: int, phase: int,
            first: wp.array[int]):
    w = wp.tid()
    if first[0] >= 0:
        return
    mask = int(0)
    for j in range(nq):
        if not wp.isfinite(qp[w, j]):
            mask = mask | 1
    for j in range(nv):
        if not wp.isfinite(qv[w, j]):
            mask = mask | 2
        if phase >= 5 and not wp.isfinite(qa[w, j]):
            mask = mask | 4
        if phase >= 4 and not wp.isfinite(smooth[w, j]):
            mask = mask | 8
    if mask != 0:
        previous = wp.atomic_cas(first, 0, -1, w)
        if previous == -1:
            first[1] = slot
            first[2] = phase
            first[3] = mask


@wp.kernel
def _capture(first: wp.array[int], slot: int, phase: int,
             qp: wp.array2d[float], qv: wp.array2d[float], qa: wp.array2d[float],
             smooth: wp.array2d[float], warm: wp.array2d[float], ctrl: wp.array2d[float],
             mass: wp.array2d[float], ld: wp.array2d[float], inv: wp.array2d[float],
             nefc: wp.array[int], niter: wp.array[int],
             efc_d: wp.array2d[float], efc_force: wp.array2d[float],
             efc_pos: wp.array2d[float], efc_aref: wp.array2d[float],
             nacon: wp.array[int], world: wp.array[int], dist: wp.array[float],
             margin: wp.array[float], kind: wp.array[int], geom: wp.array[wp.vec2i],
             dim: wp.array[int], pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
             friction: wp.array[vec5], solref: wp.array[wp.vec2], solimp: wp.array[vec5],
             address: wp.array2d[int],
             state_out: wp.array[float], mass_out: wp.array[float], ld_out: wp.array[float],
             inv_out: wp.array[float], efc_out: wp.array2d[float],
             contacts_out: wp.array2d[float], counts: wp.array[int]):
    w = first[0]
    if w < 0 or first[1] != slot or first[2] != phase:
        return
    offset = int(0)
    for j in range(qp.shape[1]):
        state_out[offset+j] = qp[w, j]
    offset += qp.shape[1]
    for j in range(qv.shape[1]):
        state_out[offset+j] = qv[w, j]
        state_out[offset+qv.shape[1]+j] = qa[w, j]
        state_out[offset+2*qv.shape[1]+j] = smooth[w, j]
        state_out[offset+3*qv.shape[1]+j] = warm[w, j]
    offset += 4*qv.shape[1]
    for j in range(ctrl.shape[1]):
        state_out[offset+j] = ctrl[w, j]
    for j in range(mass.shape[1]):
        mass_out[j] = mass[w, j]
    for j in range(ld.shape[1]):
        ld_out[j] = ld[w, j]
    for j in range(inv.shape[1]):
        inv_out[j] = inv[w, j]
    counts[0] = nacon[0]
    counts[1] = nefc[w]
    counts[2] = niter[w]
    for j in range(wp.min(nefc[w], efc_out.shape[0])):
        efc_out[j, 0] = efc_d[w, j]
        efc_out[j, 1] = efc_force[w, j]
        efc_out[j, 2] = efc_pos[w, j]
        efc_out[j, 3] = efc_aref[w, j]
    count = int(0)
    for i in range(wp.min(nacon[0], world.shape[0])):
        if world[i] != w:
            continue
        if count < contacts_out.shape[0]:
            contacts_out[count, 0] = dist[i]
            contacts_out[count, 1] = margin[i]
            contacts_out[count, 2] = float(kind[i])
            contacts_out[count, 3] = float(geom[i][0])
            contacts_out[count, 4] = float(geom[i][1])
            contacts_out[count, 5] = float(dim[i])
            for j in range(3):
                contacts_out[count, 6+j] = pos[i][j]
                for k in range(3):
                    contacts_out[count, 9+3*j+k] = frame[i][j, k]
            for j in range(5):
                contacts_out[count, 18+j] = friction[i][j]
                contacts_out[count, 25+j] = solimp[i][j]
            for j in range(2):
                contacts_out[count, 23+j] = solref[i][j]
            for j in range(address.shape[1]):
                contacts_out[count, 30+j] = float(address[i, j])
        count += 1
    counts[3] = count


class PhysicsFailureRecorder:
    PHASES = {1: "position_collision_constraints", 2: "velocity", 3: "actuation",
              4: "unconstrained_acceleration", 5: "constraint_solve", 6: "implicit_integration"}

    def __init__(self, manager):
        self.manager = manager
        self.solver = manager._solver
        self.data = d = self.solver.mjw_data
        self.nq, self.nv, self.nu = d.qpos.shape[1], d.qvel.shape[1], d.ctrl.shape[1]
        self.worlds = d.qpos.shape[0]
        self.substeps = manager._num_substeps
        self.device = d.qpos.device
        self.first = wp.full(4, -1, dtype=int, device=self.device)
        self.pre = wp.zeros((self.substeps, self.worlds, self.nq+2*self.nv+self.nu), device=self.device)
        self.state = wp.zeros(self.nq+4*self.nv+self.nu, device=self.device)
        self.mass = wp.zeros(d.M.shape[1], device=self.device)
        self.ld = wp.zeros(d.qLD.shape[1], device=self.device)
        self.inv = wp.zeros(d.qLDiagInv.shape[1], device=self.device)
        self.efc = wp.zeros((d.efc.D.shape[1], 4), device=self.device)
        self.contacts = wp.zeros((512, 30+d.contact.efc_address.shape[1]), device=self.device)
        self.counts = wp.zeros(4, dtype=int, device=self.device)
        self.slot = -1
        forward = importlib.import_module("mujoco_warp._src.forward")
        old_loop = manager._run_solver_substeps
        old_step = manager._step_solver

        def loop(*args, **kwargs):
            self.slot = -1
            try:
                return old_loop(*args, **kwargs)
            finally:
                self.slot = -1

        def step(*args, **kwargs):
            self.slot += 1
            return old_step(*args, **kwargs)

        manager._run_solver_substeps = loop
        manager._step_solver = step
        names = {"fwd_position": 1, "fwd_velocity": 2, "fwd_actuation": 3,
                 "fwd_acceleration": 4, "implicit": 6}
        for name, phase in names.items():
            setattr(forward, name, self._wrap(getattr(forward, name), phase))
        forward.solver.solve = self._wrap(forward.solver.solve, 5)
        print(f"Physical failure recorder enabled: {self.substeps} substeps, CUDA-graph-compatible, first failing phase + warm starts", flush=True)

    def _wrap(self, original, phase):
        def wrapped(*args, **kwargs):
            if self.slot < 0:
                return original(*args, **kwargs)
            if phase == 1:
                wp.launch(_stash, dim=(self.worlds, max(self.nq, self.nv, self.nu)), inputs=[
                    self.data.qpos, self.data.qvel, self.data.qacc_warmstart, self.data.ctrl,
                    self.slot, self.nq, self.nv, self.nu, self.pre], device=self.device)
            result = original(*args, **kwargs)
            self.check(phase)
            return result
        return wrapped

    def begin_control_step(self):
        self.first.fill_(-1)

    def has_failure(self):
        return wp.to_torch(self.first)[0] >= 0

    def check(self, phase):
        d, c = self.data, self.data.contact
        wp.launch(_detect, dim=self.worlds, inputs=[d.qpos, d.qvel, d.qacc, d.qacc_smooth,
                  self.nq, self.nv, self.slot, phase, self.first], device=self.device)
        wp.launch(_capture, dim=1, inputs=[self.first, self.slot, phase,
            d.qpos, d.qvel, d.qacc, d.qacc_smooth, d.qacc_warmstart, d.ctrl,
            d.M, d.qLD, d.qLDiagInv, d.nefc, d.solver_niter,
            d.efc.D, d.efc.force, d.efc.pos, d.efc.aref,
            d.nacon, c.worldid, c.dist, c.includemargin, c.type, c.geom, c.dim,
            c.pos, c.frame, c.friction, c.solref, c.solimp, c.efc_address,
            self.state, self.mass, self.ld, self.inv, self.efc, self.contacts, self.counts], device=self.device)

    def snapshot(self):
        """Call after a failed physical step, outside CUDA capture."""
        import torch
        first = self.first.numpy().tolist()
        if first[0] < 0:
            return {"first_failure_detected": False}
        world, slot, phase, mask = first
        counts = self.counts.numpy().tolist()
        pre = self.pre.numpy()[:slot+1, world].copy()
        result = {"first_failure_detected": True, "world": world, "substep": slot+1,
                  "phase": self.PHASES[phase], "nonfinite_mask_qpos_qvel_qacc_smooth": mask,
                  "substep_dt": self.manager._solver_dt, "pre_layout": [self.nq, self.nv, self.nv, self.nu],
                  "state_layout": [self.nq, self.nv, self.nv, self.nv, self.nv, self.nu],
                  "counts_nacon_nefc_iterations_worldcontacts": counts,
                  "pre_substeps_qpos_qvel_warmstart_ctrl": torch.from_numpy(pre),
                  "first_failure_state": torch.from_numpy(self.state.numpy()),
                  "mass_packed": torch.from_numpy(self.mass.numpy()),
                  "factor_packed": torch.from_numpy(self.ld.numpy()),
                  "factor_diag_inverse": torch.from_numpy(self.inv.numpy()),
                  "constraints_D_force_pos_aref": torch.from_numpy(self.efc.numpy()[:counts[1]].copy()),
                  "contacts": torch.from_numpy(self.contacts.numpy()[:min(counts[3], 512)].copy()),
                  "contacts_truncated": counts[3] > 512,
                  "contact_layout": "dist,margin,type,geom1,geom2,dim,pos3,frame9,friction5,solref2,solimp5,efc_addresses",
                  "geom_names": [self.solver.mj_model.geom(i).name for i in range(self.solver.mj_model.ngeom)]}
        return result
