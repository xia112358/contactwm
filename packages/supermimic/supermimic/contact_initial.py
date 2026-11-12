"""Read post-reset native contacts without advancing or changing the live solver."""
import copy
import torch
import warp as wp


class InitialContactProbe:
    def __init__(self, reader):
        self.reader = copy.copy(reader)
        self.env = reader.env
        manager = self.env.sim.backend_manager
        source = reader.solver.mjw_data
        self.solver = copy.copy(reader.solver)
        with wp.ScopedDevice(reader.model.device):
            self.solver.mjw_data = self.solver._mujoco_warp.make_data(
                self.solver.mj_model, nworld=source.nworld, naconmax=source.naconmax,
                njmax=source.njmax, njmax_nnz=source.njmax_nnz, naccdmax=source.naccdmax)
            wp.copy(self.solver.mjw_data.contact.type, source.contact.type)
            for key in ('_last_contact_generation', '_contact_tid_to_cid', '_last_nacon_count'):
                setattr(self.solver, key, wp.clone(getattr(reader.solver, key)))
        self.solver._invalidate_contact_fast_path()
        self.solver._last_contacts_id = None
        self.contacts = manager._collision_pipeline.contacts()
        self.reader.solver = self.solver
        # A separate native solve is performed on each read; it is valid even
        # though the training environment's post-reset step counter is zero.
        self.reader._known_mask = lambda *, solved=False: torch.ones((self.env.N, 6), device=self.env.device, dtype=torch.bool)

    def read(self, *, raw=False):
        manager = self.env.sim.backend_manager
        model = self.reader.model
        state = manager.backend.state_0
        self.env.scene.write_data_to_sim()
        if manager.kinematics_dirty:
            manager._eval_fk()
        manager._collision_pipeline.collide(state, self.contacts)
        self.solver._update_mjc_data(self.solver.mjw_data, model, state)
        self.solver._convert_contacts_to_mjwarp(model, state, self.contacts)
        self.solver._mujoco_warp.forward(self.solver.mjw_model, self.solver.mjw_data)
        evidence = self.reader.read_device()
        if raw:
            view = self.reader.last_evidence.terrain_view(self.reader.evidence_binding)
            return (view.reduce()['contact_count'] > 0).clone()
        return (evidence['active'] & evidence['known']).clone()


def reference_entry_phase(clock, release_steps):
    """Carry the existing native release debounce across reference reset cuts.

    Uses only preceding source substeps, never future frames or a new geometric
    threshold. Source boundaries reset history; genuinely airborne entries arm
    immediately. Return both the phase and its already-observed absence count.
    """
    import numpy as np
    from .contact_release import native_release_history
    active=clock.data['substep_raw_contact_active'].cpu().numpy()
    known=clock.data['substep_contact_known'].cpu().numpy()
    position=clock.data['substep_part_position'].cpu().numpy()
    floors=clock.frame_source_floor.cpu().numpy()
    blocked=np.zeros((len(active),6),dtype=bool)
    absent=np.zeros((len(active),6),dtype=np.int32)
    boundaries=np.r_[np.flatnonzero(floors==np.arange(len(active))),len(active)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        phase,run,_=native_release_history(active[begin:end].reshape(-1,6),known[begin:end].reshape(-1,6),position[begin:end].reshape(-1,6,3),release_steps)
        tail=np.arange(1,end-begin)*active.shape[1]-1
        blocked[begin+1:end]=phase[tail]
        absent[begin+1:end]=np.where(phase[tail],run[tail],0)
    ids=clock.segment_starts.cpu()
    device=clock.segment_starts.device
    return torch.from_numpy(blocked)[ids].to(device),torch.from_numpy(absent)[ids].to(device)
