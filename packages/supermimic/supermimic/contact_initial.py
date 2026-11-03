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
        self.reader._known_mask = lambda: torch.ones((self.env.N, 6), device=self.env.device, dtype=torch.bool)

    def read(self):
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
        return (evidence['active'] & evidence['known']).clone()
