"""Passive measured WBT contacts from the currently running Newton solve.

Reference commands never participate in activation, allocation or face selection.
Per-constraint forces are decoded by MJWarp; body net forces are not a substitute.
"""
from __future__ import annotations

import torch

from holosoma.utils.wbt_experiment import PARTS


class MeasuredContacts:
    def __init__(self, env):
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from somaforge_core.native_model_binding import inspect_model, native_binding

        self.env, self.manager = env, NewtonManager
        self.model, self.solver = NewtonManager.get_model(), NewtonManager._solver
        self.binding = native_binding(self.model, inspect_model(self.model))
        from somaforge_core.collision_profiles import execution_profile, contact_body_names
        from somaforge_core.robot_assets import canonical_g1_asset_metadata
        self.collision_profile = execution_profile(env)
        self.body_names = contact_body_names(self.collision_profile)
        self.robot_asset = canonical_g1_asset_metadata(collision_profile=self.collision_profile)
        from somaforge_core.live_contact_device import DeviceSupportContactSelector
        self.contact_selector = DeviceSupportContactSelector(self.binding,self.model.body_label,env.device, body_names_by_part=self.body_names)
        self.backend = 'native_solver_force_point_gpu_v1'

    @property
    def face_indexes(self):
        """Retain the native CPU face view for nine-part ACP readers."""
        if not hasattr(self, '_cpu_face_indexes'):
            from somaforge_core.native_surface_index import NativeSurfaceIndex
            self._cpu_face_indexes = {
                int(shape): NativeSurfaceIndex(faces)
                for shape, faces in self.binding['shape_surface'].items()
            }
        return self._cpu_face_indexes

    def read_evidence(self):
        """Owned raw rows; task face eligibility remains an independent view."""
        from somaforge_core.contact_system.evidence import BodyBinding, ContactEvidence, Sampling
        data = self.read()
        config = self.env.simulator.simulator_config.sim
        timing = Sampling(float(self.env.dt), 1./float(config.fps)/int(config.substeps),
                          substep=int(config.control_decimation)*int(config.substeps)-1)
        if not hasattr(self, "evidence_binding"):
            self.evidence_binding = BodyBinding(self.model.body_label, self.binding["body_env"],
                getattr(self, "evidence_parts", PARTS),
                getattr(self, "evidence_body_names", self.body_names), self.env.device)
        return ContactEvidence(data["snapshot"], sampling=timing, known=data["known"],
            control_step=self.env.episode_length_buf,
            provenance=dict(backend="isaaclab_newton_mjwarp", parts=self.evidence_binding.parts,
                            model_fingerprint=self.binding["model_fingerprint"], robot_asset=self.robot_asset))

    def _snapshot_gpu(self):
        import warp as wp
        wp.stream_from_torch(torch.cuda.current_stream(self.env.device)).wait_stream(
            wp.get_stream(self.model.device))
        contact = self.solver.mjw_data.contact
        get = lambda name:wp.to_torch(getattr(contact,name))
        dist, margin, kind = get('dist'), get('includemargin'), get('type')
        world, geom = get('worldid').long(), get('geom').long()
        mapping = wp.to_torch(self.solver.mjc_geom_to_newton_shape)
        shape_body = wp.to_torch(self.model.shape_body)
        count = wp.to_torch(self.solver.mjw_data.nacon).reshape(-1)[0]
        valid = torch.arange(len(dist),device=dist.device) < count
        mapped = (world >= 0) & (world < mapping.shape[0]) & ((geom >= 0) & (geom < mapping.shape[1])).all(-1)
        shapes = mapping[world.clamp(0,mapping.shape[0]-1)[:,None],geom.clamp(0,mapping.shape[1]-1)].long()
        shape_valid = ((shapes >= 0) & (shapes < len(shape_body))).all(-1)
        bodies = shape_body[shapes.clamp(0,len(shape_body)-1)]
        cone = int(self.solver.mjw_model.opt.cone)
        if cone not in (0,1):
            raise ValueError('Unknown native MJWarp contact cone')
        dim = get('dim')
        rows = torch.where(dim == 1,1,2*(dim-1)) if cone == 0 else dim
        return dict(dist=dist,includemargin=margin,type=kind,efc_address=get('efc_address'),
            worldid=world,position_w=get('pos'),frame_w=get('frame'),count=count,constraint_rows=rows,
            shape0=shapes[:,0],shape1=shapes[:,1],body0=bodies[:,0],body1=bodies[:,1],
            mapping_error=(valid & (~mapped | ~shape_valid)).any())

    @staticmethod
    def export_snapshot(snapshot):
        """Explicit audit export; normal training never downloads contact rows."""
        count = int(snapshot['count'])
        return {key:(value[:count].detach().cpu().numpy().copy() if value.ndim else value.item())
                if isinstance(value,torch.Tensor) else value for key,value in snapshot.items()}

    def read(self):
        import warp as wp
        import mujoco_warp as mjw

        cache = self.env._holosoma_step_cache
        key = ("wbt_measured_contacts", id(self.model))
        if key not in cache:
            snapshot = self._snapshot_gpu()
            selected = self.contact_selector(snapshot)
            total = self.env.num_envs*len(PARTS)
            selected['errors']['group'] = (selected['eligible'] &
                ((selected['groups'] < 0) | (selected['groups'] >= total))).any()
            self.contact_selector.validate(selected)
            count = int(snapshot["count"])
            eligible = selected['eligible'][:count]
            groups = selected['groups'][:count].clamp(0,total-1)
            active = torch.zeros(total,dtype=torch.int32,device=self.env.device)
            active.index_add_(0,groups,eligible.int())
            force = torch.zeros(total,3,device=self.env.device)
            snapshot.update(active=selected['active'],constraint_allocated=selected['constraint_allocated'])
            if count:
                # contact_force decodes the native friction cone/efc rows and
                # returns the world-space force ON body1 when world=True.
                id_tensor = torch.arange(count,dtype=torch.int32,device=self.env.device)
                ids = wp.from_torch(id_tensor)
                with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.env.device))):
                    wrench = wp.zeros(count, dtype=wp.spatial_vector, device=self.model.device)
                    mjw.contact_force(self.solver.mjw_model, self.solver.mjw_data, ids, True, wrench)
                decoded = wp.to_torch(wrench)
                values = decoded[:,:3].clone()
                snapshot["torque_on_body1_w"] = decoded[:,3:].clone()
                snapshot["force_on_body1_w"] = values
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite native per-contact force")
                sign = torch.where(selected['robot_side'][:count] == 1,1.,-1.)
                force.index_add_(0,groups,torch.where(eligible[:,None],values*sign[:,None],0))
            else:
                snapshot["force_on_body1_w"] = torch.zeros(0,3,device=self.env.device)
                snapshot["torque_on_body1_w"] = torch.zeros(0,3,device=self.env.device)
            cache[key] = {
                "active": (active > 0).reshape(self.env.num_envs,len(PARTS)),
                "force_w": force.reshape(self.env.num_envs,len(PARTS),3),
                "snapshot": snapshot,
                "excluded_rows": torch.where(selected['excluded'][:count])[0],
                "solved_root_state": self.env.simulator.robot_root_states[:].clone(),
                "solved_dof_pos": self.env.simulator.dof_pos.clone(),
            }
        data = cache[key]
        # Command window transitions can teleport without resetting the episode
        # counter. Require the exact post-solve articulation state as well.
        # This freshness check never defines or changes contact activation.
        unchanged = (self.env.simulator.robot_root_states[:] == data["solved_root_state"]).all(dim=-1)
        unchanged &= (self.env.simulator.dof_pos == data["solved_dof_pos"]).all(dim=-1)
        known = ((self.env.episode_length_buf > 0) & unchanged)[:, None].expand(-1, len(PARTS))
        return {**data, "known": known,
            "active": data["active"] & known,
            "force_w": data["force_w"] * known[..., None]}


def measured_contacts(env):
    reader = getattr(env, "_wbt_measured_contact_reader", None)
    if reader is None:
        if not hasattr(env, "_holosoma_step_cache"):
            env._holosoma_step_cache = {}
        reader = env._wbt_measured_contact_reader = MeasuredContacts(env)
    return reader.read()
