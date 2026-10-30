"""Nine-body view of the existing Newton/MJWarp constraint evidence."""
from __future__ import annotations

import numpy as np
import torch

from holosoma.acp.labels import PARTS
from holosoma.utils.wbt_contacts import MeasuredContacts
from somaforge_core.contact_schema import CONTACT_BODY_NAMES_BY_PART

BODY_MAPPING = {**CONTACT_BODY_NAMES_BY_PART,
                "left_elbow": ("left_elbow_link",), "right_elbow": ("right_elbow_link",),
                "pelvis": ("pelvis", "pelvis_contour_link")}


class ACPMeasuredContacts(MeasuredContacts):
    def read(self):
        self.evidence_parts = PARTS
        self.evidence_body_names = {**BODY_MAPPING, **self.body_names}
        import mujoco_warp as mjw
        import warp as wp
        from somaforge_core.newton_contacts import snapshot_solver_contacts
        from somaforge_core.newton_support import select_support_contacts

        cache = self.env._holosoma_step_cache
        key = ("acp_native_contacts", id(self.model))
        if key not in cache:
            snapshot = snapshot_solver_contacts(self.solver, self.model)
            selected = select_support_contacts(snapshot, self.binding, self.model.body_label,
                face_indexes=self.face_indexes, part_names=PARTS, body_names_by_part=self.evidence_body_names)
            count, size = int(snapshot["count"]), self.env.num_envs * 9
            active, force, positions = np.zeros(size, bool), np.zeros((size, 3), np.float32), np.zeros((size, 3), np.float32)
            score = np.full(size, np.inf)
            values = np.zeros((count, 3), np.float32)
            if count:
                indices = wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=self.model.device)
                wrench = wp.zeros(count, dtype=wp.spatial_vector, device=self.model.device)
                mjw.contact_force(self.solver.mjw_model, self.solver.mjw_data, indices, True, wrench)
                decoded = wrench.numpy()
                values = decoded[:, :3]
                snapshot["torque_on_body1_w"] = decoded[:, 3:].copy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite native contact wrench")
                for row in np.flatnonzero(selected["eligible"]):
                    group = int(selected["groups"][row])
                    if not 0 <= group < size:
                        raise ValueError("ACP contact body/world mapping mismatch")
                    active[group] = True
                    force[group] += values[row] * (1 if selected["robot_side"][row] == 1 else -1)
                    depth = snapshot["dist"][row] - snapshot["includemargin"][row]
                    if depth < score[group]:
                        score[group], positions[group] = depth, snapshot["position_w"][row]
            snapshot["force_on_body1_w"] = values
            if not count:
                snapshot["torque_on_body1_w"] = np.zeros((0, 3), np.float32)
            cache[key] = {"active": torch.as_tensor(active.reshape(-1, 9), device=self.env.device),
                "force_w": torch.as_tensor(force.reshape(-1, 9, 3), device=self.env.device),
                "position_w": torch.as_tensor(positions.reshape(-1, 9, 3), device=self.env.device),
                "snapshot": snapshot, "excluded_rows": selected["excluded_rows"],
                "root": self.env.simulator.robot_root_states[:].clone(), "q": self.env.simulator.dof_pos.clone()}
        data = cache[key]
        unchanged = (data["root"] == self.env.simulator.robot_root_states[:]).all(-1)
        unchanged &= (data["q"] == self.env.simulator.dof_pos).all(-1)
        known = ((self.env.episode_length_buf > 0) & unchanged)[:, None].expand(-1, 9)
        return {**data, "known": known, "active": data["active"] & known,
                "force_w": data["force_w"] * known[..., None], "position_w": data["position_w"] * known[..., None]}


def measured_contacts(env):
    reader = getattr(env, "_acp_contact_reader", None)
    if reader is None:
        if not hasattr(env, "_holosoma_step_cache"):
            env._holosoma_step_cache = {}
        reader = env._acp_contact_reader = ACPMeasuredContacts(env)
    return reader.read()
