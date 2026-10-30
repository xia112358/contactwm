"""Owned, lossless solver evidence with explicit selection and sampling time."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping

import torch

from . import SCHEMA


@dataclass(frozen=True)
class Sampling:
    control_dt: float
    physics_dt: float
    substep: int | None = None
    complete_interval: bool = False
    pose_contact_same_state: bool = False
    position_time: str = "last_constraint_evaluation"
    pose_time: str = "post_integration"

    def __post_init__(self):
        import math
        if not all(math.isfinite(x) and x > 0 for x in (self.control_dt, self.physics_dt)):
            raise ValueError("Contact sampling requires positive finite timesteps")
        if self.physics_dt > self.control_dt or (self.substep is not None and self.substep < 0):
            raise ValueError("Invalid contact sampling interval/substep")
        if self.complete_interval:
            raise ValueError("One solver snapshot cannot certify a complete control interval")


class BodyBinding:
    """Actual body/environment identities; part views never change raw facts."""

    def __init__(self, labels, body_env: Mapping[int, int], parts, names_by_part, device="cpu"):
        import numpy as np
        self.parts = tuple(parts)
        self.labels = tuple(str(label) for label in labels)
        if not len(labels) or not self.parts or len(set(self.parts)) != len(self.parts):
            raise ValueError("Contact part names must be unique")
        # Compile on the host once, then upload once, rather than issuing one
        # scalar device write per body in thousands of environments.
        world = np.full(len(labels), -1, dtype=np.int64)
        part = np.full_like(world, -1)
        by_name = {}
        for i, name in enumerate(self.parts):
            for body in names_by_part[name]:
                if body in by_name and by_name[body] != i:
                    raise ValueError("A physical body belongs to multiple contact parts")
                by_name[body] = i
        for body, env in body_env.items():
            body, env = int(body), int(env)
            if not 0 <= body < len(labels) or env < 0:
                raise ValueError("Invalid actual body/environment binding")
            world[body] = env
            part[body] = by_name.get(str(labels[body]).rsplit("/", 1)[-1], -1)
        self.world = torch.as_tensor(world, device=device)
        self.part = torch.as_tensor(part, device=device)

    def metadata(self):
        return dict(parts=self.parts, body_world=self.world.cpu().tolist(),
                    body_part=self.part.cpu().tolist(), body_labels=self.labels)

    @classmethod
    def from_model(cls, model, parts, names_by_part, device):
        labels = model.body_label
        actual = model.body_world.numpy()
        body_env = {i: int(actual[i]) for i, label in enumerate(labels)
                    if re.search(r"/env_\d+/Robot/", str(label))}
        return cls(labels, body_env, parts, names_by_part, device)


class ContactEvidence:
    """All valid solver rows, including inactive, self and excluded contacts.

    Every tensor owns its storage. A following physics solve cannot overwrite
    retained evidence. No persistent identity is inferred from a solver row ID.
    """

    REQUIRED = ("dist", "includemargin", "type", "efc_address", "constraint_rows",
                "worldid", "shape0", "shape1", "body0", "body1", "position_w", "frame_w",
                "force_on_body1_w")

    def __init__(self, snapshot, *, sampling: Sampling, known, provenance=None, control_step=None):
        self.sampling = sampling
        self.provenance = dict(provenance or {})
        self.known = torch.as_tensor(known, dtype=torch.bool).detach().clone()
        if self.known.ndim != 2 or min(self.known.shape) < 1:
            raise ValueError("Evidence requires explicit [environment, part] known masks")
        device = self.known.device
        missing = set(self.REQUIRED) - snapshot.keys()
        if missing:
            raise ValueError("Missing native contact evidence: " + ", ".join(sorted(missing)))
        count = torch.as_tensor(snapshot["count"])
        if count.numel() != 1 or not torch.isfinite(count).all() or count.item() != int(count.item()):
            raise ValueError("Invalid native contact count")
        self.count = int(count.item())
        if self.count < 0:
            raise ValueError("Invalid native contact count")
        self.rows = {}
        optional = ("torque_on_body1_w", "application_point0_local", "application_point1_local",
                    "point_velocity0_w", "point_velocity1_w", "pre_point_velocity0_w", "pre_point_velocity1_w",
                    "geometry_point0_w", "geometry_point1_w")
        for key in (*self.REQUIRED, *optional):
            if key not in snapshot:
                continue
            value = torch.as_tensor(snapshot[key], device=device)
            if value.ndim < 1 or len(value) < self.count:
                raise ValueError("Contact count exceeds " + key + " capacity")
            self.rows[key] = value[:self.count].detach().clone()
        r = self.rows
        errors = {}
        for key in ("dist", "includemargin", "type", "constraint_rows", "worldid", "shape0", "shape1", "body0", "body1"):
            if r[key].shape != (self.count,):
                raise ValueError("Invalid contact field shape: " + key)
        for key in ("position_w", "force_on_body1_w"):
            if r[key].shape != (self.count, 3):
                raise ValueError("Invalid contact vector: " + key)
            errors[key] = ~torch.isfinite(r[key]).all()
        for key in optional:
            if key in r:
                if r[key].shape != (self.count, 3):
                    raise ValueError("Invalid optional contact vector: " + key)
                errors[key] = ~torch.isfinite(r[key]).all()
        if r["frame_w"].shape != (self.count, 3, 3):
            raise ValueError("Invalid contact frames")
        errors["frame_w"] = ~torch.isfinite(r["frame_w"]).all()
        address = r["efc_address"]
        if address.ndim != 2 or address.shape[0] != self.count or address.shape[1] < 1:
            raise ValueError("Invalid native constraint addresses")
        needed = r["constraint_rows"]
        errors["constraint_rows"] = ((needed < 1) | (needed > address.shape[1])).any()
        errors["activation_fields"] = (~torch.isfinite(r["dist"]) | ~torch.isfinite(r["includemargin"])).any()
        self.active = ((r["type"].long() & 1) != 0) & (r["dist"] < r["includemargin"])
        required = torch.arange(address.shape[1], device=device)[None] < needed[:, None]
        self.allocated = self.active & ((address >= 0) | ~required).all(-1)
        for key, value in (("active", self.active), ("constraint_allocated", self.allocated)):
            if key in snapshot:
                saved = torch.as_tensor(snapshot[key], device=device)[:self.count].bool()
                if saved.shape != value.shape:
                    raise ValueError("Invalid saved native " + key)
                errors[key] = (saved != value).any()
        errors["Unallocated active native constraint"] = (self.active & ~self.allocated).any()
        errors["worldid"] = ((r["worldid"] < 0) | (r["worldid"] >= len(self.known))).any()
        # One status transfer for numerical validation, regardless of row count.
        invalid = torch.stack(list(errors.values())).detach().cpu().tolist()
        for key, failed in zip(errors, invalid):
            if failed:
                raise ValueError("Invalid native contact evidence: " + key)
        self.control_step = None if control_step is None else torch.as_tensor(control_step, device=device).detach().clone()

    def terrain_view(self, binding: BodyBinding, *, eligible=None):
        """Select one-robot pairs; optional task eligibility is a separate view."""
        if len(binding.parts) != self.known.shape[1]:
            raise ValueError("Part order/known mask mismatch")
        r = self.rows
        bodies = torch.stack((r["body0"], r["body1"]), -1).long()
        valid = (bodies >= 0) & (bodies < len(binding.world))
        safe = bodies.clamp(0, max(0, len(binding.world)-1))
        world = torch.where(valid, binding.world[safe], -1)
        parts = torch.where(valid, binding.part[safe], -1)
        robot = world >= 0
        side = robot[:, 1].long()
        index = torch.arange(self.count, device=side.device)
        part = parts[index, side]
        own_world = world[index, side]
        selected = self.allocated & (robot.sum(-1) == 1) & (part >= 0) & (own_world == r["worldid"])
        if eligible is not None:
            mask = torch.as_tensor(eligible, device=side.device, dtype=torch.bool)
            if mask.ndim != 1 or len(mask) < self.count:
                raise ValueError("Invalid task selection mask")
            selected &= mask[:self.count]
        group = r["worldid"].long()*len(binding.parts) + part
        known = self.known.flatten()[group.clamp(0, self.known.numel()-1)]
        selected &= known
        sign = torch.where(side == 1, 1., -1.).to(r["position_w"].dtype)
        rows = selected.nonzero().flatten()
        torque = r.get("torque_on_body1_w")
        return ContactView(self, rows, group[rows], side[rows],
                           r["force_on_body1_w"][rows]*sign[rows, None],
                           None if torque is None else torque[rows]*sign[rows, None])

    def export(self):
        import json
        import numpy as np
        result = {key: value.cpu().numpy().copy() for key, value in self.rows.items()}
        result.update(count=np.asarray(self.count), active=self.active.cpu().numpy(),
                      constraint_allocated=self.allocated.cpu().numpy(), known=self.known.cpu().numpy(),
                      evidence_metadata_json=np.asarray(json.dumps(dict(schema=SCHEMA, sampling=asdict(self.sampling),
                        provenance=self.provenance,available_fields=sorted(self.rows),
                        units=dict(position="m",force="N",torque="N m",velocity="m/s"),
                        force_convention="world force/torque on body1; signs change in robot-side views"))))
        if self.control_step is not None:
            result["control_step"] = self.control_step.cpu().numpy()
        return result


@dataclass
class ContactView:
    evidence: ContactEvidence
    rows: torch.Tensor
    groups: torch.Tensor
    robot_side: torch.Tensor
    force_w: torch.Tensor
    torque_w: torch.Tensor | None

    @property
    def position_w(self):
        return self.evidence.rows["position_w"][self.rows]

    def reduce(self):
        shape = self.evidence.known.shape
        total = self.evidence.known.numel()
        count = torch.zeros(total, device=self.groups.device, dtype=torch.long)
        count.index_add_(0, self.groups, torch.ones_like(self.groups))
        force = self.force_w.new_zeros(total, 3)
        force.index_add_(0, self.groups, self.force_w)
        return dict(active=(count > 0).reshape(shape), known=self.evidence.known,
                    force_w=force.reshape(*shape, 3), contact_count=count.reshape(shape))


def solver_snapshot(solver, model):
    """Read current MJWarp rows and solve-geometry local points, without stepping."""
    import warp as wp
    import mujoco_warp as mjw
    device = str(model.device)
    if str(device).startswith("cuda"):
        wp.stream_from_torch(torch.cuda.current_stream(device)).wait_stream(wp.get_stream(model.device))
    c, d, m = solver.mjw_data.contact, solver.mjw_data, solver.mjw_model
    n = int(wp.to_torch(d.nacon).reshape(-1)[0])
    if not 0 <= n <= len(c.dist):
        raise ValueError("Native contact capacity exceeded")
    get = lambda key: wp.to_torch(getattr(c, key))[:n]
    result = {key: get(attr) for key, attr in (("dist", "dist"), ("includemargin", "includemargin"),
        ("type", "type"), ("efc_address", "efc_address"), ("worldid", "worldid"),
        ("position_w", "pos"), ("frame_w", "frame"))}
    world, geom = result["worldid"].long(), get("geom").long()
    mapping = wp.to_torch(solver.mjc_geom_to_newton_shape)
    if (((world < 0) | (world >= len(mapping))).any() | ((geom < 0) | (geom >= mapping.shape[1])).any()):
        raise ValueError("Invalid actual solver world/geometry mapping")
    shapes = mapping[world[:, None], geom].long()
    shape_body = wp.to_torch(model.shape_body)
    if ((shapes < 0) | (shapes >= len(shape_body))).any():
        raise ValueError("Invalid actual Newton shape mapping")
    bodies = shape_body[shapes]
    cone = int(m.opt.cone)
    if cone not in (0, 1):
        raise ValueError("Unknown native contact cone")
    dim = get("dim")
    result.update(count=n, constraint_rows=torch.where(dim == 1, 1, 2*(dim-1)) if cone == 0 else dim,
                  shape0=shapes[:, 0], shape1=shapes[:, 1], body0=bodies[:, 0], body1=bodies[:, 1])
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))) if str(device).startswith("cuda") else wp.ScopedDevice(device):
        wrench = wp.zeros(n, dtype=wp.spatial_vector, device=model.device)
        if n:
            mjw.contact_force(m, d, wp.from_torch(torch.arange(n, dtype=torch.int32, device=device)), True, wrench)
    values = wp.to_torch(wrench)
    result.update(force_on_body1_w=values[:, :3], torque_on_body1_w=values[:, 3:])
    geom_body = wp.to_torch(m.geom_bodyid).long()[geom]
    positions = wp.to_torch(d.xpos)
    rotations = wp.to_torch(d.xmat).reshape(*positions.shape[:2], 3, 3)
    for side in (0, 1):
        b = geom_body[:, side]
        result[f"application_point{side}_local"] = torch.einsum(
            "nji,nj->ni", rotations[world, b], result["position_w"]-positions[world, b])
    return result
