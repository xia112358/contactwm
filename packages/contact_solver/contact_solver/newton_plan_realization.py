"""Realize a predicted limb/surface plan with fresh Newton witnesses.

This module does not invent a contact threshold.  Contact activation,
allocation, margins, surface attribution, and full-body separation all come
from the current Newton/MJWarp query.  Frozen witness derivatives only provide
the local optimization direction at the queried pose.
"""

from __future__ import annotations

import torch
from torch import Tensor
from contact_solver.interaction_acceptance import DEFAULT_ACCEPTANCE, contact_acceptance

from somaforge_core.contact_face_selection import select_contact_pairs, upward_face_mask
from somaforge_core.contact_source_geometry import SOURCE_NORMAL_FAN_SCHEMA

from contact_solver.newton_witness_loss import full_body_violation, query_local_distances, selected_contact_cost, unwanted_contact_cost


def newton_plan_realization(
    model,
    qpos: Tensor,
    intended_contact: Tensor,
    intended_surface: Tensor,
    scene: dict[str, Tensor],
    *,
    world_frame: tuple[Tensor, Tensor] | None = None,
    fingerprints: Tensor | None = None,
    invalid_witness_policy: str = "error",
    witness_audit_path=None,
    query_override=None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return an unconditional own-plan loss and authoritative diagnostics."""

    if intended_contact.shape != (len(qpos), 6) or intended_contact.dtype != torch.bool:
        raise ValueError("intended_contact must be bool [B,6]")
    if intended_surface.shape != intended_contact.shape:
        raise ValueError("intended_surface must be [B,6]")
    if bool(((intended_surface < 0) & intended_contact).any()):
        raise ValueError("active intended contacts require a support surface")
    if query_override is not None:
        from contact_solver.device_contact_objective import DeviceWitnessRows, realization
        if isinstance(query_override[0], DeviceWitnessRows):
            if len(query_override[0]) != len(qpos):
                raise ValueError('Device witness batch mismatch')
            return realization(query_override[0], intended_contact, intended_surface,
                invalid_policy=invalid_witness_policy, audit_path=witness_audit_path)

    if query_override is None:
        rows, observed = query_local_distances(
            model.fk,
            qpos,
            scene,
            world_frame=world_frame,
            fingerprints=fingerprints,
        )
    else:
        rows, observed = query_override
        if len(rows) != len(qpos):
            raise ValueError("Newton plan-realization query batch mismatch")
    if observed.get("surface_attribution_schema") != SOURCE_NORMAL_FAN_SCHEMA:
        raise ValueError("plan realization requires source-aware Newton contact queries")
    if "full_robot_separation" not in observed:
        raise ValueError("plan realization requires Newton full-body separation")

    pair_cost, missing, realized = selected_contact_cost(
        qpos,
        rows,
        intended_contact,
        intended_surface,
        # Upper bound of the contact gap interval, independent of activation.
        # Its lower bound is penalized exactly once by full-body safety below.
        continuous_gap=True,
    )
    catalogs = observed.get("surface_catalog_by_sample")
    if catalogs is None:
        catalogs = [observed["surface_catalog"]] * len(qpos)
    eligible = [
        {
            int(face["surface"])
            for face, keep in zip(
                catalog,
                upward_face_mask([face["normal_w"] for face in catalog]),
                strict=True,
            )
            if keep
        }
        for catalog in catalogs
    ]
    configured = observed.get("configured_terrain_includemargins_by_sample")
    if configured is None:
        common = observed.get("configured_terrain_includemargins")
        configured = None if common is None else [common] * len(qpos)
    if configured is None or len(configured) != len(qpos):
        raise ValueError("plan realization lacks per-sample configured Newton margins")
    configured_margin = []
    for values in configured:
        positive = [float(value) for value in values if float(value) > 0.0]
        if not positive:
            raise ValueError("plan realization lacks a positive configured Newton margin")
        # The smallest realized task-pair margin is a conservative interior
        # target while the requested pair does not yet exist.
        configured_margin.append(min(positive))
    configured_margin = qpos.new_tensor(configured_margin)
    unwanted, unwanted_active = unwanted_contact_cost(
        qpos, rows, intended_contact, eligible
    )
    depth, invalid = full_body_violation(
        model.fk,
        qpos,
        observed["full_robot_separation"],
        world_frame=world_frame,
        invalid_policy=invalid_witness_policy,
        return_validity=True,
        audit_path=witness_audit_path,
    )

    active_count = intended_contact.sum(-1).clamp_min(1)
    contact_normalized = pair_cost / 0.02**2
    contact_loss = (
        (contact_normalized * intended_contact).sum(-1) / active_count
        + (contact_normalized * intended_contact).amax(-1)
    )
    unwanted_normalized = unwanted / 0.02**2
    unwanted_loss = unwanted_normalized.mean(-1) + unwanted_normalized.amax(-1)
    collision_loss = ((depth-DEFAULT_ACCEPTANCE.shallow_penetration_m).relu() / 0.005).square()
    loss = contact_loss + unwanted_loss + collision_loss

    exact = []
    accepted = []
    ignored_extras = []
    blocking_extras = []
    extra_overlaps = []
    intended_cpu = intended_contact.detach().cpu().tolist()
    surface_cpu = intended_surface.detach().cpu().tolist()
    for index, catalog in enumerate(catalogs):
        selected = select_contact_pairs([observed["pairs"][index]], catalog)
        audit = contact_acceptance(selected["contact_pairs"][0],
                                   intended_cpu[index], surface_cpu[index])
        accepted.append(audit["contact_accepted"])
        ignored_extras.append(audit["ignored_extra_pairs"])
        blocking_extras.append(audit["blocking_extra_pairs"])
        extra_overlaps.append(audit["max_extra_activation_overlap_m"])
        actual = qpos.new_tensor(selected["contact_part_mask"][0], dtype=torch.bool)
        surface = qpos.new_tensor(selected["contact_surface"][0], dtype=torch.long)
        exact.append(
            (actual == intended_contact[index]).all()
            & (
                (surface == intended_surface[index]) | ~intended_contact[index]
            ).all()
        )

    metrics = {
        "newton_contact_accepted": qpos.new_tensor(accepted),
        "newton_generation_accepted": (
            qpos.new_tensor(accepted, dtype=torch.bool)
            & torch.isfinite(depth) & (depth >= 0)
            & (depth <= DEFAULT_ACCEPTANCE.shallow_penetration_m) & (invalid == 0)
        ).float(),
        "newton_ignored_extra_pairs": qpos.new_tensor(ignored_extras),
        "newton_blocking_extra_pairs": qpos.new_tensor(blocking_extras),
        "newton_max_extra_activation_overlap_m": qpos.new_tensor(extra_overlaps),
        "newton_plan_realization_loss": contact_loss,
        "newton_unwanted_contact_loss": unwanted_loss,
        "newton_collision_loss": collision_loss,
        "newton_plan_exact": torch.stack(exact).float(),
        "newton_generation_valid": (
            torch.stack(exact) & (depth == 0.0) & (invalid == 0)
        ).float(),
        "newton_intended_contacts": intended_contact.sum(-1).float(),
        "newton_realized_intended_contacts": (intended_contact & realized).sum(-1).float(),
        "newton_unrealized_intended_contacts": (intended_contact & ~realized).sum(-1).float(),
        "newton_missing_intended_pairs": missing.sum(-1).float(),
        "newton_missing_intended_mask": missing,
        "newton_unwanted_contact_parts": unwanted_active.sum(-1).float(),
        "newton_penetration_cm": 100.0 * depth,
        "newton_invalid_fullbody_witnesses": invalid,
        "newton_configured_includemargin": configured_margin,
    }
    return loss, metrics


__all__ = ["newton_plan_realization"]
