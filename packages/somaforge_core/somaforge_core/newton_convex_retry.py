# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
High-level collision detection functions for convex shapes.

Fused MPR + GJK approach with shared support functions and frame transform:
1. MPR with small inflate — exact normals for overlapping and near-touching shapes.
   Exits early for separated shapes (just a few support queries).
2. Only if MPR finds no overlap: GJK for accurate speculative contacts.

Both algorithms share support mapping code and the relative-frame coordinate
transform, reducing compiled code size and register pressure.
"""

from typing import Any

import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.mpr import create_solve_mpr, create_support_map_function
from newton._src.geometry.multicontact import create_build_manifold
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from .newton_convex_sat import solve_overlap

# Project-owned compatibility implementation; upstream license retained.
SCHEMA = 'newton_convex_conditional_retry_sat_v2'

@wp.func
def valid_result(n: wp.vec3, d: float):
    return (wp.isfinite(n[0]) and wp.isfinite(n[1]) and wp.isfinite(n[2])
            and wp.isfinite(d) and wp.abs(wp.length(n)-1.0) < 1.0e-4)



def create_solve_convex_multi_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """Factory: fused MPR+GJK multi-contact solver with shared support code."""

    # Create support functions ONCE — shared between MPR and GJK.
    support_funcs = create_support_map_function(support_func)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)

    @wp.func
    def solve_convex_multi_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        data_provider: Any,
        contact_threshold: float,
        skip_multi_contact: bool,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        # Shared relative-frame transform (computed once for both algorithms).
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # MPR inflate to prevent MPR/GJK flickering for resting contacts.
        # The switchover must never coincide with the resting signed distance
        # (which equals margin_sum when bodies are in stable contact):
        #   - margin == 0:       enlarge = 1e-4  (resting at 0, switch at 1e-4)
        #   - 0 < margin < 1e-4: enlarge = 2e-4  (resting < 1e-4, switch at 2e-4)
        #   - margin >= 1e-4:    enlarge = 0      (resting far from 0, no trick needed)
        margin_sum = contact_template.margin_a + contact_template.margin_b
        eps = 1.0e-4
        if margin_sum <= 0.0:
            enlarge = eps
        elif margin_sum < eps:
            enlarge = 2.0 * eps
        else:
            enlarge = 0.0

        # MPR with inflate for overlapping shapes.
        # Exits early (few support queries) when shapes are separated.
        collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            enlarge,
            data_provider,
        )

        _separated = bool(False)
        if collision:
            signed_distance = -penetration + enlarge
            # Undo the inflate on the witness points so downstream consumers
            # (manifold builder, contact writer) see true-surface positions.
            # The midpoint 0.5*(point_a + point_b) is unchanged (corrections cancel).
            half_enlarge = enlarge * 0.5
            point_a = point_a - normal * half_enlarge
            point_b = point_b + normal * half_enlarge
        else:
            # GJK fallback for separated shapes -- no Minkowski inflate; accurate normals/distances.
            _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a,
                geom_b,
                relative_orientation_b,
                relative_position_b,
                0.0,
                data_provider,
            )

        # Normal paths retain the upstream result bit-for-bit. Only ambiguous
        # GJK/MPR results are re-evaluated on the SAME geometry and pose.
        accepted = valid_result(normal, signed_distance)
        if not collision:
            accepted = accepted and _separated
        if not accepted:
            separated_retry, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a, geom_b, relative_orientation_b, relative_position_b,
                0.0, data_provider, 60, 1.0e-6)
            accepted = separated_retry and valid_result(normal, signed_distance)
            if not accepted:
                overlap_retry, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
                    geom_a, geom_b, relative_orientation_b, relative_position_b,
                    enlarge, data_provider, 100, 1.0e-7)
                signed_distance = -penetration + enlarge
                accepted = overlap_retry and valid_result(normal, signed_distance)
                if accepted:
                    point_a = point_a - normal * (enlarge * 0.5)
                    point_b = point_b + normal * (enlarge * 0.5)
            if not accepted:
                separated_fine, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                    geom_a, geom_b, relative_orientation_b, relative_position_b,
                    0.0, data_provider, 100, 1.0e-8)
                accepted = separated_fine and valid_result(normal, signed_distance)
            if not accepted:
                overlap_sat, point_a, point_b, normal, signed_distance = solve_overlap(
                    geom_a, geom_b, relative_orientation_b, relative_position_b)
                accepted = overlap_sat and valid_result(normal, signed_distance)
            if not accepted:
                wp.printf("SOMAFORGE_UNRESOLVED_CONVEX_CONTACT shapes=%d,%d normal=(%g,%g,%g) distance=%g\n",
                    contact_template.shape_a, contact_template.shape_b,
                    normal[0], normal[1], normal[2], signed_distance)
                # Fail closed even in release builds (assert may be compiled out).
                wp.breakpoint()
                return 0

        if skip_multi_contact or signed_distance > contact_threshold:
            # Transform to world space only for the single-contact early-out.
            point = 0.5 * (point_a + point_b)
            point = wp.quat_rotate(orientation_a, point) + position_a
            normal_ws = wp.quat_rotate(orientation_a, normal)

            contact_data = contact_template
            contact_data.contact_point_center = point
            contact_data.contact_normal_a_to_b = normal_ws
            contact_data.contact_distance = signed_distance
            contact_data.sort_sub_key = contact_template.sort_sub_key << 3
            contact_data = post_process_contact(
                contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b
            )
            writer_func(contact_data, writer_data, -1)
            return 1

        # Generate multi-contact manifold -- pass A-local-frame data directly
        # to avoid redundant world-space round-trip.
        count = wp.static(
            create_build_manifold(support_func, writer_func, post_process_contact, _support_funcs=support_funcs)
        )(
            geom_a,
            geom_b,
            orientation_a,
            position_a,
            relative_orientation_b,
            relative_position_b,
            point_a,
            point_b,
            normal,
            data_provider,
            writer_data,
            contact_template,
        )

        return count

    return solve_convex_multi_contact


def create_solve_convex_single_contact(support_func: Any, writer_func: Any, post_process_contact: Any):
    """Factory: fused MPR+GJK single-contact solver with shared support code."""

    # Create support functions ONCE — shared between MPR and GJK.
    support_funcs = create_support_map_function(support_func)
    solve_mpr = create_solve_mpr(support_func, _support_funcs=support_funcs)
    solve_gjk = create_solve_closest_distance(support_func, _support_funcs=support_funcs)

    @wp.func
    def solve_convex_single_contact(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        data_provider: Any,
        contact_threshold: float,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        # Shared relative-frame transform (computed once for both algorithms).
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        # MPR inflate to prevent MPR/GJK flickering for resting contacts.
        # See create_solve_convex_multi_contact for detailed explanation.
        margin_sum = contact_template.margin_a + contact_template.margin_b
        eps = 1.0e-4
        if margin_sum <= 0.0:
            enlarge = eps
        elif margin_sum < eps:
            enlarge = 2.0 * eps
        else:
            enlarge = 0.0

        # MPR with inflate for overlapping shapes.
        collision, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            enlarge,
            data_provider,
        )

        _separated = bool(False)
        if collision:
            signed_distance = -penetration + enlarge
            half_enlarge = enlarge * 0.5
            point_a = point_a - normal * half_enlarge
            point_b = point_b + normal * half_enlarge
        else:
            # GJK fallback for separated shapes -- no Minkowski inflate; accurate normals/distances.
            _separated, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a,
                geom_b,
                relative_orientation_b,
                relative_position_b,
                0.0,
                data_provider,
            )

        # Normal paths retain the upstream result bit-for-bit. Only ambiguous
        # GJK/MPR results are re-evaluated on the SAME geometry and pose.
        accepted = valid_result(normal, signed_distance)
        if not collision:
            accepted = accepted and _separated
        if not accepted:
            separated_retry, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                geom_a, geom_b, relative_orientation_b, relative_position_b,
                0.0, data_provider, 60, 1.0e-6)
            accepted = separated_retry and valid_result(normal, signed_distance)
            if not accepted:
                overlap_retry, point_a, point_b, normal, penetration = wp.static(solve_mpr.core)(
                    geom_a, geom_b, relative_orientation_b, relative_position_b,
                    enlarge, data_provider, 100, 1.0e-7)
                signed_distance = -penetration + enlarge
                accepted = overlap_retry and valid_result(normal, signed_distance)
                if accepted:
                    point_a = point_a - normal * (enlarge * 0.5)
                    point_b = point_b + normal * (enlarge * 0.5)
            if not accepted:
                separated_fine, point_a, point_b, normal, signed_distance = wp.static(solve_gjk.core)(
                    geom_a, geom_b, relative_orientation_b, relative_position_b,
                    0.0, data_provider, 100, 1.0e-8)
                accepted = separated_fine and valid_result(normal, signed_distance)
            if not accepted:
                overlap_sat, point_a, point_b, normal, signed_distance = solve_overlap(
                    geom_a, geom_b, relative_orientation_b, relative_position_b)
                accepted = overlap_sat and valid_result(normal, signed_distance)
            if not accepted:
                wp.printf("SOMAFORGE_UNRESOLVED_CONVEX_CONTACT shapes=%d,%d normal=(%g,%g,%g) distance=%g\n",
                    contact_template.shape_a, contact_template.shape_b,
                    normal[0], normal[1], normal[2], signed_distance)
                # Fail closed even in release builds (assert may be compiled out).
                wp.breakpoint()
                return 0

        # Transform results back to world space (once).
        point = 0.5 * (point_a + point_b)
        point = wp.quat_rotate(orientation_a, point) + position_a
        normal = wp.quat_rotate(orientation_a, normal)

        contact_data = contact_template
        contact_data.contact_point_center = point
        contact_data.contact_normal_a_to_b = normal
        contact_data.contact_distance = signed_distance
        contact_data.sort_sub_key = contact_template.sort_sub_key << 3

        contact_data = post_process_contact(
            contact_data, geom_a, position_a, orientation_a, geom_b, position_b, orientation_b
        )
        writer_func(contact_data, writer_data, -1)

        return 1

    return solve_convex_single_contact
