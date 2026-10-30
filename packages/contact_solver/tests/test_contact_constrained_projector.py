import torch

import contact_solver.contact_constrained_projector as projector_module
from contact_solver.contact_constrained_projector import (
    CanonicalContactCollisionGeometry,
    NewtonContactProjector,
    apply_tangent_delta,
)
from somaforge_core.g1_kinematics import CanonicalG1ForwardKinematics


def canonical_q(fk: CanonicalG1ForwardKinematics) -> torch.Tensor:
    q = torch.zeros((1, 36), dtype=torch.float32)
    q[:, 3] = 1.0
    q[:, 7:] = 0.5 * (fk.joint_lower + fk.joint_upper)
    return q


def test_tangent_update_normalizes_root_and_clamps_canonical_joint_limits() -> None:
    fk = CanonicalG1ForwardKinematics()
    q = canonical_q(fk)
    delta = torch.zeros((1, 35))
    delta[0, :6] = torch.tensor((0.1, -0.2, 0.3, 0.1, -0.2, 0.3))
    delta[0, 6:] = 100.0
    result = apply_tangent_delta(q, delta, fk)
    torch.testing.assert_close(result[0, :3], delta[0, :3])
    torch.testing.assert_close(result[0, 3:7].norm(), torch.tensor(1.0))
    assert bool((result[0, 7:] <= fk.joint_upper).all())
    assert bool((result[0, 7:] >= fk.joint_lower).all())


def test_active_and_inactive_residuals_use_actual_pair_includemargin(monkeypatch) -> None:
    instance = NewtonContactProjector(lambda q, geometry: {}, activation_clearance_fraction=0.05)
    q = canonical_q(instance.fk)
    pair = {
        "part": 0,
        "surface": 0,
        "dist": 0.006,
        "includemargin": 0.02,
    }
    rows = [[(pair, q.new_tensor(0.006))]]
    observed = {
        "surface_catalog": [{"surface": 0, "normal_w": [0.0, 0.0, 1.0], "plane_offset": 0.0}],
        "full_robot_separation": [{}],
    }
    monkeypatch.setattr(
        projector_module,
        "full_body_violation",
        lambda *args, **kwargs: (q.new_tensor([-1.0]), q.new_zeros((1,))),
    )
    anchor = q.new_zeros((6, 3))
    active = torch.tensor((True, False, False, False, False, False))
    surface = torch.zeros(6, dtype=torch.long)
    active_residual, _ = instance._residual(
        q, rows, observed, active, surface, anchor, patch=True
    )
    # The active interior target is 5% of this pair's 2 cm Newton margin.
    torch.testing.assert_close(active_residual, torch.tensor([(0.006 - 0.001) / 0.02]))

    inactive_residual, _ = instance._residual(
        q, rows, observed, torch.zeros(6, dtype=torch.bool), surface, anchor, patch=True
    )
    # An inactive pair is pushed to its own 2 cm activation boundary.
    torch.testing.assert_close(inactive_residual, torch.tensor([(0.02 - 0.006) / 0.02]))


def test_missing_candidate_uses_realized_configured_margin(monkeypatch) -> None:
    instance = NewtonContactProjector(lambda q, geometry: {})
    q = canonical_q(instance.fk)
    monkeypatch.setattr(
        instance.approach_geometry,
        "approach_residual",
        lambda *args, **kwargs: args[-1].reshape(1),
    )
    monkeypatch.setattr(
        projector_module,
        "full_body_violation",
        lambda *args, **kwargs: (q.new_tensor([-1.0]), q.new_zeros((1,))),
    )
    observed = {
        "surface_catalog": [
            {"surface": 1, "normal_w": [0.0, 0.0, 1.0], "plane_offset": 1.0}
        ],
        "configured_terrain_includemargins": [0.02],
        "full_robot_separation": [{}],
    }
    active = torch.tensor((True, False, False, False, False, False))
    surface = torch.tensor((1, -1, -1, -1, -1, -1))
    residual, missing = instance._residual(
        q, [[]], observed, active, surface, q.new_zeros((6, 3)), patch=False
    )
    assert missing == 1
    torch.testing.assert_close(residual, torch.tensor([1.0]))


def test_new_contact_geometry_keeps_authoritative_mesh_and_sphere_shapes() -> None:
    geometry = CanonicalContactCollisionGeometry()
    right_hand = [shape for shape in geometry.shapes if shape.part == 3]
    mesh = next(shape for shape in right_hand if shape.link_name == "right_sphere_hand_link")
    tip = next(shape for shape in right_hand if shape.link_name == "right_sphere_hand_tip_link")
    assert mesh.kind == "mesh"
    assert mesh.local_points is not None and len(mesh.local_points) == 23131
    assert tip.kind == "sphere"
    assert tip.radius == 0.001


def test_penetrating_projection_returns_verified_safe_fallback(monkeypatch) -> None:
    instance = NewtonContactProjector(lambda q, geometry: {}, iterations=1)
    nominal = canonical_q(instance.fk)[0]
    safe = nominal.clone()
    safe[0] = 1.0

    def observed(penetration: float) -> dict:
        witness = None if penetration == 0.0 else {"dist": -penetration}
        return {
            "pairs": [[]],
            "surface_catalog": [],
            "full_robot_separation": [{"worst_terrain": witness, "worst_self": None}],
        }

    monkeypatch.setattr(
        instance,
        "_query_rows",
        lambda q: ([[]], observed(0.0 if float(q[0, 0].detach()) == 1.0 else 0.03)),
    )
    monkeypatch.setattr(
        instance,
        "_residual",
        lambda q, *args, **kwargs: (q.new_zeros((0,)), 0),
    )
    monkeypatch.setattr(
        projector_module,
        "_selected_actual",
        lambda _observed: (torch.zeros(6, dtype=torch.bool).numpy(), -torch.ones(6, dtype=torch.long).numpy()),
    )
    result = instance.project(
        nominal,
        torch.zeros(6, dtype=torch.bool),
        -torch.ones(6, dtype=torch.long),
        torch.zeros((6, 3)),
        patch=False,
        safe_qpos=safe,
    )
    torch.testing.assert_close(result.qpos, safe)
    assert result.maximum_penetration_m == 0.0
    assert result.attempted_maximum_penetration_m == 0.03
    assert result.safety_fallback_used
    assert not result.converged


def test_penetrating_projection_without_fallback_is_rejected(monkeypatch) -> None:
    instance = NewtonContactProjector(lambda q, geometry: {}, iterations=1)
    nominal = canonical_q(instance.fk)[0]
    penetrating = {
        "pairs": [[]],
        "surface_catalog": [],
        "full_robot_separation": [
            {"worst_terrain": {"dist": -0.01}, "worst_self": None}
        ],
    }
    monkeypatch.setattr(instance, "_query_rows", lambda q: ([[]], penetrating))
    monkeypatch.setattr(
        instance,
        "_residual",
        lambda q, *args, **kwargs: (q.new_zeros((0,)), 0),
    )
    monkeypatch.setattr(
        projector_module,
        "_selected_actual",
        lambda _observed: (torch.zeros(6, dtype=torch.bool).numpy(), -torch.ones(6, dtype=torch.long).numpy()),
    )
    try:
        instance.project(
            nominal,
            torch.zeros(6, dtype=torch.bool),
            -torch.ones(6, dtype=torch.long),
            torch.zeros((6, 3)),
            patch=False,
        )
    except RuntimeError as error:
        assert "no verified safe fallback" in str(error)
    else:
        raise AssertionError("penetrating pose was returned without a safe fallback")
