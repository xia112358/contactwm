import torch

import generator.full_interaction_predictor as predictor_module
from generator.full_interaction_predictor import (
    FullInteractionPrediction,
    FullHeightmapInteractionPredictor,
    full_interaction_objective,
)


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    q = torch.zeros((batch, 36))
    q[:, 3] = 1.0
    return {
        "current_q": q,
        "current_contact": torch.zeros((batch, 6), dtype=torch.bool),
        "current_anchor": torch.zeros((batch, 6, 3)),
        "current_surface": torch.full((batch, 6), -1, dtype=torch.long),
        "heightmap": torch.zeros((batch, 71, 61)),
    }


def test_inference_pose_uses_its_own_hard_contact_plan() -> None:
    model = FullHeightmapInteractionPredictor(width=24, layers=1).eval()
    with torch.no_grad():
        model.role_head.weight.zero_()
        model.role_head.bias.copy_(torch.tensor((0.0, 1.0, 0.0, 0.0)))
        model.next_surface_head.weight.zero_()
        model.next_surface_head.bias.copy_(torch.tensor((0.0, 1.0)))
    prediction = model(**inputs())
    assert bool(prediction.conditioned_contact.all())
    assert bool((prediction.conditioned_surface == 1).all())
    assert torch.equal(prediction.conditioned_contact, prediction.contact)
    assert torch.equal(prediction.conditioned_surface, prediction.surface)
    assert not bool(prediction.teacher_conditioned.any())


def test_teacher_conditioning_is_explicit_and_per_sample() -> None:
    model = FullHeightmapInteractionPredictor(width=24, layers=1).eval()
    batch = inputs()
    role = torch.zeros((2, 6), dtype=torch.long)
    role[0, 0] = 1
    surface = torch.full((2, 6), -1, dtype=torch.long)
    surface[0, 0] = 0
    prediction = model(
        **batch,
        teacher_role=role,
        teacher_surface=surface,
        teacher_mask=torch.tensor((True, False)),
    )
    assert prediction.conditioned_contact[0].tolist() == [True, False, False, False, False, False]
    assert prediction.conditioned_surface[0, 0].item() == 0
    assert prediction.teacher_conditioned.tolist() == [True, False]


def test_model_has_no_binding_or_event_index_parameters() -> None:
    names = tuple(dict(FullHeightmapInteractionPredictor(width=24, layers=1).named_parameters()))
    assert not any("binding" in name or "event" in name for name in names)


def test_pose_realization_does_not_change_the_contact_decision() -> None:
    torch.manual_seed(11)
    model = FullHeightmapInteractionPredictor(width=24, layers=1)
    prediction = model(**inputs(1))
    prediction.qpos.square().sum().backward()
    assert model.role_head.weight.grad is None
    assert model.next_surface_head.weight.grad is None


def test_wrong_plan_cannot_rewrite_pose_with_realization_or_safety(monkeypatch) -> None:
    model = FullHeightmapInteractionPredictor(width=24, layers=1)
    role_logits = torch.zeros((1, 6, 4))
    surface_logits = torch.zeros((1, 6, 2))
    prediction = FullInteractionPrediction(
        qpos=inputs(1)["current_q"],
        role_logits=role_logits,
        surface_logits=surface_logits,
        conditioned_contact=torch.zeros((1, 6), dtype=torch.bool),
        conditioned_surface=torch.zeros((1, 6), dtype=torch.long),
        teacher_conditioned=torch.zeros(1, dtype=torch.bool),
    )
    target = {
        "role": torch.ones((1, 6), dtype=torch.long),
        "planned_surface": torch.zeros((1, 6), dtype=torch.long),
    }

    def fake_pose_objective(*_args, **_kwargs):
        imitation = torch.tensor([100.0])
        approach = torch.tensor([13.0])
        realization = torch.tensor([11.0])
        safety = torch.tensor([7.0])
        return imitation + realization + safety, {
            "imitation_loss": imitation,
            "missing_pair_approach_loss": approach,
            "own_plan_realization_loss": realization,
            "safety_loss": safety,
        }

    monkeypatch.setattr(
        predictor_module, "conditioned_pose_objective", fake_pose_objective
    )
    loss, metrics = full_interaction_objective(model, prediction, target, {})
    assert metrics["pose_supervised"].item() == 0.0
    torch.testing.assert_close(loss, metrics["role_loss"] + metrics["surface_loss"])
