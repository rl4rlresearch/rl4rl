MECHANISM: Late flip-consistency self-distillation

HYPOTHESIS: Adding a modest late-ramped consistency penalty between each image and its flipped counterpart will exceed 9,247 correct predictions by reducing validation-ensemble disagreement that pair-label supervision does not directly penalize.

INTENDED_EDIT: Preserve the proven architecture, augmentation, optimizer, and losses while adding an evaluation-aligned KL penalty that distills each flip pair toward its detached mean prediction, ramping from zero to 0.08.

EVIDENCE: Ramping flip-pair allocation improved fixed supervision from 9,246 to 9,247 correct, while widening or narrowing that allocation regressed to 9,245 and 9,243; this tests complementary direct agreement within the same successful flip geometry.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * view_loss
        + ensemble_weight * ensemble_loss
    )
=======
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    supervised_loss = (
        (1.0 - ensemble_weight) * view_loss
        + ensemble_weight * ensemble_loss
    )

    def flip_consistency(
        original: torch.Tensor,
        flipped: torch.Tensor,
    ) -> torch.Tensor:
        mean_probability = (
            0.5
            * (
                F.softmax(original, dim=1)
                + F.softmax(flipped, dim=1)
            )
        ).detach()
        return 0.5 * (
            F.kl_div(
                F.log_softmax(original, dim=1),
                mean_probability,
                reduction="batchmean",
            )
            + F.kl_div(
                F.log_softmax(flipped, dim=1),
                mean_probability,
                reduction="batchmean",
            )
        )

    full_consistency = 0.5 * (
        flip_consistency(logits_a, flipped_a)
        + flip_consistency(logits_b, flipped_b)
    )
    central_consistency = flip_consistency(
        central_logits,
        flipped_central,
    )
    consistency_loss = (
        0.9 * full_consistency + 0.1 * central_consistency
    )
    consistency_weight = 0.04 * (
        1.0 - math.cos(math.pi * progress)
    )
    return supervised_loss + consistency_weight * consistency_loss
>>>>>>> REPLACE