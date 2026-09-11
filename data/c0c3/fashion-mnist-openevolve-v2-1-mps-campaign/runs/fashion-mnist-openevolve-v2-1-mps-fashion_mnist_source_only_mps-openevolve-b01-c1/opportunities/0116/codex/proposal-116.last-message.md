MECHANISM: Ramped teacher-view consistency regularization

HYPOTHESIS: Aligning each translated and flipped prediction with their detached probability ensemble will exceed 9,257 correct predictions by reducing harmful view disagreement without changing the proven transformations or architecture.

INTENDED_EDIT: Retain the existing supervised losses and add a cosine-ramped KL consistency loss between all six view predictions and their 0.9/0.1 full/central probability mixture.

EVIDENCE: Center-weighted aggregation showed that predictions vary meaningfully across geometric views, while changing the crop distribution or adding rotation regressed; this targets disagreement among the existing successful views instead.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=0.02,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * view_loss
        + ensemble_weight * ensemble_loss
    )
=======
    ensemble_logits = (
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central)
    )
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    supervised_loss = (
        (1.0 - ensemble_weight) * view_loss
        + ensemble_weight * ensemble_loss
    )

    all_view_logits = torch.cat(
        (
            logits_a,
            logits_b,
            flipped_a,
            flipped_b,
            central_logits,
            flipped_central,
        ),
        dim=0,
    )
    (
        probability_a,
        probability_b,
        probability_flipped_a,
        probability_flipped_b,
        probability_central,
        probability_flipped_central,
    ) = F.softmax(all_view_logits, dim=-1).chunk(6, dim=0)
    consistency_target = (
        0.225
        * (
            probability_a
            + probability_b
            + probability_flipped_a
            + probability_flipped_b
        )
        + 0.05
        * (probability_central + probability_flipped_central)
    ).detach()
    consistency_loss = F.kl_div(
        F.log_softmax(all_view_logits, dim=-1),
        consistency_target.repeat(6, 1),
        reduction="batchmean",
    )
    consistency_weight = 0.06 * (
        1.0 - math.cos(math.pi * progress)
    )
    return supervised_loss + consistency_weight * consistency_loss
>>>>>>> REPLACE