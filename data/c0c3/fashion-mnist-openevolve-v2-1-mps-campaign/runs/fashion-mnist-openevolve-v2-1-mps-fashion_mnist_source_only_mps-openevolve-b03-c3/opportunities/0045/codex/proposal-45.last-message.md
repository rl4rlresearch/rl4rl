MECHANISM: Ramped flip-consistency regularization

HYPOTHESIS: Encouraging original and horizontally flipped views to agree will exceed 9,240 correct predictions by aligning training with the proven flip-averaged evaluation while preserving the successful architecture.

INTENDED_EDIT: Add a gradually ramped Jensen–Shannon-style consistency loss between paired-view predictions; retain the existing supervised loss, optimizer, augmentation, and calibrated evaluation logits.

EVIDENCE: Architecture gating and EMA reduced correctness to 9,225 and 9,226, respectively, whereas the baseline repeatedly achieved 9,240; this motivates a parameter-free change targeting the evaluation invariance already encoded by paired training and flip averaging.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
=======
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    supervised_loss = (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )

    consensus_probabilities = (
        0.5
        * (
            F.softmax(original_logits, dim=1)
            + F.softmax(flipped_logits, dim=1)
        )
    ).detach()
    consistency_loss = 0.5 * (
        F.kl_div(
            F.log_softmax(original_logits, dim=1),
            consensus_probabilities,
            reduction="batchmean",
        )
        + F.kl_div(
            F.log_softmax(flipped_logits, dim=1),
            consensus_probabilities,
            reduction="batchmean",
        )
    )
    consistency_weight = 0.05 * (
        1.0 - math.cos(math.pi * progress)
    )
    return supervised_loss + consistency_weight * consistency_loss
>>>>>>> REPLACE