MECHANISM: Late-ramped symmetric flip-consistency regularization

HYPOTHESIS: Explicitly reducing disagreement between paired original and flipped predictions late in training will exceed 9,284 correct predictions while preserving the validated logit-space TTA.

INTENDED_EDIT: Add a symmetric detached-target KL loss between existing paired logits, ramp its weight from zero to 0.20, and use the best verified evaluation scale of 1.16727.

EVIDENCE: Arithmetic probability TTA reduced accuracy to 9,280, so the validated logit ensemble should remain intact; the existing paired training path supplies both views but does not directly penalize their disagreement, enabling this consistency test without architectural or computational expansion.

<<<<<<< SEARCH
        return 1.1645 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
=======
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)

    original_log_prob = F.log_softmax(original_logits, dim=1)
    flipped_log_prob = F.log_softmax(flipped_logits, dim=1)
    original_prob = original_log_prob.exp()
    flipped_prob = flipped_log_prob.exp()
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_prob,
            flipped_prob.detach(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_prob,
            original_prob.detach(),
            reduction="batchmean",
        )
    )
    consistency_weight = 0.10 * (
        1.0 - math.cos(math.pi * progress)
    )
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
        + consistency_weight * consistency_loss
    )
>>>>>>> REPLACE