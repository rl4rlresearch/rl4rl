MECHANISM: Late-ramped horizontal-flip consistency regularization

HYPOTHESIS: Adding a modest symmetric KL penalty between original and flipped predictions late in training will exceed 9,283 correct predictions by making the two views agree before their logits are averaged at validation.

INTENDED_EDIT: Preserve the verified architecture, optimizer, augmentation, schedule, and TTA while adding a symmetric flip-consistency loss that ramps from zero to 0.10.

EVIDENCE: The 9,283-correct baseline already trains and evaluates a horizontal-flip logit ensemble, while optimizer, schedule, translation, representation, and probability-averaging changes regressed; explicit agreement between the paired predictions remains untested.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
=======
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
    original_log_probs = F.log_softmax(original_logits, dim=1)
    flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probs,
            flipped_log_probs.exp().detach(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probs,
            original_log_probs.exp().detach(),
            reduction="batchmean",
        )
    )
    consistency_weight = 0.05 * (
        1.0 - math.cos(math.pi * progress)
    )
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
        + consistency_weight * consistency_loss
    )
>>>>>>> REPLACE