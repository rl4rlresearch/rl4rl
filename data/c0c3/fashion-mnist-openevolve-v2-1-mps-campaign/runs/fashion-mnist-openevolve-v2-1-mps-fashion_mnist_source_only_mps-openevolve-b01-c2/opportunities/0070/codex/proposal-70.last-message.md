MECHANISM: Ramp-weighted bidirectional flip consistency

HYPOTHESIS: Adding mild symmetric KL agreement between original and flipped predictions will exceed 9,330 correct by strengthening the flip invariance already responsible for the best verified result.

INTENDED_EDIT: Add a parameter-free, progressively weighted consistency term to the existing ensemble and per-view loss without changing forwards, architecture, or evaluation.

EVIDENCE: Symmetric flip training and inference achieved the best verified 9,330 correct; this directly reinforces that proven symmetry while avoiding the extra image operations associated with timed-out translation experiments.

<<<<<<< SEARCH
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
=======
    classification_loss = (
        ensemble_weight * ensemble_loss
        + (1.0 - ensemble_weight) * view_loss
    )
    consistency_loss = 0.5 * (
        F.kl_div(
            F.log_softmax(original_logits, dim=1),
            F.softmax(flipped_logits.detach(), dim=1),
            reduction="batchmean",
        )
        + F.kl_div(
            F.log_softmax(flipped_logits, dim=1),
            F.softmax(original_logits.detach(), dim=1),
            reduction="batchmean",
        )
    )
    return classification_loss + 0.05 * progress * consistency_loss
>>>>>>> REPLACE