MECHANISM: Flip-view Jensen–Shannon consistency regularization

HYPOTHESIS: Restoring the best mildly distance-stratified TTA and explicitly aligning original and flipped predictions will exceed 9,268 correct predictions by reducing view-specific errors that the increasingly dominant ensemble loss can otherwise conceal.

INTENDED_EDIT: Restore the verified 0.009375/0.00625/0.003125 radius-2 weights and add a small Jensen–Shannon consistency penalty between original and horizontally flipped predictions.

EVIDENCE: Mild radius-2 weighting achieved the best 9,268-correct result, while stronger weighting lost one prediction. Architecture, EMA, probability-averaging, and batch-size changes all failed; the existing loss shifts to 95% ensemble supervision late in training, motivating direct agreement regularization without changing capacity or exposure.

<<<<<<< SEARCH
                    if manhattan_distance == 2:
                        weight = 0.0125
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.0
=======
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
>>>>>>> REPLACE

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
    mean_probs = 0.5 * (
        original_log_probs.exp() + flipped_log_probs.exp()
    )
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probs,
            mean_probs,
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probs,
            mean_probs,
            reduction="batchmean",
        )
    )
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
        + 0.10 * consistency_loss
    )
>>>>>>> REPLACE