MECHANISM: Auxiliary per-view supervision for flip ensembles

HYPOTHESIS: Blending 10% individual-view cross-entropy into the paired-flip ensemble objective will exceed 9,335 correct predictions by discouraging orientation-specific errors while retaining the validated ensemble predictor.

INTENDED_EDIT: Preserve the architecture, schedule, and EMA behavior while training each original and reflected view with a small auxiliary loss alongside the averaged-logit loss.

EVIDENCE: Paired-flip ensemble training improved validation_correct from 9,302 to 9,322, demonstrating that flip-paired optimization is effective; lightly supervising both constituent predictions is the most direct refinement of that successful mechanism.

<<<<<<< SEARCH
    ensemble_logits = 0.5 * (logits + flipped_logits)
    return F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
=======
    ensemble_logits = 0.5 * (logits + flipped_logits)
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    original_loss = F.cross_entropy(
        logits,
        labels,
        label_smoothing=0.02,
    )
    flipped_loss = F.cross_entropy(
        flipped_logits,
        labels,
        label_smoothing=0.02,
    )
    return (
        0.9 * ensemble_loss
        + 0.05 * original_loss
        + 0.05 * flipped_loss
    )
>>>>>>> REPLACE