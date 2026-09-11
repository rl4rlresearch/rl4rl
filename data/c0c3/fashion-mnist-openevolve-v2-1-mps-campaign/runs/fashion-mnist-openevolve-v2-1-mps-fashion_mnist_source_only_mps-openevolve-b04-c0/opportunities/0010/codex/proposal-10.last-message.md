MECHANISM: Stronger auxiliary per-view supervision

HYPOTHESIS: Increasing total individual-view loss weight from 10% to 20% will exceed 9,341 correct predictions by further reducing orientation-specific errors while keeping the flip-ensemble objective dominant.

INTENDED_EDIT: Weight the ensemble loss at 80% and each constituent-view loss at 10%, preserving all other training and evaluation behavior.

EVIDENCE: Adding 10% individual-view supervision improved validation_correct from 9,335 to 9,341; increasing that successful auxiliary signal is the most direct test of whether stronger constituent accuracy further improves the ensemble.

<<<<<<< SEARCH
    return (
        0.9 * ensemble_loss
        + 0.05 * original_loss
        + 0.05 * flipped_loss
    )
=======
    return (
        0.8 * ensemble_loss
        + 0.1 * original_loss
        + 0.1 * flipped_loss
    )
>>>>>>> REPLACE