MECHANISM: Local auxiliary-loss weight refinement

HYPOTHESIS: Reducing total individual-view supervision from 12.5% to 12% will exceed 9,350 correct predictions by moving toward the estimated peak between the 10% and 15% results.

INTENDED_EDIT: Weight the ensemble loss at 88% and each constituent-view loss at 6%, preserving all other training and evaluation behavior.

EVIDENCE: Total individual-view weights of 10%, 12.5%, and 15% produced 9,341, 9,350, and 9,332 correct respectively; this brackets the optimum near 12.5%, with the asymmetric decline suggesting a slightly lower weight.

<<<<<<< SEARCH
    return (
        0.875 * ensemble_loss
        + 0.0625 * original_loss
        + 0.0625 * flipped_loss
    )
=======
    return (
        0.88 * ensemble_loss
        + 0.06 * original_loss
        + 0.06 * flipped_loss
    )
>>>>>>> REPLACE