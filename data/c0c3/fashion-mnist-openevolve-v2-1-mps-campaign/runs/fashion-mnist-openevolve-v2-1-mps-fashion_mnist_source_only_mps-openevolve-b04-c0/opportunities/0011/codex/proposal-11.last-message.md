MECHANISM: Bracketed auxiliary-loss weight refinement

HYPOTHESIS: Reducing total individual-view supervision from 10% to 7.5% will exceed 9,341 correct predictions by retaining its benefit while moving away from the 20% setting that regressed to 9,330.

INTENDED_EDIT: Weight the ensemble loss at 92.5% and each constituent-view loss at 3.75%, preserving all other behavior.

EVIDENCE: Total individual-view weights of 0%, 10%, and 20% produced 9,335, 9,341, and 9,330 correct respectively, indicating a local optimum near but slightly below 10%.

<<<<<<< SEARCH
    return (
        0.9 * ensemble_loss
        + 0.05 * original_loss
        + 0.05 * flipped_loss
    )
=======
    return (
        0.925 * ensemble_loss
        + 0.0375 * original_loss
        + 0.0375 * flipped_loss
    )
>>>>>>> REPLACE