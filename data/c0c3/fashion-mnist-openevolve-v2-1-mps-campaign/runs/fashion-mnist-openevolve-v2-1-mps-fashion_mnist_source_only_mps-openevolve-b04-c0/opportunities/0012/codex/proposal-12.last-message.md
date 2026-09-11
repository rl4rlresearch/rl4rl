MECHANISM: Upper-bracket auxiliary-loss weight refinement

HYPOTHESIS: Increasing total individual-view supervision from 10% to 12.5% will exceed 9,341 correct predictions by strengthening constituent-view accuracy without approaching the 20% setting that regressed to 9,330.

INTENDED_EDIT: Weight the ensemble loss at 87.5% and each constituent-view loss at 6.25%, preserving all other training and evaluation behavior.

EVIDENCE: Total individual-view weights of 7.5%, 10%, and 20% produced 9,331, 9,341, and 9,330 correct respectively; testing 12.5% most directly brackets the successful 10% setting on its unexplored upper side.

<<<<<<< SEARCH
    return (
        0.9 * ensemble_loss
        + 0.05 * original_loss
        + 0.05 * flipped_loss
    )
=======
    return (
        0.875 * ensemble_loss
        + 0.0625 * original_loss
        + 0.0625 * flipped_loss
    )
>>>>>>> REPLACE