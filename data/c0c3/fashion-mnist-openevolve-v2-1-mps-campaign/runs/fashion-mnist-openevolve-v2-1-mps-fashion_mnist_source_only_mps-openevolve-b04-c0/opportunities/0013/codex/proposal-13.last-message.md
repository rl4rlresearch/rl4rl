MECHANISM: Upper-bracket auxiliary-loss refinement

HYPOTHESIS: Increasing total individual-view supervision from 12.5% to 15% will exceed 9,350 correct predictions by further strengthening constituent views while remaining below the 20% setting that regressed.

INTENDED_EDIT: Weight the ensemble loss at 85% and each constituent-view loss at 7.5%, preserving all other behavior.

EVIDENCE: Individual-view weights of 10%, 12.5%, and 20% produced 9,341, 9,350, and 9,330 correct respectively, so 15% directly tests the unexplored interval between the current best and the known-regressive setting.

<<<<<<< SEARCH
    return (
        0.875 * ensemble_loss
        + 0.0625 * original_loss
        + 0.0625 * flipped_loss
    )
=======
    return (
        0.85 * ensemble_loss
        + 0.075 * original_loss
        + 0.075 * flipped_loss
    )
>>>>>>> REPLACE