MECHANISM: Late hard-label consolidation

HYPOTHESIS: Completing label-smoothing annealing at 75% progress will exceed 9,328 correct predictions by providing a substantial hard-label optimization phase while retaining early regularization.

INTENDED_EDIT: Keep the successful dropout schedule unchanged, but anneal label smoothing from 0.02 to zero between 50% and 75% of training.

EVIDENCE: Annealing label smoothing only at the very end increased validation correct from 9,323 to 9,328; reaching zero earlier directly tests whether a longer bias-free phase provides further gains.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    label_smoothing_decay = min(
        max(progress - 0.50, 0.0) / 0.25,
        1.0,
    )
    label_smoothing = 0.02 * (1.0 - label_smoothing_decay)
>>>>>>> REPLACE