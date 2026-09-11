MECHANISM: Conservative hard-label consolidation

HYPOTHESIS: Completing label-smoothing annealing at 87.5% progress will exceed 9,328 correct predictions by adding a sustained hard-label phase without the larger schedule change of the unresolved 75% design.

INTENDED_EDIT: Anneal label smoothing from 0.02 to zero between 50% and 87.5% of training while preserving the successful dropout schedule.

EVIDENCE: Annealing label smoothing through training’s end improved validation correct from 9,323 to 9,328; the more aggressive 75%-completion attempts timed out, motivating this low-cost intermediate schedule.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
=======
    smoothing_decay = min(
        max(progress - 0.50, 0.0) / 0.375,
        1.0,
    )
    label_smoothing = 0.02 * (1.0 - smoothing_decay)
    return F.cross_entropy(
>>>>>>> REPLACE