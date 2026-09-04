MECHANISM: Late hard-label consolidation

HYPOTHESIS: Completing label-smoothing annealing at 75% progress will exceed 9,328 correct predictions by allowing a longer hard-label optimization phase before training ends.

INTENDED_EDIT: Anneal label smoothing from 0.02 to zero between 50% and 75% of training while preserving the successful dropout schedule.

EVIDENCE: Annealing label smoothing through the end of training improved validation correct from 9,323 to 9,328; the prior 75%-completion verification timed out rather than disproving this low-cost extension.

<<<<<<< SEARCH
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
=======
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing_decay = min(
        max(progress - 0.50, 0.0) / 0.25,
        1.0,
    )
    label_smoothing = 0.02 * (1.0 - label_smoothing_decay)
    return F.cross_entropy(
>>>>>>> REPLACE