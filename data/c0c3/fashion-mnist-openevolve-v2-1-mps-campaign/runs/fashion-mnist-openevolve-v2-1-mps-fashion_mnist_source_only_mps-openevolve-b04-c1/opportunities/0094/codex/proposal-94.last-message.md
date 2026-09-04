MECHANISM: Earlier hard-label transition during EMA

HYPOTHESIS: Ending label smoothing at 80% progress will exceed 9,328 correct predictions by allowing more hard-label optimization during EMA collection while retaining early regularization.

INTENDED_EDIT: Decay label smoothing linearly from 0.02 at mid-training to zero at 80% progress, while preserving the existing dropout schedule.

EVIDENCE: Late label-smoothing annealing produced the strongest 9,328-correct design, and the accelerated quadratic-decay attempt timed out without negative accuracy evidence; this runtime-neutral edit directly retests earlier soft-target removal.

<<<<<<< SEARCH
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    smoothing_decay = min(
        max(progress - 0.50, 0.0) / 0.30,
        1.0,
    )
    label_smoothing = 0.02 * (1.0 - smoothing_decay)
>>>>>>> REPLACE