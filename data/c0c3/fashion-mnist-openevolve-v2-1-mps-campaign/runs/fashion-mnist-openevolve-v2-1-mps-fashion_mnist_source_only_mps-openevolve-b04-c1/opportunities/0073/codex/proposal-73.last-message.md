MECHANISM: Conservative early dropout shutdown

HYPOTHESIS: Completing dropout annealing at 87.5% progress will exceed 9,328 correct predictions by providing a short dropout-free EMA consolidation phase while preserving the winning label-smoothing schedule.

INTENDED_EDIT: Decouple dropout decay from label-smoothing decay, ending dropout at 87.5% of training while label smoothing continues annealing to zero at the end.

EVIDENCE: Joint regularization annealing improved validation correct from 9,316 to 9,328, while every tested label-smoothing timing change reduced accuracy; the unresolved 75% dropout-shutdown attempt motivates a more conservative isolation of dropout timing.

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    regularization_decay = max(progress - 0.50, 0.0) / 0.50
    dropout_decay = min(
        max(progress - 0.50, 0.0) / 0.375,
        1.0,
    )
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - regularization_decay)
>>>>>>> REPLACE