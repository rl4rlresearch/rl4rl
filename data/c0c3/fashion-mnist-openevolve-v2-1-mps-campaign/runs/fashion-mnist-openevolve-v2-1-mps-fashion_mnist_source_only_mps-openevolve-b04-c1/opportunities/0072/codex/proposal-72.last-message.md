MECHANISM: Early dropout shutdown during EMA consolidation

HYPOTHESIS: Completing dropout annealing at 75% progress while retaining the winning label-smoothing schedule will exceed 9,328 correct predictions by removing late stochastic feature masking without weakening soft-target regularization.

INTENDED_EDIT: Decouple dropout and label-smoothing decay, ending dropout at 75% while continuing label smoothing linearly to zero at training’s end.

EVIDENCE: Joint regularization annealing improved 9,316 to 9,328 correct, while changing label-smoothing timing reduced performance to 9,312 and 9,323; preserving that schedule while isolating faster dropout removal is the most direct next test.

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    dropout_decay = min(
        max(progress - 0.50, 0.0) / 0.25,
        1.0,
    )
    smoothing_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - smoothing_decay)
>>>>>>> REPLACE