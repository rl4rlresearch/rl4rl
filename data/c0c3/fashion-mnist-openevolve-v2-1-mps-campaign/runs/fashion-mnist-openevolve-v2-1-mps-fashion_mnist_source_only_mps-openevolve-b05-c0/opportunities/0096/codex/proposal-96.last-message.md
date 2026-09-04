MECHANISM: Annealed label smoothing

HYPOTHESIS: Decaying label smoothing from 0.05 to zero will increase validation_correct above 9,258 by retaining early regularization while allowing sharper class boundaries late in the fixed two-epoch training budget.

INTENDED_EDIT: Replace constant label smoothing with a linear schedule that reaches zero at the final optimizer step.

EVIDENCE: Increasing evaluation-logit scale repeatedly lowered cross-entropy without changing the 9,258 predictions, indicating underconfident outputs; annealing the training-time smoothing directly addresses that underconfidence with negligible runtime overhead while targeting the primary accuracy objective.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    progress = step / max(total_steps, 1)
    smoothing = 0.05 * max(1.0 - progress, 0.0)
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE