MECHANISM: Front-loaded cosine regularization taper

HYPOTHESIS: A cosine taper for dropout and label smoothing will exceed 9,330 correct predictions by preserving total regularization exposure while reducing late objective drift for the EMA classifier.

INTENDED_EDIT: Replace the second-half linear regularization decay with a cosine decay that is stronger early and weaker near convergence.

EVIDENCE: Classifier-specific EMA at 0.04 improved correctness to 9,330, indicating sensitivity to the annealed terminal objective, while faster EMA rates regressed; reducing late regularization without further shortening the EMA horizon targets that lag.

<<<<<<< SEARCH
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
    model.classifier[3].p = 0.15 * (1.0 - dropout_decay)
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    anneal_progress = max(progress - 0.50, 0.0) / 0.50
    regularization_multiplier = 0.5 * (
        1.0 + math.cos(math.pi * anneal_progress)
    )
    model.classifier[3].p = 0.15 * regularization_multiplier
    logits = model(images)
    label_smoothing = 0.02 * regularization_multiplier
>>>>>>> REPLACE