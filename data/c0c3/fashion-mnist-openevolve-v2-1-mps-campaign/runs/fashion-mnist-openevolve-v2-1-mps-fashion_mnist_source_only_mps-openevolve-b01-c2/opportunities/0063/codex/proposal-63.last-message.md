MECHANISM: Moderately stronger fixed target smoothing

HYPOTHESIS: Increasing label smoothing from 0.02 to 0.025 will exceed 9,330 correct predictions without increasing runtime, while retaining the verified 1.15 evaluation calibration.

INTENDED_EDIT: Raise smoothing for both ensemble and per-view cross-entropy losses to 0.025; preserve all other architecture, optimization, and inference settings.

EVIDENCE: Annealing smoothing toward zero reduced correctness from 9,330 to 9,312, supporting soft targets for accuracy; the 0.03 trial timed out without negative model evidence, so 0.025 tests the indicated direction more conservatively.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.025,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.025)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.025)
    )
>>>>>>> REPLACE