MECHANISM: Stronger fixed target smoothing with proven ensemble calibration

HYPOTHESIS: Increasing fixed label smoothing from 0.02 to 0.03 will exceed 9,330 correct predictions, while the verified 1.15 evaluation scale will minimize cross-entropy if correctness ties.

INTENDED_EDIT: Raise label smoothing for both ensemble and per-view losses to 0.03 and restore the best verified symmetric-ensemble scale of 1.15.

EVIDENCE: Annealing smoothing toward zero reduced correctness from 9,330 to 9,312, indicating soft targets support accuracy; the prior 0.03 verification timed out without negative model evidence, while scale 1.15 preserved 9,330 predictions and improved cross-entropy to 0.196639.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE

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
        label_smoothing=0.03,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.03)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.03)
    )
>>>>>>> REPLACE