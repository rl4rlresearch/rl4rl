MECHANISM: Mild label-smoothing regularization

HYPOTHESIS: Applying 2% label smoothing will exceed 9,240 correct predictions by reducing overconfident fitting while preserving nearly all hard-label learning during the fixed two-pass exposure.

INTENDED_EDIT: Add 0.02 label smoothing to the existing cross-entropy loss without changing architecture, runtime-intensive computation, or the strongest verified 12.5%-floor schedule.

EVIDENCE: The current unsmoothed design is strongest at 9,240 correct; nearby schedule changes fell to 9,232 and 9,219, while architectural additions either regressed or timed out, motivating an isolated low-cost loss regularizer.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE