MECHANISM: Quadratically tuned mild label smoothing

HYPOTHESIS: Label smoothing of 0.026 will exceed 9,318 correct predictions by retaining the gain at 0.03 while avoiding the over-regularization observed at 0.05.

INTENDED_EDIT: Reduce training-only cross-entropy label smoothing from 0.03 to 0.026; leave architecture, optimization, augmentation, and inference unchanged.

EVIDENCE: Smoothing improved correctness from 9,311 at the unsmoothed baseline to 9,318 at 0.03, then fell to 9,312 at 0.05; these measured points place the estimated optimum near 0.026, whose prior verification timed out without contradictory accuracy evidence.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.026)
>>>>>>> REPLACE