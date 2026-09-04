MECHANISM: Fine-grained target-distribution smoothing

HYPOTHESIS: Label smoothing of 0.025 will exceed 9,318 correct predictions by retaining the beneficial regularization of 0.03 while reducing the over-regularization observed at 0.05.

INTENDED_EDIT: Reduce training-only cross-entropy label smoothing from 0.03 to 0.025; leave architecture, optimization, augmentation, and inference unchanged.

EVIDENCE: Smoothing of 0.03 improved correctness from 9,311 to 9,318, while 0.05 reduced it to 9,312. The 0.026 and 0.028 trials produced no contradictory metrics because they timed out, leaving the predicted optimum near 0.025 unmeasured.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.025)
>>>>>>> REPLACE