MECHANISM: Quadratically tuned mild label smoothing

HYPOTHESIS: Label smoothing of 0.026 will exceed 9,324 correct predictions by preserving beneficial regularization while reducing the over-regularization indicated at 0.05.

INTENDED_EDIT: Reduce training-only cross-entropy label smoothing from 0.03 to 0.026 while retaining the successful AdamW beta2=0.99 configuration.

EVIDENCE: Smoothing increased correctness from 9,311 without smoothing to 9,318 at 0.03, then decreased it to 9,312 at 0.05; a quadratic fit to those measured results peaks near 0.026, and prior 0.026 attempts timed out without contradictory metrics.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.026)
>>>>>>> REPLACE