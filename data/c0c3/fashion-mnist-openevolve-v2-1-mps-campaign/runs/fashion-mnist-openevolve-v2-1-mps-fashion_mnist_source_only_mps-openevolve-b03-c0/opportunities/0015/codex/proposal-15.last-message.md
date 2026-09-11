MECHANISM: Local label-smoothing optimum refinement

HYPOTHESIS: Setting label smoothing to 0.038 will exceed 9,316 correct predictions by targeting the empirical peak implied by results at 0.02, 0.04, and 0.06.

INTENDED_EDIT: Reduce cross-entropy label smoothing from 0.04 to 0.038 while preserving the strongest architecture and training configuration.

EVIDENCE: Correct predictions rose from 9,291 at 0.02 smoothing to 9,316 at 0.04, then fell to 9,279 at 0.06; quadratic interpolation places the local optimum near 0.038.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.038)
>>>>>>> REPLACE