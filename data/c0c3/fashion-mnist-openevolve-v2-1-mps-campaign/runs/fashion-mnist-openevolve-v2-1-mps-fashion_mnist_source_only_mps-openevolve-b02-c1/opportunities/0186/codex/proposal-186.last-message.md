MECHANISM: Fine-grained target-distribution smoothing

HYPOTHESIS: Reducing label smoothing from 0.03 to 0.026 will exceed 9,318 correct predictions by retaining beneficial smoothing while avoiding the over-regularization observed at 0.05.

INTENDED_EDIT: Set training-only cross-entropy label smoothing to 0.026; leave architecture, optimization, augmentation, and inference unchanged.

EVIDENCE: Smoothing of 0.03 improved correctness from 9,311 to 9,318, while 0.05 fell to 9,312, placing the most promising region slightly below the successful 0.03 setting.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.026)
>>>>>>> REPLACE