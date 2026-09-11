MECHANISM: Stronger target-distribution smoothing

HYPOTHESIS: Increasing label smoothing from 0.03 to 0.05 will exceed 9,318 correct predictions by further improving ambiguous-class decision boundaries, despite potentially increasing validation cross-entropy.

INTENDED_EDIT: Raise training-only cross-entropy label smoothing to 0.05 while leaving architecture, optimization, augmentation, and inference unchanged.

EVIDENCE: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, whereas a 0.15 true-class margin reduced it to 9,305; this supports testing a modestly stronger move in the successful smoothing direction.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE