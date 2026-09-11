MECHANISM: Fine-grained target-distribution smoothing

HYPOTHESIS: Reducing label smoothing from 0.03 to 0.028 will exceed 9,318 correct predictions by retaining the demonstrated smoothing benefit while modestly reducing over-regularization.

INTENDED_EDIT: Set training-only cross-entropy label smoothing to 0.028, leaving architecture, optimization, augmentation, and inference unchanged.

EVIDENCE: Label smoothing of 0.03 improved correctness from 9,311 to 9,318, while increasing it to 0.05 reduced correctness to 9,312; the 0.026 trial produced no contradictory metrics because it timed out, motivating a smaller downward refinement.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.028)
>>>>>>> REPLACE