MECHANISM: Fine-grained target regularization

HYPOTHESIS: Reducing label smoothing from 0.05 to 0.04 will exceed 9,254 correct predictions by retaining beneficial regularization while reducing target underconfidence.

INTENDED_EDIT: Change only the cross-entropy label-smoothing coefficient from 0.05 to 0.04.

EVIDENCE: Label smoothing 0.05 outperformed both hard targets and 0.10, locating the useful region near mild smoothing; a small downward refinement is runtime-neutral and may improve both classification decisions and tie-breaking cross-entropy.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
>>>>>>> REPLACE