MECHANISM: Stronger soft-target regularization

HYPOTHESIS: Increasing label smoothing from 0.04 to 0.06 will exceed 9,316 correct predictions by continuing the verified gain from moderately stronger soft targets without changing the successful architecture or optimization.

INTENDED_EDIT: Increase cross-entropy label smoothing from 0.04 to 0.06.

EVIDENCE: On the same 249,934-parameter model, increasing label smoothing from 0.02 to 0.04 improved validation correctness from 9,291 to 9,316, while removing smoothing previously reduced correctness.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.06)
>>>>>>> REPLACE