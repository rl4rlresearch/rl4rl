MECHANISM: Conservative fixed label-smoothing refinement

HYPOTHESIS: Fixed 0.039 label smoothing will exceed 9,319 correct predictions by retaining nearly all successful regularization while slightly reducing the over-regularization indicated by the 0.05 result.

INTENDED_EDIT: Reduce fixed label smoothing from 0.04 to 0.039 while preserving architecture, optimizer, schedule, and calibrated evaluation temperature.

EVIDENCE: Fixed 0.05 smoothing reduced correctness to 9,299, while a much larger late-stage reduction reduced it to 9,307; these results bracket the verified 0.04 setting and motivate a small, fixed downward refinement.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.039)
>>>>>>> REPLACE