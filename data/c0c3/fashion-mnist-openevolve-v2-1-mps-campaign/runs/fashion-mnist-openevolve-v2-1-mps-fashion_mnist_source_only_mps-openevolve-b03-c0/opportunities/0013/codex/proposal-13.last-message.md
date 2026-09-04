MECHANISM: Moderate target-distribution regularization

HYPOTHESIS: Increasing label smoothing from 0.02 to 0.04 will exceed 9,291 correct predictions by extending the demonstrated accuracy benefit of soft targets while retaining the successful 249,934-parameter architecture.

INTENDED_EDIT: Increase cross-entropy label smoothing from 0.02 to 0.04; leave the model, batch size, optimizer, and schedule unchanged.

EVIDENCE: Removing 0.02 label smoothing reduced correctness from 9,290 to 9,270 despite improving cross-entropy, showing that soft-target regularization materially benefits the primary objective; this tests whether a moderately stronger value improves it further.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
>>>>>>> REPLACE