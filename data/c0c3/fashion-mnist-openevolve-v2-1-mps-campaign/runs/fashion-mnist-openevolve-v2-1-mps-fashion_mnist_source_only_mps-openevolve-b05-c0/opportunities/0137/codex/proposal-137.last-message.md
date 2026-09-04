MECHANISM: Reduced label smoothing for stronger class separation

HYPOTHESIS: Lowering label smoothing from 0.05 to 0.03 will increase validation_correct above 9,260 by strengthening correct-class gradients while retaining moderate regularization.

INTENDED_EDIT: Reduce only the training loss label-smoothing coefficient; preserve the verified EMA, BatchNorm-buffer mixture, and 1.4164 evaluation calibration.

EVIDENCE: The 1.4164 multiplier minimizes cross-entropy only by sharpening logits and cannot change predictions, while BatchNorm and reflection changes lost accuracy; the optimal multiplier above 1 also indicates underconfident logits, motivating less target smoothing as an orthogonal accuracy-focused change.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE