MECHANISM: Stronger label-smoothing regularization

HYPOTHESIS: Increasing label smoothing from 0.05 to 0.07 will exceed 9,260 correct predictions by improving generalization during the fixed two-pass exposure budget.

INTENDED_EDIT: Increase only the training loss label-smoothing coefficient while preserving the verified optimizer, EMA, BatchNorm interpolation, and evaluation calibration.

EVIDENCE: Reducing label smoothing symmetrically from 0.05 to 0.03 lowered validation_correct from 9,260 to 9,242, indicating that weaker target regularization is harmful and motivating a controlled test in the opposite direction.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.07)
>>>>>>> REPLACE