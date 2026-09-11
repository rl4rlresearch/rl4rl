MECHANISM: Increased target smoothing regularization

HYPOTHESIS: Increasing label smoothing from 0.05 to 0.06 will exceed 9,260 correct predictions by improving generalization within the fixed exposure budget.

INTENDED_EDIT: Change only the cross-entropy label-smoothing coefficient, preserving architecture, optimizer, EMA, runtime, and evaluation calibration.

EVIDENCE: Lowering smoothing to 0.04 reduced validation_correct from 9,260 to 9,246 and worsened cross-entropy, motivating a symmetric probe above the incumbent 0.05.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.06)
>>>>>>> REPLACE