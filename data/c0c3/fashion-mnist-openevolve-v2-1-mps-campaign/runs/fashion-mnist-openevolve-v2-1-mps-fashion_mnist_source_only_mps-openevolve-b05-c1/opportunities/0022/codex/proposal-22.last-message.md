MECHANISM: Mild confidence-regularized cross-entropy

HYPOTHESIS: Applying 0.02 label smoothing will exceed 9,240 correct predictions by retaining the successful 3.3e-3 optimization trajectory while reducing brittle, overconfident decision boundaries.

INTENDED_EDIT: Add 0.02 label smoothing to the training cross-entropy without changing architecture, exposure, schedule, or runtime materially.

EVIDENCE: Raising the peak learning rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened validation cross-entropy from 0.21392 to 0.21460, motivating a small confidence regularizer that preserves the stronger optimizer setting.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE