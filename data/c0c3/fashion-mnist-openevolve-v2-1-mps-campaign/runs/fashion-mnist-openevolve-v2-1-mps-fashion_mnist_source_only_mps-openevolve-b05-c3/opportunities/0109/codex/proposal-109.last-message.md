MECHANISM: Reduced soft-target regularization

HYPOTHESIS: Reducing label smoothing from 0.02 to 0.01 will exceed 9,206 correct predictions by strengthening class margins while retaining mild regularization.

INTENDED_EDIT: Use 0.01 label smoothing throughout training; leave the validated architecture, augmentation, split EMA, schedule, and 1.05× evaluation sharpening unchanged.

EVIDENCE: The verified 1.05× sharpening preserved 9,206 predictions while lowering cross-entropy from 0.235889 to 0.230772, indicating underconfidence; prior hard-label and zero-ending annealing trials timed out and supplied no contrary performance evidence, motivating a conservative intermediate smoothing reduction.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.01)
>>>>>>> REPLACE