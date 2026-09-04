MECHANISM: Reduced target smoothing for faster class separation

HYPOTHESIS: Lowering label smoothing from 0.03 to 0.01 will increase validation_correct above 9,252 by strengthening class-discriminative gradients during the fixed two exposures while retaining enough regularization for generalization.

INTENDED_EDIT: Reduce only the training loss label-smoothing coefficient, leaving the successful architecture, schedule, augmentation, late averaging, ten-view ensemble, and 1.10 calibration unchanged.

EVIDENCE: The successful ensemble remained underconfident—raising its scale from 1.05 to 1.10 preserved 9,252 predictions while lowering cross-entropy—so the current 0.03-smoothed targets are a grounded training-side source of suppressed confidence; repeated inference-only changes have supplied no accuracy gain.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.01)
>>>>>>> REPLACE