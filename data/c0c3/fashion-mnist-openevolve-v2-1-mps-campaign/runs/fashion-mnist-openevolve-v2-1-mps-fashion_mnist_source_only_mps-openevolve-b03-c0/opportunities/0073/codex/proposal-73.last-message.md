MECHANISM: Slightly stronger fixed label smoothing

HYPOTHESIS: Increasing fixed label smoothing from 0.04 to 0.05 will exceed 9,319 correct predictions by strengthening the regularization that was lost when late-stage smoothing taper reduced accuracy to 9,307.

INTENDED_EDIT: Use 0.05 label smoothing throughout training while preserving the verified architecture, optimizer, schedule, runtime profile, and evaluation calibration.

EVIDENCE: Tapering label smoothing toward zero reduced validation correctness from 9,319 to 9,307, while EMA also failed to recover those errors; this motivates testing a small increase in the successful fixed-smoothing regime.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
>>>>>>> REPLACE