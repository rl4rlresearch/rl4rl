MECHANISM: Reduced target smoothing

HYPOTHESIS: Halving label smoothing from 0.02 to 0.01 will sharpen class-boundary learning within the fixed two-pass budget and increase validation_correct above 9,279.

INTENDED_EDIT: Reduce only training-time cross-entropy label smoothing; preserve the architecture, optimizer, schedule, and best-known evaluation calibration.

EVIDENCE: Evaluation calibration has plateaued at 9,279 correct, while augmentation and residual refinement reduced accuracy; the verified scale above 1.0 also indicates softened logits, motivating a conservative prediction-changing reduction in target smoothing.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.01)
>>>>>>> REPLACE