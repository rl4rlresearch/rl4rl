MECHANISM: Moderately increased target smoothing

HYPOTHESIS: Increasing label smoothing from 0.02 to 0.03 will exceed 9,290 correct predictions by extending the accuracy-improving regularization trend observed between 0.01 and 0.02.

INTENDED_EDIT: Raise cross-entropy label smoothing from 2% to 3% while preserving the verified architecture, augmentation, optimizer, schedule, dropout, and EMA ensemble.

EVIDENCE: Reducing label smoothing from 0.02 to 0.01 lowered validation correct from 9,290 to 9,279 despite improving cross-entropy, directly motivating a controlled test in the opposite direction.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
>>>>>>> REPLACE