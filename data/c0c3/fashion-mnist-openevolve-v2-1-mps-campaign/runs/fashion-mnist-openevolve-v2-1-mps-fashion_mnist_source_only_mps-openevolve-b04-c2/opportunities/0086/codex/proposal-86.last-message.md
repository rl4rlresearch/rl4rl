MECHANISM: Lower-side target-smoothing refinement

HYPOTHESIS: Label smoothing of 0.0225 will exceed 9,281 correct predictions by probing between the strong 0.023 result and the weaker 0.02 baseline, while avoiding the sharp degradation observed at 0.0235.

INTENDED_EDIT: Increase only training-time label smoothing from 0.02 to 0.0225; preserve the architecture, optimizer, schedule, and evaluation calibration.

EVIDENCE: Smoothing 0.023 achieved the best result with 9,281 correct, versus 9,279 at 0.02, while 0.0235 fell to 9,252; this brackets the promising region on the lower side of 0.023.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.0225)
>>>>>>> REPLACE