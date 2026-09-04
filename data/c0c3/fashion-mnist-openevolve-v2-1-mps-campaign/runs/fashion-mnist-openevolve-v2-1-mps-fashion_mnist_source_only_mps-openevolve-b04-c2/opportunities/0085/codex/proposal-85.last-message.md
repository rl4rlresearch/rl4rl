MECHANISM: Local target-smoothing refinement

HYPOTHESIS: Label smoothing of 0.0235 will exceed 9,281 correct predictions by probing the local accuracy maximum implied by the verified 0.02, 0.023, and 0.03 results.

INTENDED_EDIT: Increase only training-time label smoothing from 0.023 to 0.0235, preserving the best-known architecture, optimizer, schedule, and evaluation calibration.

EVIDENCE: Smoothing 0.02 produced 9,279 correct, 0.023 improved this to 9,281, and 0.03 declined to 9,274; local quadratic interpolation places the estimated maximum near 0.0235.

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.023)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.0235)
>>>>>>> REPLACE