MECHANISM: Mild post-ensemble temperature calibration

HYPOTHESIS: Temperature 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Slightly soften evaluation logits without changing training, parameters, runtime, or predicted classes.

EVIDENCE: The current design has the best correct count, while stronger tail averaging lowered cross-entropy but damaged accuracy; positive temperature scaling targets only the tie-break metric. The prior 1.05 attempt timed out, leaving this conservative setting untested.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.03
>>>>>>> REPLACE