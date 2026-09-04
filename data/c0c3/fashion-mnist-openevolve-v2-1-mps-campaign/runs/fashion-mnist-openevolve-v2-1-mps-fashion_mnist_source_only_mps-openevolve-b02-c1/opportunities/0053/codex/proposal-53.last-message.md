MECHANISM: Mild post-ensemble temperature calibration

HYPOTHESIS: Temperature 1.03 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.197145.

INTENDED_EDIT: Divide the final evaluation ensemble logits by 1.03; training and predicted classes remain unchanged.

EVIDENCE: Stronger tail averaging reduced cross-entropy to 0.197021 but lost predictions, so calibration should target the tie-break metric without moving decision boundaries; the prior 1.03 attempt timed out and remains untested.

<<<<<<< SEARCH
        return ensemble - math.log(12.0)
=======
        return (ensemble - math.log(12.0)) / 1.03
>>>>>>> REPLACE