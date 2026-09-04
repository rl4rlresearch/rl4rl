MECHANISM: Evidence-fitted ensemble temperature calibration

HYPOTHESIS: Temperature 0.93525 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.9350 to 0.93525.

EVIDENCE: Every verified increase through 0.9350 reduced cross-entropy without changing predictions, while diminishing gains and the timed-out 0.9355 trial motivate testing the safer midpoint.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9350)
=======
        return ensemble / (12.0 * 0.93525)
>>>>>>> REPLACE