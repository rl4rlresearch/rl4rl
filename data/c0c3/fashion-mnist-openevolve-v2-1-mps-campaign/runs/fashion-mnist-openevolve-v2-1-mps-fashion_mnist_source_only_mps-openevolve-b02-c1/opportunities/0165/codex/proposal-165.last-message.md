MECHANISM: Quadratic-fit ensemble temperature calibration

HYPOTHESIS: Temperature 0.9351 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.9350 to 0.9351.

EVIDENCE: Verified temperatures from 0.9330 through 0.9350 preserved exactly 9,311 correct predictions while showing diminishing cross-entropy improvements; interpolation of those measurements places the apparent minimum near 0.9351, while the 0.93525 and 0.9355 attempts timed out without contradictory metrics.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9350)
=======
        return ensemble / (12.0 * 0.9351)
>>>>>>> REPLACE