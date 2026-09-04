MECHANISM: Evidence-fitted ensemble temperature calibration

HYPOTHESIS: Temperature 0.9355 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9350 to 0.9355.

EVIDENCE: Every verified increase from 0.9300 through 0.9350 reduced cross-entropy without changing argmax predictions; the diminishing gains place the apparent minimum just above 0.9350.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9350)
=======
        return ensemble / (12.0 * 0.9355)
>>>>>>> REPLACE