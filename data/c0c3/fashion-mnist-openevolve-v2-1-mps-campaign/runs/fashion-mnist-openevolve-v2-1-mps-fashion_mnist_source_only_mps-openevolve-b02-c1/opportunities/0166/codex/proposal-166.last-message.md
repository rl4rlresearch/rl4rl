MECHANISM: Quadratic-fit ensemble temperature calibration

HYPOTHESIS: Temperature 0.9351 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922865562.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.9350 to 0.9351.

EVIDENCE: Every verified increase from 0.9330 through 0.9350 reduced cross-entropy without changing predictions; interpolation places the minimum near 0.9351, whose prior verification timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9350)
=======
        return ensemble / (12.0 * 0.9351)
>>>>>>> REPLACE