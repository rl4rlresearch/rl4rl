MECHANISM: Evidence-fitted ensemble temperature calibration

HYPOTHESIS: Temperature 0.9350 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922867332.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9340 to 0.9350.

EVIDENCE: Every verified increase from 0.9300 through 0.9340 reduced cross-entropy without changing predictions, and the diminishing improvements place the apparent calibration minimum near 0.9350; the 0.9345 attempts timed out without contradictory metrics.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9340)
=======
        return ensemble / (12.0 * 0.9350)
>>>>>>> REPLACE