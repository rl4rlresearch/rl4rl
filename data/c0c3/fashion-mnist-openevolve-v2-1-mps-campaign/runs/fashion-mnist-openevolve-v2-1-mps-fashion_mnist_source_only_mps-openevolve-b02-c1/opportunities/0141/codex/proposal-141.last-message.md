MECHANISM: Bracketed ensemble temperature calibration

HYPOTHESIS: Temperature 0.932 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922903751.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.930 to 0.932.

EVIDENCE: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy with unchanged argmax predictions; 0.932 is a conservative step toward the fitted minimum near 0.933–0.934, whose attempted verifications timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.930)
=======
        return ensemble / (12.0 * 0.932)
>>>>>>> REPLACE