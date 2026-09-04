MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9305 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.1922903751.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9300 to 0.9305.

EVIDENCE: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy with identical correct counts; 0.9305 is the smallest untested step toward the fitted minimum near 0.933–0.934.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.930)
=======
        return ensemble / (12.0 * 0.9305)
>>>>>>> REPLACE