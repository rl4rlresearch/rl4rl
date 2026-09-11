MECHANISM: Conservative local temperature calibration

HYPOTHESIS: Temperature 0.930 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.1923047371.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.924 to 0.930.

EVIDENCE: Temperatures 0.900, 0.912, and 0.924 successively improved cross-entropy while preserving exactly 9,311 correct predictions; 0.930 is a conservative step toward the fitted minimum near 0.934, whose attempted verifications timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.924)
=======
        return ensemble / (12.0 * 0.930)
>>>>>>> REPLACE