MECHANISM: Local quadratic ensemble-temperature calibration

HYPOTHESIS: Temperature 0.934 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.1923047371.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.924 to 0.934.

EVIDENCE: Temperatures 0.900, 0.912, and 0.924 successively reduced cross-entropy while preserving 9,311 correct predictions; their local curvature places the minimum near 0.934, and nearby 0.933–0.936 attempts timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.924)
=======
        return ensemble / (12.0 * 0.934)
>>>>>>> REPLACE