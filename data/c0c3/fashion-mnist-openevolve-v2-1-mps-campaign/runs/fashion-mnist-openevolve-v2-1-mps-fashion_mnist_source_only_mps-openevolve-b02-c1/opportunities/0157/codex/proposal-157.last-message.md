MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9340 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922869278.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9335 to 0.9340.

EVIDENCE: Temperatures 0.9300 through 0.9335 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; 0.9340 is the smallest continuation toward the evidence-fitted minimum.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9335)
=======
        return ensemble / (12.0 * 0.9340)
>>>>>>> REPLACE