MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9325 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922879642.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9320 to 0.9325.

EVIDENCE: Temperatures 0.9300, 0.9305, 0.9310, 0.9315, and 0.9320 successively reduced cross-entropy while preserving exactly 9,311 correct predictions; 0.9325 is the smallest continuation toward the evidence-fitted minimum near 0.934.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9320)
=======
        return ensemble / (12.0 * 0.9325)
>>>>>>> REPLACE