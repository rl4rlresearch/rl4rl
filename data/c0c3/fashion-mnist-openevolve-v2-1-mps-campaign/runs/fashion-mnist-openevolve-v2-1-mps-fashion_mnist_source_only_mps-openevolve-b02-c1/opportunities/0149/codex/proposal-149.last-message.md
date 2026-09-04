MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9320 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922884567.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.9315 to 0.9320.

EVIDENCE: Temperatures 0.9300, 0.9305, 0.9310, and 0.9315 successively reduced cross-entropy while preserving exactly 9,311 correct predictions, motivating the smallest continuation toward the evidence-fitted minimum near 0.934.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9315)
=======
        return ensemble / (12.0 * 0.9320)
>>>>>>> REPLACE