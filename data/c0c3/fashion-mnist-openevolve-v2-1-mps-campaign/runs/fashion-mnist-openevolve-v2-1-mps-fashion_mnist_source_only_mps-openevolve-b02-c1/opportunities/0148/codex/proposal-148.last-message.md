MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9315 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922890224.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9310 to 0.9315.

EVIDENCE: Temperatures 0.9300, 0.9305, and 0.9310 successively reduced cross-entropy with unchanged argmax predictions; the previous 0.9315 attempt timed out and supplied no contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9310)
=======
        return ensemble / (12.0 * 0.9315)
>>>>>>> REPLACE