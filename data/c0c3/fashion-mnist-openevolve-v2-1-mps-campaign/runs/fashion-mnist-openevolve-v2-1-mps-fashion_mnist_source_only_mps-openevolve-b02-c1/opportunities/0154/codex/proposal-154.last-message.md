MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9330 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922875435.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9330.

EVIDENCE: Temperatures 0.9300 through 0.9325 successively reduced cross-entropy without changing argmax predictions; the previous 0.9330 verification timed out and supplied no contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9325)
=======
        return ensemble / (12.0 * 0.9330)
>>>>>>> REPLACE