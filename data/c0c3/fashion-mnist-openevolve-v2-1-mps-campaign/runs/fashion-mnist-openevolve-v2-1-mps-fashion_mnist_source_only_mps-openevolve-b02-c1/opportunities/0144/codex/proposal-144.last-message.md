MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9305 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922903751.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9300 to 0.9305.

EVIDENCE: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy with unchanged argmax predictions; the prior 0.9305 verification timed out and therefore provides no contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.930)
=======
        return ensemble / (12.0 * 0.9305)
>>>>>>> REPLACE